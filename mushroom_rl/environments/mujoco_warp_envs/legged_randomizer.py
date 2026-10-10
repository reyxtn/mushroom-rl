"""
Domain randomization for legged robots in MuJoCo Warp.

A port of the IsaacSim ``QuadrupedRandomizer`` (new_isaac branch) to the warp
backend. The parameter names and the seen/unseen split are kept, so the
documentation of the Isaac version applies; what differs is where the values
go. In Isaac the PD control law is written in torch, so gains, motor strength
and torque limits are plain tensors. Here the PD runs inside MuJoCo's
position actuators, so those quantities, together with the trunk mass, the
centre of mass, the ground friction and the joint dynamics, are written into
the mujoco_warp model arrays, which the environment allocates with one entry
per world (``model_batch_fields`` of :class:`MuJoCoWarp`).

Every write is in place, into torch views of the warp arrays, so it is safe
under CUDA graph capture: the captured kernels keep reading the same memory.

Not ported, because the warp model has no per-world counterpart: the joint
velocity limit (``joint_velocity_factor``), which Isaac only used in a reward
term the warp tasks do not have. ``body_invweight0`` and ``dof_invweight0``
(constraint regularisation weights derived from the nominal mass) are left
nominal when the trunk mass changes; for offsets of a few kilograms on a
70 kg robot the effect is below the solver tolerance.

"""

import math

import torch


def _uniform(bounds, shape, device):
    """Uniform draw between ``bounds``, drawing nothing when they coincide."""
    lower, upper = bounds
    if lower == upper:
        return torch.full(shape, float(lower), device=device)
    return torch.rand(shape, device=device) * (upper - lower) + lower


def _bernoulli(probability, n, device):
    if probability <= 0.0 or probability >= 1.0:
        return torch.full((n,), probability >= 1.0, dtype=torch.bool, device=device)
    return torch.rand(n, device=device) < probability


class LeggedRandomizationParams:
    """
    The domain randomization ranges of a legged robot. Same accessor API as
    the Isaac ``QuadrupedRandomizationParams``; the table below lists every
    parameter, its default and its meaning.

    Disturbance, friction, latency and reset:

    ``push_probability``        1/750   per-step chance an environment is pushed,
                                        when no interval is set
    ``push_interval_range``     None    (lo, hi) seconds between two pushes of the
                                        same environment; overrides the probability
    ``push_min_episode_length`` 50      steps an environment must be alive to be pushed
    ``push_max_velocity``       1.0     half-width of the planar velocity a push sets
    ``ground_friction_factor``  5/9     spread of the foot friction around the model's
                                        value, one factor per environment
    ``max_delay_steps``         4       largest number of intermediate steps an action
                                        is delayed by; drawn uniformly in [0, max]
    ``mixed_chance``            0.0     chance an episode redraws the delay at every step
    ``reset_yaw_range``         pi      half-width of the random yaw at reset
    ``reset_joint_noise``       0.05    half-width of the joint angle noise at reset, rad
    ``reset_base_velocity``     0.2     half-width of the planar base velocity at reset

    Robot and actuation ranges:

    ``add_trunk_mass``          (-2, 4)        offset on the trunk mass, kg (drawn once)
    ``add_com_displacement``    (-0.05, 0.05)  offset on each axis of the trunk COM, m
    ``p_gain_scale``            (0.85, 1.15)   factor on the proportional gain (drawn once)
    ``d_gain_scale``            (0.85, 1.15)   factor on the derivative gain (drawn once)
    ``torque_limit_factor``     0.0            spread of the actuator force limit
    ``add_scaling_factor``      (0, 0)         offset on the action scale
    ``add_joint_nominal_position`` (0, 0)      offset on the pose actions are relative to
    ``stay_at_default_percentage`` 1.0         chance the three joint properties below stay
                                               nominal instead of being drawn
    ``joint_damping``           (0.0, 0.3)     absolute range of the dof damping
    ``joint_stiffness``         (0.0, 0.5)     absolute range of the joint stiffness
    ``joint_armature``          (0.009, 0.023) absolute range of the dof armature
    ``joint_friction_factor``   0.0            spread of the dof friction loss

    Unseen noise, each a half-width around 1 (0 disables it):

    ``trunk_mass_factor``, ``trunk_com_factor``, ``joint_damping_factor``,
    ``joint_stiffness_factor``, ``joint_armature_factor``, ``p_gain_factor``,
    ``d_gain_factor``, ``motor_strength_factor``: all 0.0

    ``position_offset``         0.04    half-width of the encoder miscalibration, rad:
                                        the controller and the policy see the joint
                                        angle offset by this much from the truth

    """

    def __init__(self, **overrides):
        self._values = self._default_values()
        unknown = set(overrides) - set(self._values)
        if unknown:
            raise ValueError(f"unknown randomization parameters: {sorted(unknown)}")
        self._values.update(overrides)

    def __getitem__(self, name):
        return self._values[name]

    def __setitem__(self, name, value):
        if name not in self._values:
            raise ValueError(f"unknown randomization parameter: {name}")
        self._values[name] = value

    def __contains__(self, name):
        return name in self._values

    def as_dict(self):
        return dict(self._values)

    @staticmethod
    def _default_values():
        return dict(
            push_probability=1.0 / 750.0,
            push_interval_range=None,
            push_min_episode_length=50,
            push_max_velocity=1.0,
            ground_friction_factor=5.0 / 9.0,
            max_delay_steps=4,
            mixed_chance=0.0,
            reset_yaw_range=math.pi,
            reset_joint_noise=0.05,
            reset_base_velocity=0.2,
            add_trunk_mass=(-2.0, 4.0),
            add_com_displacement=(-0.05, 0.05),
            p_gain_scale=(0.85, 1.15),
            d_gain_scale=(0.85, 1.15),
            torque_limit_factor=0.0,
            add_scaling_factor=(0.0, 0.0),
            add_joint_nominal_position=(0.0, 0.0),
            stay_at_default_percentage=1.0,
            joint_damping=(0.0, 0.3),
            joint_stiffness=(0.0, 0.5),
            joint_armature=(0.009, 0.023),
            joint_friction_factor=0.0,
            trunk_mass_factor=0.0,
            trunk_com_factor=0.0,
            joint_damping_factor=0.0,
            joint_stiffness_factor=0.0,
            joint_armature_factor=0.0,
            p_gain_factor=0.0,
            d_gain_factor=0.0,
            motor_strength_factor=0.0,
            position_offset=0.04,
        )


class LeggedRandomizer:
    """
    Samples the randomized parameters of every world and writes them into the
    per-world model arrays and into the torch-side control law.

    The robot is described by index tensors into the model, so the class is
    independent of the robot: the trunk body, the foot geoms, the actuators
    (all of them, in actuator order) and the dofs and joints they drive, plus
    the subset of actuators the policy controls.

    Two layers, as in the Isaac version: every parameter has a *seen* value,
    which the environment may expose to the agent as a privileged
    observation, and the simulation runs on an *unseen* value, which is the
    seen one times a noise factor. All ``*_factor`` noises default to zero, so
    by default the two coincide.

    Args:
        n_envs (int): number of worlds;
        views (dict): torch views of the batched model arrays, keyed by the
            mujoco_warp field name. Needed: ``body_mass`` (W, nbody),
            ``body_inertia`` (W, nbody, 3), ``body_ipos`` (W, nbody, 3),
            ``body_subtreemass`` (W, nbody), ``geom_friction`` (W, ngeom, 3),
            ``actuator_gainprm`` (W, nu, 10), ``actuator_biasprm`` (W, nu, 10),
            ``actuator_forcerange`` (W, nu, 2), ``dof_damping`` (W, nv),
            ``dof_armature`` (W, nv), ``dof_frictionloss`` (W, nv),
            ``jnt_stiffness`` (W, njnt). Row 0 must still hold the nominal
            model when the randomizer is built;
        layout (dict): ``trunk_body`` (int), ``trunk_ancestors`` (list of body
            ids whose subtree contains the trunk, trunk included),
            ``foot_geoms`` (long tensor), ``actuators`` (long tensor, all nu
            in order), ``dofs`` (long tensor, dof of each actuator's joint),
            ``joints`` (long tensor, joint of each actuator), ``controlled``
            (long tensor, actuator indices the policy controls);
        default_joint_pos (torch.Tensor): nominal angle of every actuator's
            joint, (nu,);
        action_scale (float): nominal action scaling;
        params (LeggedRandomizationParams, None): the ranges.

    """

    def __init__(self, n_envs, views, layout, default_joint_pos, action_scale, params=None):
        self._n_envs = n_envs
        self._views = views
        self._params = LeggedRandomizationParams() if params is None else params
        dev = views["body_mass"].device
        self._device = dev

        self._trunk = int(layout["trunk_body"])
        self._trunk_ancestors = torch.as_tensor(layout["trunk_ancestors"], device=dev, dtype=torch.long)
        self._feet = layout["foot_geoms"].to(dev).long()
        self._act = layout["actuators"].to(dev).long()
        self._dofs = layout["dofs"].to(dev).long()
        self._jnts = layout["joints"].to(dev).long()
        self._ctrl = layout["controlled"].to(dev).long()
        nu = len(self._act)
        nc = len(self._ctrl)
        self._nu, self._nc = nu, nc

        # Nominal values, read from world 0 before anything is written.
        v = views
        self._default = dict(
            trunk_mass=v["body_mass"][0, self._trunk].clone(),
            trunk_inertia=v["body_inertia"][0, self._trunk].clone(),
            trunk_com=v["body_ipos"][0, self._trunk].clone(),
            subtreemass=v["body_subtreemass"][0].clone(),
            foot_friction=v["geom_friction"][0, self._feet, 0].clone(),
            kp=v["actuator_gainprm"][0, self._act, 0].clone(),
            kd=-v["actuator_biasprm"][0, self._act, 2].clone(),
            forcerange=v["actuator_forcerange"][0, self._act].clone(),
            damping=v["dof_damping"][0, self._dofs].clone(),
            armature=v["dof_armature"][0, self._dofs].clone(),
            frictionloss=v["dof_frictionloss"][0, self._dofs].clone(),
            stiffness=v["jnt_stiffness"][0, self._jnts].clone(),
            joint_nominal_pos=default_joint_pos.to(dev).clone(),
            action_scale=float(action_scale),
        )
        if not torch.allclose(v["actuator_gainprm"][0, self._act, 0], -v["actuator_biasprm"][0, self._act, 1]):
            raise ValueError("actuators are not position servos (gainprm[0] != -biasprm[1])")

        # Seen values: what the control law and the observations work from.
        self._seen = dict(
            mass=v["body_mass"][0].sum().expand(n_envs).clone(),
            p_gain=self._default["kp"].expand(n_envs, nu).clone(),
            d_gain=self._default["kd"].expand(n_envs, nu).clone(),
            torque_limit=self._default["forcerange"][:, 1].expand(n_envs, nu).clone(),
            action_scaling_factor=torch.full((n_envs, nc), self._default["action_scale"], device=dev),
            joint_nominal_position=self._default["joint_nominal_pos"].expand(n_envs, nu).clone(),
        )
        # Unseen factors the simulation actually runs on, all 1 by default.
        self._unseen = dict(
            p_gain=torch.ones(n_envs, nu, device=dev),
            d_gain=torch.ones(n_envs, nu, device=dev),
            motor_strength=torch.ones(n_envs, 1, device=dev),
            position_offset=torch.zeros(n_envs, nu, device=dev),
        )
        self._mixed = torch.zeros(n_envs, dtype=torch.bool, device=dev)
        self._n_delay_steps = torch.zeros(n_envs, dtype=torch.long, device=dev)
        # Interval mode: start every world's push timer at a random phase, so
        # a world that is never reset is not pushed on its first eligible step.
        interval = self._params["push_interval_range"]
        self._time_to_push = (
            _uniform(interval, (n_envs,), dev) if interval is not None else torch.zeros(n_envs, device=dev)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def params(self):
        return self._params

    @property
    def seen_parameters(self):
        return self._seen

    @property
    def default_parameters(self):
        return self._default

    @property
    def position_offset(self):
        """Encoder miscalibration of every actuator's joint, (W, nu)."""
        return self._unseen["position_offset"]

    @property
    def joint_nominal_pos(self):
        return self._seen["joint_nominal_position"]

    @property
    def scaling_factor(self):
        return self._seen["action_scaling_factor"]

    @property
    def delay_steps(self):
        return self._n_delay_steps

    def resample_startup(self, env_indices):
        """
        Draws what describes the robot itself and stays fixed for the whole
        run: trunk mass, centre of mass and the actuator gains.

        """
        idx = self._idx(env_indices)
        n = len(idx)
        if n == 0:
            return
        dev = self._device
        p = self._params

        # Trunk mass and inertia. The inertia scales with the mass, as in Isaac.
        mass = self._default["trunk_mass"] + _uniform(p["add_trunk_mass"], (n,), dev)
        unseen_mass = mass * self._noise_factor(n, "trunk_mass_factor").squeeze(1)
        ratio = unseen_mass / self._default["trunk_mass"]
        v = self._views
        v["body_mass"][idx, self._trunk] = unseen_mass
        v["body_inertia"][idx, self._trunk] = self._default["trunk_inertia"] * ratio.unsqueeze(1)
        delta = unseen_mass - self._default["trunk_mass"]
        v["body_subtreemass"][idx.unsqueeze(1), self._trunk_ancestors.unsqueeze(0)] = (
            self._default["subtreemass"][self._trunk_ancestors].unsqueeze(0) + delta.unsqueeze(1)
        )
        self._seen["mass"][idx] = v["body_mass"][idx].sum(dim=1)

        com = self._default["trunk_com"] + _uniform(p["add_com_displacement"], (n, 3), dev)
        v["body_ipos"][idx, self._trunk] = com * self._noise_factor(n, "trunk_com_factor")

        # Gains: seen scale, unseen noise on top.
        self._seen["p_gain"][idx] = self._default["kp"] * _uniform(p["p_gain_scale"], (n, self._nu), dev)
        self._seen["d_gain"][idx] = self._default["kd"] * _uniform(p["d_gain_scale"], (n, self._nu), dev)
        self._unseen["p_gain"][idx] = self._noise_factor(n, "p_gain_factor")
        self._unseen["d_gain"][idx] = self._noise_factor(n, "d_gain_factor")
        self._write_gains(idx)

    def resample_reset(self, env_indices):
        """
        Draws what varies from episode to episode: joint dynamics, torque
        limit, action scaling, encoder offset, motor strength, latency regime
        and ground friction.

        """
        idx = self._idx(env_indices)
        n = len(idx)
        if n == 0:
            return
        dev = self._device
        p = self._params
        v = self._views
        nu, nc = self._nu, self._nc

        # Joint dynamics: absolute range with probability (1 - stay), nominal
        # otherwise, then unseen noise.
        stay = _bernoulli(p["stay_at_default_percentage"], n, dev)
        for name, nominal_key, field, sel in (
            ("joint_damping", "damping", "dof_damping", self._dofs),
            ("joint_armature", "armature", "dof_armature", self._dofs),
            ("joint_stiffness", "stiffness", "jnt_stiffness", self._jnts),
        ):
            nominal = self._default[nominal_key]
            drawn = _uniform(p[name], (n, nu), dev)
            seen = torch.where(stay.unsqueeze(1), nominal.expand(n, nu), drawn)
            v[field][idx.unsqueeze(1), sel.unsqueeze(0)] = seen * self._noise_factor(n, f"{name}_factor")

        hw = p["joint_friction_factor"]
        v["dof_frictionloss"][idx.unsqueeze(1), self._dofs.unsqueeze(0)] = (
            self._default["frictionloss"] * (1.0 + _uniform((-hw, hw), (n, nu), dev))
        )

        hw = p["torque_limit_factor"]
        limit = self._default["forcerange"][:, 1] * (1.0 + _uniform((-hw, hw), (n, nu), dev))
        self._seen["torque_limit"][idx] = limit
        v["actuator_forcerange"][idx.unsqueeze(1), self._act.unsqueeze(0), 0] = -limit
        v["actuator_forcerange"][idx.unsqueeze(1), self._act.unsqueeze(0), 1] = limit

        # Actuation seen by the control law.
        self._seen["action_scaling_factor"][idx] = self._default["action_scale"] + _uniform(
            p["add_scaling_factor"], (n, nc), dev
        )
        self._seen["joint_nominal_position"][idx] = self._default["joint_nominal_pos"] + _uniform(
            p["add_joint_nominal_position"], (n, nu), dev
        )
        hw = p["position_offset"]
        self._unseen["position_offset"][idx] = _uniform((-hw, hw), (n, nu), dev)
        self._unseen["motor_strength"][idx] = self._noise_factor(n, "motor_strength_factor")
        self._write_gains(idx)

        # Latency regime.
        self._mixed[idx] = _bernoulli(p["mixed_chance"], n, dev)
        self._n_delay_steps[idx] = torch.randint(0, int(p["max_delay_steps"]) + 1, (n,), device=dev)

        # Ground friction: one factor per world, on the foot geoms, whose
        # priority makes them win every foot-floor contact.
        hw = p["ground_friction_factor"]
        factor = 1.0 + _uniform((-hw, hw), (n, 1), dev)
        v["geom_friction"][idx.unsqueeze(1), self._feet.unsqueeze(0), 0] = self._default["foot_friction"] * factor

        if p["push_interval_range"] is not None:
            self._time_to_push[idx] = _uniform(p["push_interval_range"], (n,), dev)

    def reset_to_nominal(self):
        """Puts every world back on the nominal model and control law."""
        v, d = self._views, self._default
        W = self._n_envs
        v["body_mass"][:, self._trunk] = d["trunk_mass"]
        v["body_inertia"][:, self._trunk] = d["trunk_inertia"]
        v["body_ipos"][:, self._trunk] = d["trunk_com"]
        v["body_subtreemass"][:, self._trunk_ancestors] = d["subtreemass"][self._trunk_ancestors]
        v["geom_friction"][:, self._feet, 0] = d["foot_friction"]
        v["actuator_forcerange"][:, self._act] = d["forcerange"]
        v["dof_damping"][:, self._dofs] = d["damping"]
        v["dof_armature"][:, self._dofs] = d["armature"]
        v["dof_frictionloss"][:, self._dofs] = d["frictionloss"]
        v["jnt_stiffness"][:, self._jnts] = d["stiffness"]

        self._seen["mass"][:] = v["body_mass"][0].sum()
        self._seen["p_gain"][:] = d["kp"]
        self._seen["d_gain"][:] = d["kd"]
        self._seen["torque_limit"][:] = d["forcerange"][:, 1]
        self._seen["action_scaling_factor"][:] = d["action_scale"]
        self._seen["joint_nominal_position"][:] = d["joint_nominal_pos"]
        for name in ("p_gain", "d_gain", "motor_strength"):
            self._unseen[name][:] = 1.0
        self._unseen["position_offset"][:] = 0.0
        self._mixed[:] = False
        self._n_delay_steps[:] = 0
        self._time_to_push[:] = 0.0
        self._write_gains(torch.arange(W, device=self._device))

    def sample_disturbance(self, episode_length, dt):
        """
        Returns the indices of the worlds to push this step and the planar
        velocity to push each with, (k,) and (k, 2).

        """
        p = self._params
        W = self._n_envs
        if p["push_interval_range"] is None:
            do_push = _bernoulli(p["push_probability"], W, self._device)
        else:
            self._time_to_push -= dt
            do_push = self._time_to_push <= 0.0
            self._time_to_push = torch.where(
                do_push, _uniform(p["push_interval_range"], (W,), self._device), self._time_to_push
            )
        do_push &= episode_length > p["push_min_episode_length"]
        idx = torch.nonzero(do_push, as_tuple=True)[0]
        vmax = p["push_max_velocity"]
        return idx, _uniform((-vmax, vmax), (len(idx), 2), self._device)

    def sample_latency(self):
        """Delay of the next action in intermediate steps, (W,) long."""
        # No `.any()` test: that would be a host sync at every intermediate
        # step. With mixed_chance = 0 (the default) nothing runs at all.
        if self._params["mixed_chance"] > 0.0:
            redrawn = torch.randint(
                0, int(self._params["max_delay_steps"]) + 1, (self._n_envs,), device=self._device
            )
            self._n_delay_steps = torch.where(self._mixed, redrawn, self._n_delay_steps)
        return self._n_delay_steps

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _idx(self, env_indices):
        if isinstance(env_indices, torch.Tensor):
            return env_indices.to(self._device).long()
        return torch.as_tensor(env_indices, device=self._device, dtype=torch.long)

    def _noise_factor(self, n, name):
        hw = self._params[name]
        return _uniform((1.0 - hw, 1.0 + hw), (n, 1), self._device)

    def _write_gains(self, idx):
        """
        Position servo in MuJoCo: force = kp * ctrl - kp * q - kd * qd, that is
        gainprm[0] = kp, biasprm[1] = -kp, biasprm[2] = -kd. Motor strength
        scales the whole force, so it multiplies all three.

        """
        kp = self._seen["p_gain"][idx] * self._unseen["p_gain"][idx] * self._unseen["motor_strength"][idx]
        kd = self._seen["d_gain"][idx] * self._unseen["d_gain"][idx] * self._unseen["motor_strength"][idx]
        rows = idx.unsqueeze(1)
        cols = self._act.unsqueeze(0)
        self._views["actuator_gainprm"][rows, cols, 0] = kp
        self._views["actuator_biasprm"][rows, cols, 1] = -kp
        self._views["actuator_biasprm"][rows, cols, 2] = -kd

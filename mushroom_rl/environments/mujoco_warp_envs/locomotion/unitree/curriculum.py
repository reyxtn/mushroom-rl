"""
The within-stage schedule the legged training scripts tighten the task
along: velocity command ranges, actuation latency ceiling and tracking
tolerance, as fractions of the stage's policy updates.

Ported from the new_isaac ``go2_curriculum.py``; the environment only accepts new values,
when to change them is decided here.

Used by ``examples/environments/mujoco_warp/h2_curriculum.py`` and
``go2_curriculum.py``.

"""


class Curriculum:
    """
    The schedule a stage is tightened along, applied after every policy
    update (a ``Core`` fit callback). Thresholds are fractions of the stage's
    updates rather than step counts, so they do not depend on the epoch
    count.

    ``complete`` is True once every schedule has reached its final value.
    The stage's best checkpoint should only be looked for from then on:
    before, the commands are easier and the tracking kernel wider, so the
    returns are higher for reasons that have nothing to do with the policy.

    With ``enabled=False`` every schedule sits at its final value from the
    start and ``complete`` is True throughout.

    Args:
        mdp: the environment; needs the ``command_ranges``, ``tracking_sigma``
            and ``max_delay`` setters, ``max_delay_steps``,
            ``max_delay_steps_limit`` and ``domain_randomization``;
        n_fits (int): policy updates in the stage;
        command_stages (tuple): fractions at which the command ranges switch
            to the next entry of ``command_ranges``;
        command_ranges (list): dicts for the ``command_ranges`` setter, or
            None to leave the ranges alone;
        delay_stages (tuple): fractions at which the latency ceiling switches
            to the next entry of ``delays``;
        delays (list): latency ceilings in seconds, clamped to the
            environment's limit;
        sigma_window (tuple, None): fractions between which the tracking
            sigma is interpolated from ``sigmas[0]`` to ``sigmas[1]``;
        sigmas (tuple, None): the two ends of the sigma schedule;
        enabled (bool, True): see above.

    """

    def __init__(
        self,
        mdp,
        n_fits,
        command_stages,
        command_ranges,
        delay_stages,
        delays,
        sigma_window,
        sigmas,
        enabled=True,
    ):
        self._mdp = mdp
        self._n_fits = max(int(n_fits), 1)
        self._command_stages = tuple(command_stages)
        self._command_ranges = list(command_ranges)
        self._delay_stages = tuple(delay_stages)
        # The ring buffer was sized at construction; a schedule entry above
        # its ceiling would be refused by the max_delay setter.
        limit = mdp.max_delay_steps_limit * mdp._step_dt
        self._delays = [min(float(d), limit) for d in delays]
        self._sigma_window = sigma_window
        self._sigmas = sigmas
        self._enabled = enabled
        self._fit = 0
        thresholds = list(self._command_stages) + list(self._delay_stages)
        if sigma_window is not None:
            thresholds.append(sigma_window[1])
        self._complete_at = max(thresholds) if thresholds else 0.0
        self.apply()

    def __call__(self, dataset):
        self._fit += 1
        self.apply()

    @property
    def fraction(self):
        if not self._enabled:
            return 1.0
        return min(self._fit / self._n_fits, 1.0)

    @property
    def complete_at(self):
        return self._complete_at

    @property
    def complete(self):
        return self.fraction >= self._complete_at

    @property
    def command_stage(self):
        return sum(self.fraction >= t for t in self._command_stages)

    @property
    def delay_stage(self):
        return sum(self.fraction >= t for t in self._delay_stages)

    @property
    def delays(self):
        """The latency ceilings of the schedule, in seconds."""
        return list(self._delays)

    @property
    def progress(self):
        if self._sigma_window is None:
            return 1.0
        start, end = self._sigma_window
        return min(max((self.fraction - start) / (end - start), 0.0), 1.0)

    def apply(self):
        ranges = self._command_ranges[
            min(self.command_stage, len(self._command_ranges) - 1)
        ]
        if ranges is not None:
            self._mdp.command_ranges = ranges
        if self._mdp.domain_randomization:
            self._mdp.max_delay = self._delays[
                min(self.delay_stage, len(self._delays) - 1)
            ]
        if self._sigmas is not None:
            start, end = self._sigmas
            self._mdp.tracking_sigma = start + self.progress * (end - start)

    def state(self):
        return dict(
            cmd_stage=self.command_stage,
            delay_steps=self._mdp.max_delay_steps,
            sigma=self._mdp.tracking_sigma,
            progress=self.progress,
            complete=float(self.complete),
        )

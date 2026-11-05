"""
The schedule the task is tightened along, applied after every policy update.
A port of the ``Curriculum`` of new_isaac's ``go2_curriculum.py``: same
thresholds (counted in control steps per environment, i.e. dataset rows), same
staged command ranges and delay ceilings, same linear interpolation of the
tracking tolerances.

Two additions that do not change the schedule: the delays are given in
seconds (Isaac counts 5 ms physics steps; the warp env converts with its own
step length), and ``complete`` / ``state()`` for the best-checkpoint logic and
the logs of the training script.

"""


class Curriculum:
    """
    Args:
        mdp (Go2Walk): the environment the schedule is applied to;
        command_steps (list): the step counts at which the next stage
            begins, one fewer than the stages;
        command_ranges (list): the linear velocity command range of each
            stage: a ``(low, high)`` tuple applied to x and y, a dict for the
            generator's ``command_ranges`` setter, or None to leave them;
        delays (list): the ceiling on the actuation delay of each stage, in
            seconds;
        tracking_steps (tuple): the step counts the tracking tolerance is
            interpolated between;
        tracking_stds (tuple, None): the tolerances (dicts for
            ``mdp.tracking_stds``) at the two ends of that interpolation, or
            None to leave the tolerance alone;
        enabled (bool, True): with False, every schedule sits at its final
            value from the start.

    """

    def __init__(self, mdp, command_steps, command_ranges, delays, tracking_steps, tracking_stds, enabled=True):
        self._mdp = mdp
        self._command_steps = list(command_steps)
        self._command_ranges = list(command_ranges)
        self._delays = list(delays)
        self._tracking_steps = tuple(tracking_steps)
        self._tracking_stds = tracking_stds
        self._enabled = enabled

        self._step = 0
        self.apply()

    def __call__(self, dataset):
        # a vectorized dataset holds one row per control step, whichever
        # environment it came from
        self._step += len(dataset)
        self.apply()

    def apply(self):
        ranges = self._command_ranges[self.stage]
        if isinstance(ranges, (tuple, list)):
            low, high = ranges
            ranges = dict(lin_vel_x=(low, high), lin_vel_y=(low, high))
        if ranges is not None:
            self._mdp.command_generator.command_ranges = ranges
        if self._mdp.domain_randomization:
            self._mdp.max_delay = min(self._delays[self.stage], self._mdp.max_delay_limit)

        if self._tracking_stds is not None:
            start, end = self._tracking_stds
            self._mdp.tracking_stds = {name: start[name] + self.progress * (end[name] - start[name]) for name in start}

    @property
    def step(self):
        return self._step

    @property
    def stage(self):
        if not self._enabled:
            return len(self._command_steps)
        return sum(self._step >= threshold for threshold in self._command_steps)

    @property
    def progress(self):
        if not self._enabled or self._tracking_stds is None:
            return 1.
        start, end = self._tracking_steps
        return min(max((self._step - start) / (end - start), 0.), 1.)

    @property
    def complete_at(self):
        """The step count from which every schedule sits at its final value."""
        steps = list(self._command_steps)
        if self._tracking_stds is not None:
            steps.append(self._tracking_steps[1])
        return max(steps) if steps else 0

    @property
    def complete(self):
        return not self._enabled or self._step >= self.complete_at

    def state(self):
        return dict(
            stage=self.stage,
            progress=self.progress,
            delay_steps=self._mdp.max_delay_steps,
            complete=float(self.complete),
        )

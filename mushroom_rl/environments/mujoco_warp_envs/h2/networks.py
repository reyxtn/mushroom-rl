import torch

from mushroom_rl.approximators.parametric.networks import ActorNetwork


class MaskedActorNetwork(ActorNetwork):
    """
    An actor network reading only part of the observation, so that the policy
    is trained on what the real robot can measure while the critic, which only
    ever runs in simulation, keeps the privileged entries (the randomized
    latency, the encoder offset, ...).

    Same as the ``PolicyNetwork`` of the new_isaac Go2 curriculum script, kept
    in the package so that a saved agent can be loaded by any script, not only
    the one that defined the class.

    Args:
        input_shape (tuple): shape of the full observation, with the stacked
            history as leading axes if there is one;
        output_shape (tuple): shape of the action;
        observed_indices (torch.Tensor): the entries of the observation the
            policy is allowed to see;
        **kwargs: other parameters of :class:`ActorNetwork`.

    """

    def __init__(self, input_shape, output_shape, observed_indices, **kwargs):
        observed_indices = torch.as_tensor(observed_indices, dtype=torch.long)
        super().__init__(tuple(input_shape[:-1]) + (len(observed_indices),), output_shape, **kwargs)
        self.register_buffer("_observed_indices", observed_indices)

    def forward(self, state, **kwargs):
        return super().forward(state[..., self._observed_indices], **kwargs)

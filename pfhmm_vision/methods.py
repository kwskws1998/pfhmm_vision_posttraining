"""Interchangeable reward processors with the same action-conditioned input interface."""

import numpy as np

from .filter import CauseFilter


class ObservedReward:
    """Standard REINFORCE control using the observed binary reward without denoising."""

    def process(self, rewards, action_correctness):
        """Return unchanged bits with no access to oracle labels, phases, or contexts."""
        return {"training_reward": np.asarray(rewards, dtype=np.float64).copy()}

    def state_dict(self):
        """Expose the empty state of a stateless baseline."""
        return {}

    def load_state_dict(self, state):
        """Reject unexpected state instead of silently changing baseline behavior."""
        if state:
            raise ValueError("The observed-reward baseline has no inference state")


class InferredReward:
    """Apply the proposed cause filter to each chronological feedback observation."""

    def __init__(self, config, seed):
        self.filter = CauseFilter(config, seed)

    def process(self, rewards, action_correctness):
        """Use action probabilities and feedback only; latent ground truth is inaccessible."""
        if action_correctness.shape != (len(rewards), 2):
            raise ValueError("Expected one correctness estimate per example and latent context")
        records = [self.filter.update(int(reward), q)
                   for reward, q in zip(rewards, action_correctness)]
        output = {key: np.asarray([row[key] for row in records]) for key in records[0]}
        output["training_reward"] = output.pop("clean_reward")
        return output

    def state_dict(self):
        """Checkpoint all inference state including its independent RNG."""
        return self.filter.state_dict()

    def load_state_dict(self, state):
        """Resume without resetting inferred noise or context beliefs."""
        self.filter.load_state_dict(state)


def make_method(name, config, seed):
    """Construct an explicit registered method instead of embedding branches in the optimizer."""
    providers = {"baseline": lambda: ObservedReward(),
                 "pfhmm": lambda: InferredReward(config, seed)}
    if name not in providers:
        raise ValueError(f"Unknown reward processor: {name}")
    return providers[name]()

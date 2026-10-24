"""A private reward oracle and common random numbers for matched contextual-bandit arms."""

import numpy as np
import torch


class FeedbackOracle:
    """Keep adaptation labels and true causes outside the learner-facing feedback interface."""

    def __init__(self, labels, phases, batch_size, seed):
        self._labels = np.asarray(labels, dtype=np.int64)
        if self._labels.ndim != 1 or len(self._labels) == 0 or np.any(self._labels < 0):
            raise ValueError("A nonempty hidden class-label vector is required")
        self._classes = int(self._labels.max()) + 1
        self._next_step = 0
        self.batch_size = batch_size
        sequence = np.random.SeedSequence(seed)
        index_rng, context_rng, flip_rng, action_rng = [np.random.default_rng(s)
                                                       for s in sequence.spawn(4)]
        count = sum(p.updates for p in phases) * batch_size
        self.indices = index_rng.integers(0, len(labels), size=count)
        self._context = np.empty(count, dtype=np.int64)
        self._flip_rates, self._volatility = np.empty(count), np.empty(count)
        self._flips = flip_rng.random(count)
        self.action_uniforms = action_rng.random(count)
        self._phase = np.empty(count, dtype=np.int64)
        start, context = 0, int(context_rng.integers(0, 2))
        for phase_id, phase in enumerate(phases):
            end = start + phase.updates * batch_size
            switches = context_rng.random(end - start) < phase.volatility
            for offset, switch in enumerate(switches):
                context ^= int(switch)
                self._context[start + offset] = context
            self._flip_rates[start:end] = phase.feedback_flip
            self._volatility[start:end] = phase.volatility
            self._phase[start:end] = phase_id
            start = end
        self._rewards = np.full(count, -1, dtype=np.int8)
        self._actions = np.full(count, -1, dtype=np.int64)

    def observation(self, step, cache):
        """Return cached observations and sampling uniforms without hidden state or labels."""
        selection = self._selection(step)
        context = torch.from_numpy(self._context[selection])
        indices = torch.from_numpy(self.indices[selection])
        features = cache["features"][context, indices]
        logits = cache["logits"][context, indices]
        return features, logits, self.action_uniforms[selection].copy()

    def feedback(self, step, actions):
        """Reveal exactly one possibly flipped correctness bit per selected action."""
        selection = self._selection(step)
        actions = np.asarray(actions)
        if actions.shape != (self.batch_size,) or not np.issubdtype(actions.dtype, np.integer):
            raise ValueError("One action per budgeted observation is required")
        if np.any(actions < 0) or np.any(actions >= self._classes):
            raise ValueError("An action lies outside the available class set")
        if np.any(self._actions[selection] != -1):
            raise RuntimeError("This feedback batch has already been consumed")
        clean = actions == self._labels[self.indices[selection]]
        rewards = np.logical_xor(clean, self._flips[selection] < self._flip_rates[selection])
        self._actions[selection], self._rewards[selection] = actions, rewards
        self._next_step += 1
        return rewards.astype(np.float32)

    def _selection(self, step):
        """Enforce chronological observations and prevent querying future or negative budget slots."""
        if step != self._next_step:
            raise RuntimeError("Feedback batches must be consumed once in chronological order")
        start = step * self.batch_size
        if start < 0 or start + self.batch_size > len(self.indices):
            raise IndexError("Feedback budget exhausted")
        return slice(start, start + self.batch_size)

    def committed_state(self, interactions):
        """Save consumed actions only; future random numbers are regenerated from the common seed."""
        return {"actions": self._actions[:interactions].copy(),
                "rewards": self._rewards[:interactions].copy()}

    def restore(self, state):
        """Restore the committed interaction prefix and verify its rewards against the oracle."""
        actions, rewards = state["actions"], state["rewards"]
        n = len(actions)
        if (n > len(self.indices) or n != len(rewards) or n % self.batch_size != 0 or
                np.any(actions < 0) or np.any(actions >= self._classes)):
            raise ValueError("Invalid committed oracle history")
        clean = actions == self._labels[self.indices[:n]]
        expected = np.logical_xor(clean, self._flips[:n] < self._flip_rates[:n])
        if not np.array_equal(expected, rewards):
            raise ValueError("Stored feedback is inconsistent with the common environment")
        self._actions[:n], self._rewards[:n] = actions, rewards
        self._next_step = n // self.batch_size

    def audit(self):
        """Release ground truth only to completed-run reporting, never to the reward processor."""
        if np.any(self._actions < 0):
            raise RuntimeError("Ground-truth audit is unavailable before the complete budget is consumed")
        return {"indices": self.indices.copy(), "actions": self._actions.copy(),
                "observed_reward": self._rewards.copy(),
                "clean_reward": (self._actions == self._labels[self.indices]).astype(np.int8),
                "context": self._context.copy(), "feedback_flip": self._flip_rates.copy(),
                "volatility": self._volatility.copy(), "phase": self._phase.copy()}

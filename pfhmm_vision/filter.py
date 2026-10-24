"""Action-conditioned PF-HMM extension, not a reproduction of the human binary-state task.

The human model jointly filters volatility and stochasticity. Here its latent binary
state indexes two predeclared visual contexts, with an action-conditioned likelihood
estimated on a separate labeled calibration partition. Neither hidden context nor
the true feedback-flip probability is supplied to this module.
"""

from copy import deepcopy
from dataclasses import asdict

import numpy as np

from .config import FilterConfig


class CauseFilter:
    """Rao-Blackwellize binary context while particle-filtering two unknown noise sources."""

    def __init__(self, config, seed):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.weights = np.full(config.particles, 1.0 / config.particles)
        self.context = np.full((config.particles, 2), 0.5)
        self.volatility = np.full(config.particles, config.volatility_initial)
        self.noise = np.full(config.particles, config.noise_initial)
        self.observations = 0

    def _diffuse(self, values, scale):
        """Use bounded beta diffusion with conditional mean equal to the current parameter."""
        if scale == 0:
            return values.copy()
        epsilon = self.config.probability_floor
        doubled = np.clip(2 * values, epsilon, 1 - epsilon)
        result = 0.5 * self.rng.beta(doubled / scale, (1 - doubled) / scale)
        return np.clip(result, epsilon, 0.5 - epsilon)

    def update(self, reward, correctness_by_context):
        """Filter one observed bit and estimate latent clean correctness for its selected action."""
        q = np.asarray(correctness_by_context, dtype=np.float64)
        if reward not in (0, 1) or q.shape != (2,) or not np.isfinite(q).all():
            raise ValueError("The filter accepts one observed bit and two finite action probabilities")
        if np.any(q < 0) or np.any(q > 1):
            raise ValueError("Correctness probabilities must be in [0, 1]")
        epsilon = self.config.probability_floor
        q = np.clip(q, epsilon, 1 - epsilon)
        self.volatility = self._diffuse(self.volatility, self.config.volatility_diffusion)
        self.noise = self._diffuse(self.noise, self.config.noise_diffusion)
        prior_context = ((1 - self.volatility[:, None]) * self.context +
                         self.volatility[:, None] * self.context[:, ::-1])
        joint_prior = self.weights[:, None] * prior_context
        positive = self.noise[:, None] + (1 - 2 * self.noise[:, None]) * q[None, :]
        likelihood = positive if reward == 1 else 1 - positive
        joint_posterior = joint_prior * likelihood
        evidence = float(joint_posterior.sum())
        clean_joint = q[None, :] * ((1 - self.noise[:, None]) if reward == 1
                                  else self.noise[:, None])
        corrected = float(np.clip((joint_prior * clean_joint).sum() / evidence, 0, 1))
        particle_likelihood = (prior_context * likelihood).sum(axis=1)
        self.weights = self.weights * particle_likelihood / evidence
        self.weights /= self.weights.sum()
        self.context = prior_context * likelihood / particle_likelihood[:, None]
        ess = float(1 / np.square(self.weights).sum())
        result = {"clean_reward": corrected,
                  "noise": float(self.weights @ self.noise),
                  "volatility": float(self.weights @ self.volatility),
                  "context_one": float(self.weights @ self.context[:, 1]),
                  "predictive_likelihood": evidence,
                  "effective_particles": ess}
        if ess < self.config.resample_fraction * len(self.weights):
            positions = (self.rng.random() + np.arange(len(self.weights))) / len(self.weights)
            indices = np.searchsorted(np.cumsum(self.weights), positions, side="right")
            indices = np.minimum(indices, len(self.weights) - 1)
            self.volatility = self.volatility[indices].copy()
            self.noise = self.noise[indices].copy()
            self.context = self.context[indices].copy()
            self.weights.fill(1 / len(self.weights))
        self.observations += 1
        return result

    def state_dict(self):
        """Export particle arrays and private RNG for exact checkpoint continuation."""
        return {"config": asdict(self.config), "rng": deepcopy(self.rng.bit_generator.state),
                "weights": self.weights.copy(), "context": self.context.copy(),
                "volatility": self.volatility.copy(), "noise": self.noise.copy(),
                "observations": self.observations}

    def load_state_dict(self, state):
        """Reject a changed inference model instead of silently reusing incompatible particles."""
        if state["config"] != asdict(self.config):
            raise ValueError("Particle-filter configuration changed")
        for key in ("weights", "context", "volatility", "noise"):
            setattr(self, key, np.asarray(state[key], dtype=np.float64).copy())
        n = self.config.particles
        if self.weights.shape != (n,) or self.context.shape != (n, 2):
            raise ValueError("Invalid particle checkpoint shape")
        if self.volatility.shape != (n,) or self.noise.shape != (n,):
            raise ValueError("Invalid noise checkpoint shape")
        if any(not np.isfinite(getattr(self, key)).all()
               for key in ("weights", "context", "volatility", "noise")):
            raise ValueError("Nonfinite particle checkpoint")
        if (np.any(self.weights < 0) or not np.isclose(self.weights.sum(), 1) or
                np.any(self.context < 0) or not np.allclose(self.context.sum(1), 1) or
                np.any(self.volatility < 0) or np.any(self.volatility >= 0.5) or
                np.any(self.noise < 0) or np.any(self.noise >= 0.5)):
            raise ValueError("Invalid probability normalization or bounds in particle checkpoint")
        self.observations = int(state["observations"])
        self.rng.bit_generator.state = deepcopy(state["rng"])

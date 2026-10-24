"""Check inference identities, information separation, gradients, and exact run continuation."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from pfhmm_vision.artifacts import atomic_json, atomic_torch, load_checkpoint, seed_everything
from pfhmm_vision.calibration import fit_temperatures
from pfhmm_vision.config import (BanditConfig, CalibrationConfig, Config, DataConfig,
                                 FilterConfig, ModelConfig, Phase, PretrainConfig,
                                 View, load_config)
from pfhmm_vision.data import apply_view, stratified_splits
from pfhmm_vision.environment import FeedbackOracle
from pfhmm_vision.filter import CauseFilter
from pfhmm_vision.metrics import classification_metrics
from pfhmm_vision.models import ResidualAdapter
from pfhmm_vision.train import actor_loss, sample_actions, train_arm


class ConfigurationTests(unittest.TestCase):
    """Protect the declared experimental budget and split contract."""

    def test_default_budget_is_one_32000_feedback_arm(self):
        config = Config().validate()
        self.assertEqual(config.bandit.updates * config.bandit.batch_size, 32000)
        self.assertEqual(sum(getattr(config.data, f"{s}_per_class")
                             for s in ("pretrain", "calibration", "adaptation", "validation")), 500)

    def test_unknown_configuration_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            atomic_json(path, {"bandit": {"learning_reate": 1}})
            with self.assertRaises(ValueError):
                load_config(path)

    def test_overlapping_budget_and_invalid_noise_are_rejected(self):
        with self.assertRaises(ValueError):
            replace(Config(), data=replace(DataConfig(), adaptation_per_class=200)).validate()
        with self.assertRaises(ValueError):
            replace(Config(), bandit=replace(BanditConfig(), phases=(Phase(feedback_flip=0.5),))).validate()

    def test_splits_are_disjoint_stratified_and_reproducible(self):
        labels = np.repeat(np.arange(4), 20)
        counts = {"source": 8, "calibration": 3, "adaptation": 5, "validation": 4}
        left = stratified_splits(labels, counts, 17)
        right = stratified_splits(labels, counts, 17)
        self.assertEqual(len(np.unique(np.concatenate(list(left.values())))), len(labels))
        for name, indices in left.items():
            np.testing.assert_array_equal(indices, right[name])
            np.testing.assert_array_equal(np.bincount(labels[indices]), np.full(4, counts[name]))


class FilterTests(unittest.TestCase):
    """Verify the filter against direct Bayesian enumeration and chronological replay."""

    def test_one_step_matches_enumeration_over_hidden_correctness(self):
        config = FilterConfig(particles=8, noise_initial=0.2,
                              noise_diffusion=0, volatility_diffusion=0)
        filter_ = CauseFilter(config, 3)
        q = np.array([0.8, 0.3])
        for reward in (0, 1):
            result = CauseFilter(config, 3).update(reward, q)
            denominator, numerator = 0.0, 0.0
            for context in (0, 1):
                for clean in (0, 1):
                    probability_clean = q[context] if clean else 1 - q[context]
                    probability_reward = 0.8 if clean == reward else 0.2
                    joint = 0.5 * probability_clean * probability_reward
                    denominator += joint
                    numerator += clean * joint
            self.assertAlmostEqual(result["clean_reward"], numerator / denominator, places=12)
            self.assertAlmostEqual(result["predictive_likelihood"], denominator, places=12)

    def test_zero_feedback_noise_returns_the_observed_bit(self):
        config = FilterConfig(particles=8, noise_diffusion=0, volatility_diffusion=0)
        for reward in (0, 1):
            filter_ = CauseFilter(config, 3)
            filter_.noise.fill(0)
            self.assertAlmostEqual(filter_.update(reward, [0.2, 0.7])["clean_reward"], reward)

    def test_selected_action_changes_the_likelihood(self):
        config = FilterConfig(particles=8, noise_diffusion=0, volatility_diffusion=0)
        low = CauseFilter(config, 1).update(1, [0.1, 0.1])
        high = CauseFilter(config, 1).update(1, [0.9, 0.9])
        self.assertLess(low["clean_reward"], high["clean_reward"])

    def test_filter_resume_restores_future_particle_randomness(self):
        config = FilterConfig(particles=64, noise_diffusion=0.01, volatility_diffusion=0.01)
        first = CauseFilter(config, 11)
        for outcome in [1, 1, 0, 1, 0, 0]:
            first.update(outcome, [0.8, 0.2])
        resumed = CauseFilter(config, 999)
        resumed.load_state_dict(first.state_dict())
        for outcome in [0, 1, 1, 1, 0, 0, 1]:
            self.assertEqual(first.update(outcome, [0.7, 0.3]), resumed.update(outcome, [0.7, 0.3]))
        np.testing.assert_array_equal(first.context, resumed.context)

    def test_known_noise_filter_tracks_a_controlled_hidden_state(self):
        rng = np.random.default_rng(21)
        config = FilterConfig(particles=16, noise_initial=0.1, volatility_initial=0.02,
                              noise_diffusion=0, volatility_diffusion=0)
        filter_ = CauseFilter(config, 5)
        context, squared_errors = 0, []
        for _ in range(1000):
            context ^= int(rng.random() < 0.02)
            clean = int(rng.random() < (0.9 if context else 0.1))
            reward = clean ^ int(rng.random() < 0.1)
            posterior = filter_.update(reward, [0.1, 0.9])["context_one"]
            squared_errors.append((posterior - context) ** 2)
        self.assertLess(np.mean(squared_errors), 0.15)

    def test_invalid_probability_is_rejected(self):
        filter_ = CauseFilter(FilterConfig(particles=8), 0)
        for q in ([0.4], [float("nan"), 0.5], [-0.1, 1.1]):
            with self.assertRaises(ValueError):
                filter_.update(1, q)


class EnvironmentTests(unittest.TestCase):
    """Check common random numbers and the strict one-bit feedback interface."""

    def test_reward_tracks_the_chosen_action_and_cannot_be_queried_twice(self):
        oracle = FeedbackOracle([0, 1, 2], [Phase(updates=1, feedback_flip=0)], 4, 0)
        actions = np.asarray([0, 1, 2])[oracle.indices]
        np.testing.assert_array_equal(oracle.feedback(0, actions), np.ones(4))
        with self.assertRaises(RuntimeError):
            oracle.feedback(0, actions)

    def test_hidden_ground_truth_is_unavailable_until_completion(self):
        oracle = FeedbackOracle([0, 1], [Phase(updates=2)], 2, 7)
        with self.assertRaises(RuntimeError):
            oracle.audit()
        oracle.feedback(0, np.zeros(2, dtype=int))
        with self.assertRaises(RuntimeError):
            oracle.audit()
        oracle.feedback(1, np.zeros(2, dtype=int))
        self.assertEqual(len(oracle.audit()["actions"]), 4)

    def test_common_environment_and_checkpoint_history_are_reproducible(self):
        left = FeedbackOracle([0, 1, 2], [Phase(updates=3)], 4, 71)
        right = FeedbackOracle([0, 1, 2], [Phase(updates=3)], 4, 71)
        np.testing.assert_array_equal(left.indices, right.indices)
        np.testing.assert_array_equal(left.action_uniforms, right.action_uniforms)
        left.feedback(0, np.zeros(4, dtype=int))
        right.restore(left.committed_state(4))
        np.testing.assert_array_equal(left.feedback(1, np.ones(4, dtype=int)),
                                      right.feedback(1, np.ones(4, dtype=int)))


class LearningTests(unittest.TestCase):
    """Check policy gradients, calibration isolation, and complete deterministic continuation."""

    def setUp(self):
        """Avoid oversubscribing CPU kernels during tiny numerical tests."""
        seed_everything(0, threads=2)

    def test_zero_initialized_adapter_equals_reference(self):
        adapter = ResidualAdapter(7, 5, 3)
        reference = torch.randn(6, 3)
        torch.testing.assert_close(adapter(torch.randn(6, 7), reference), reference, rtol=0, atol=0)

    def test_expected_reinforce_gradient_matches_negative_correct_class_probability(self):
        logits = torch.tensor([[0.2, -0.1, 0.5]], dtype=torch.float64, requires_grad=True)
        probability = logits.softmax(1).detach()[0]
        config = BanditConfig(kl_coefficient=0, entropy_coefficient=0)
        objective = 0
        for action in range(3):
            loss, _ = actor_loss(logits, logits.detach(), torch.tensor([action]),
                                 torch.tensor([float(action == 1)]), torch.tensor([0.4]), config)
            objective = objective + probability[action] * loss
        gradient = torch.autograd.grad(objective, logits, retain_graph=True)[0]
        expected = torch.autograd.grad(-logits.softmax(1)[0, 1], logits)[0]
        torch.testing.assert_close(gradient, expected, rtol=1e-7, atol=1e-9)

    def test_coupled_sampling_respects_policy_and_uniform_bounds(self):
        logits = torch.zeros(3, 3)
        np.testing.assert_array_equal(sample_actions(logits, [0.1, 0.5, 0.9]), [0, 1, 2])
        with self.assertRaises(ValueError):
            sample_actions(logits, [0, 0.5, 1])
        with self.assertRaises(ValueError):
            sample_actions(logits, [0, float("nan"), 0.5])
        np.testing.assert_array_equal(sample_actions(torch.tensor([[-10000., 0.]]), [0.0]), [1])

    def test_calibration_reduces_nll_without_changing_argmax(self):
        logits = torch.tensor([[[10., 0.], [10., 0.], [10., 0.], [0., 10.]]])
        labels = torch.tensor([0, 1, 0, 1])
        temperature, diagnostics = fit_temperatures(logits, labels, CalibrationConfig(steps=100))
        self.assertLess(diagnostics[0]["calibrated_nll"], diagnostics[0]["uncalibrated_nll"])
        torch.testing.assert_close(logits.argmax(-1), (logits / temperature[:, None, None]).argmax(-1))

    def test_metrics_expose_single_class_collapse(self):
        metrics = classification_metrics(torch.tensor([[10., 0.], [10., 0.]]), [0, 1])
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["predicted_classes"], 1)
        self.assertGreater(metrics["ece"], 0.49)

    def test_corruptions_are_stateless_and_do_not_modify_source_pixels(self):
        from PIL import Image
        pixels = np.full((8, 8, 3), 128, dtype=np.uint8)
        image = Image.fromarray(pixels)
        left = apply_view(image, View("gaussian_noise", 0.1), 5, [0, 0, 0], [1, 1, 1])
        right = apply_view(image, View("gaussian_noise", 0.1), 5, [0, 0, 0], [1, 1, 1])
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        np.testing.assert_array_equal(np.asarray(image), pixels)

    def test_both_arms_resume_bitwise_with_identical_initialization(self):
        config = replace(Config(), model=ModelConfig(adapter_hidden=8),
                         data=replace(DataConfig(), workers=0),
                         filter=FilterConfig(particles=32),
                         bandit=BanditConfig(batch_size=4, checkpoint_every=2, log_every=4,
                                             phases=(Phase(updates=4),)))
        generator = torch.Generator().manual_seed(10)
        cache = {"features": torch.randn(2, 9, 6, generator=generator),
                 "logits": torch.randn(2, 9, 3, generator=generator)}
        validation = {key: value[:, :4].clone() for key, value in cache.items()}
        labels = np.arange(9) % 3
        hashes = []
        with tempfile.TemporaryDirectory() as directory:
            for name in ("baseline", "pfhmm"):
                full = Path(directory) / name / "full"
                partial = Path(directory) / name / "partial"
                args = (name, config, cache, labels, validation, torch.tensor(labels[:4]),
                        torch.ones(2))
                full_model, full_summary = train_arm(*args, full, 0, torch.device("cpu"), "test")
                _, stopped = train_arm(*args, partial, 0, torch.device("cpu"), "test", stop_after=2)
                self.assertFalse(stopped["complete"])
                resumed, resumed_summary = train_arm(*args, partial, 0, torch.device("cpu"), "test")
                self.assertTrue(resumed_summary["complete"])
                for key in full_model.state_dict():
                    torch.testing.assert_close(full_model.state_dict()[key], resumed.state_dict()[key], rtol=0, atol=0)
                left = load_checkpoint(full / name / "checkpoint.pt")
                right = load_checkpoint(partial / name / "checkpoint.pt")
                self.assertEqual(left["history"], right["history"])
                for key in left["trace"]:
                    np.testing.assert_array_equal(left["trace"][key], right["trace"][key])
                hashes.append(full_summary["initial_adapter_hash"])
                with self.assertRaises(ValueError):
                    train_arm(*args, full, 0, torch.device("cpu"), "different identity")
            self.assertEqual(hashes[0], hashes[1])


if __name__ == "__main__":
    unittest.main()

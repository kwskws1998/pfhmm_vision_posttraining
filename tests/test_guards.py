"""Exercise setup safety, cache integrity, invalid inputs, and final-test access guards."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import bootstrap
from pfhmm_vision.artifacts import atomic_json, execution_lock
from pfhmm_vision.cache import feature_cache
from pfhmm_vision.config import Config, DataConfig, FilterConfig, View
from pfhmm_vision.environment import FeedbackOracle
from pfhmm_vision.evaluate import evaluate
from pfhmm_vision.filter import CauseFilter
from pfhmm_vision.metrics import classification_metrics


class GuardTests(unittest.TestCase):
    """Verify failure behavior rather than just successful method outputs."""

    def test_invalid_config_numbers_and_noninteger_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            replace(Config(), filter=replace(FilterConfig(), noise_diffusion=float("nan"))).validate()
        with self.assertRaises(ValueError):
            replace(Config(), filter=replace(FilterConfig(), particles=3.5)).validate()
        with self.assertRaises(ValueError):
            replace(Config(), data=replace(DataConfig(), contexts=(View(), View()))).validate()

    def test_oracle_rejects_future_batches_and_invalid_actions(self):
        from pfhmm_vision.config import Phase
        oracle = FeedbackOracle([0, 1], [Phase(updates=2)], 2, 0)
        with self.assertRaises(RuntimeError):
            oracle.feedback(1, np.zeros(2, dtype=int))
        for actions in (np.array([-1, 0]), np.array([0, 2]), np.array([0.1, 1.0])):
            with self.assertRaises(ValueError):
                oracle.feedback(0, actions)

    def test_invalid_filter_checkpoint_probability_is_rejected(self):
        filter_ = CauseFilter(FilterConfig(particles=4), 0)
        state = filter_.state_dict()
        state["weights"][0] = -1
        with self.assertRaises(ValueError):
            filter_.load_state_dict(state)

    def test_execution_lock_excludes_another_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            with execution_lock(directory):
                with self.assertRaises(RuntimeError):
                    with execution_lock(directory):
                        self.fail("The nested writer acquired an exclusive run lock")

    def test_final_test_is_blocked_for_an_incomplete_arm(self):
        with self.assertRaises(RuntimeError):
            evaluate({"pfhmm": None}, None, None, None, Config(), None, None,
                     {"pfhmm": {"complete": False}})

    def test_cache_checksums_detect_modified_bytes(self):
        from pfhmm_vision.artifacts import atomic_torch, file_hash
        from pfhmm_vision.config import digest
        from pfhmm_vision.data import view_fingerprint
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.pt"
            normalization = [[0.5] * 3, [0.2] * 3]
            specification = {"identity": "source", "indices": digest([0, 1]),
                             "views": [view_fingerprint(View())], "view_seed": 3,
                             "normalization": normalization, "format_version": 1}
            atomic_torch(path, {"fingerprint": digest(specification), "features": torch.ones(1, 2, 3)})
            atomic_json(path.with_suffix(".json"), {"specification": specification, "sha256": file_hash(path)})
            with path.open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaises(ValueError):
                feature_cache(None, None, [0, 1], [View()], normalization,
                              Config(), path, "source", torch.device("cpu"), 3)

    def test_metrics_reject_nonfinite_predictions(self):
        with self.assertRaises(FloatingPointError):
            classification_metrics([[float("inf"), 0]], [0])


class BootstrapTests(unittest.TestCase):
    """Check installer command construction without changing a user's existing environments."""

    def test_exact_canonical_dependency_pins(self):
        expected = bootstrap.pins(Path(bootstrap.__file__).parent)
        self.assertEqual(set(expected), {"torch", "torchvision", "numpy", "Pillow"})
        self.assertEqual(expected["torchvision"], "0.23.0")

    def test_cuda_and_cpu_wheel_requirements_have_different_exact_builds(self):
        expected = bootstrap.pins(Path(bootstrap.__file__).parent)
        cuda = bootstrap.wheel_requirements(expected, "cuda", "Linux")
        cpu = bootstrap.wheel_requirements(expected, "cpu", "Linux")
        self.assertEqual(cuda, ["torch==2.8.0+cu128", "torchvision==0.23.0+cu128"])
        self.assertEqual(cpu, ["torch==2.8.0+cpu", "torchvision==0.23.0+cpu"])
        with self.assertRaises(ValueError):
            bootstrap.wheel_requirements(expected, "cuda", "Darwin")

    def test_existing_non_conda_directory_is_never_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("bootstrap.subprocess.check_output", return_value="/some/base\n"):
                with self.assertRaises(ValueError):
                    bootstrap.setup(Path(bootstrap.__file__).parent, "cpu", directory, conda="conda")

    def test_conda_base_is_never_modified(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("bootstrap.subprocess.check_output", return_value=directory + "\n"):
                with self.assertRaises(ValueError):
                    bootstrap.setup(Path(bootstrap.__file__).parent, "cpu", directory, conda="conda")


if __name__ == "__main__":
    unittest.main()

"""Verify Python configuration, complete study planning, and paired source reuse on synthetic data."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiment import load_settings, main as experiment_main
from pfhmm_vision.config import Config, configure
from pfhmm_vision.study import plan_study, run_study


def tiny_config(root):
    """Return a real-network configuration with a tiny synthetic training budget."""
    return configure(
        data={"root": str(Path(root) / "unused"), "pretrain_per_class": 6,
              "calibration_per_class": 2, "adaptation_per_class": 2,
              "validation_per_class": 2, "workers": 0, "download": False,
              "test_views": [{"kind": "clean", "severity": 0.0}]},
        pretrain={"epochs": 1, "batch_size": 4, "warmup_epochs": 0},
        calibration={"steps": 2},
        bandit={"batch_size": 4, "checkpoint_every": 1, "log_every": 1,
                "phases": [{"updates": 1}]},
        filter={"particles": 8},
        evaluation={"batch_size": 4},
    )


class PythonConfigurationTests(unittest.TestCase):
    """Preserve immutable defaults while accepting strict nested Python overrides."""

    def test_nested_overrides_preserve_unmodified_base_fields(self):
        base = configure(bandit={"learning_rate": 0.002}, filter={"particles": 32})
        changed = configure(base, bandit={"kl_coefficient": 0.3})
        self.assertEqual(base.bandit.kl_coefficient, 0.1)
        self.assertEqual(changed.bandit.learning_rate, 0.002)
        self.assertEqual(changed.bandit.kl_coefficient, 0.3)
        self.assertEqual(changed.filter.particles, 32)

    def test_nested_phases_and_views_are_resolved_to_dataclasses(self):
        config = configure(
            bandit={"phases": [{"name": "test", "updates": 3, "feedback_flip": 0.2}]},
            data={"test_views": [{"kind": "contrast", "severity": 0.5}]},
        )
        self.assertEqual(config.bandit.updates, 3)
        self.assertEqual(config.bandit.phases[0].name, "test")
        self.assertEqual(config.bandit.phases[0].volatility, 0.002)
        self.assertEqual(config.data.test_views[0].severity, 0.5)

    def test_unknown_fields_and_invalid_values_fail_before_training(self):
        for overrides in ({"unknown": 1}, {"bandit": {"learning_reate": 0.1}},
                          {"bandit": {"phases": [{"update": 2}]}},
                          {"data": {"test_views": [{"severty": 0.1}]}},
                          {"filter": {"particles": 0}},
                          {"bandit": {"learning_rate": float("nan")}}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                configure(**overrides)
        with self.assertRaises(TypeError):
            configure({"bandit": {"learning_rate": 0.1}})


class StudyPlanningTests(unittest.TestCase):
    """Validate all named conditions and deterministic output paths before executing a study."""

    def test_default_seeds_and_config_identity_determine_output_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "runs"
            plans = plan_study(Config(), {"first": {}, "second": {"filter": {"particles": 32}}}, root=root)
            self.assertEqual([case["seeds"] for case in plans], [(0, 2), (0, 2)])
            for case in plans:
                self.assertEqual(case["directory"], root / case["name"] / case["config_identity"][:16])
                self.assertTrue(Path(case["config"].data.root).is_absolute())
            self.assertNotEqual(plans[0]["config_identity"], plans[1]["config_identity"])
            again = plan_study(Config(), {"first": {}}, root=root)[0]
            self.assertEqual(again["directory"], plans[0]["directory"])
            changed = plan_study(Config(), {"first": {}}, seeds=(0, 3), root=root)[0]
            self.assertNotEqual(changed["directory"], plans[0]["directory"])
            self.assertFalse(root.exists())

    def test_path_traversal_empty_and_invalid_conditions_are_rejected(self):
        for name in ("", ".", "..", "../escape", "/absolute", "a/b", "a\\b", "a b"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                plan_study(Config(), {name: {}})
        for conditions in ({}, [], None):
            with self.subTest(conditions=conditions), self.assertRaises(ValueError):
                plan_study(Config(), conditions)
        with self.assertRaises(TypeError):
            plan_study(Config(), {"bad": []})

    def test_invalid_later_condition_blocks_the_entire_study_before_downloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "runs"
            with patch("run_two_seed.run_two_seed") as runner:
                with self.assertRaises(ValueError):
                    run_study(Config(), {"good": {}, "bad": {"filter": {"particles": 0}}}, root=root)
            runner.assert_not_called()
            self.assertFalse(root.exists())

    def test_invalid_seed_selection_is_rejected_before_execution(self):
        for seeds in ((0, 0), (0,), (0, 1, 2), (0, -2), (False, 2)):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                plan_study(Config(), {"default": {}}, seeds=seeds)

    def test_default_settings_and_dry_run_do_not_start_training(self):
        project = Path(__file__).resolve().parents[1]
        settings = load_settings(project / "configs" / "experiment.py")
        self.assertEqual(settings["SEEDS"], (0, 2))
        self.assertEqual(settings["DEVICE"], "cuda")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "runs"
            output = io.StringIO()
            with patch("sys.argv", ["experiment.py", "--dry-run", "--runs-dir", str(root)]), \
                    patch("experiment.run_study") as runner, redirect_stdout(output):
                experiment_main()
            runner.assert_not_called()
            self.assertIn("seeds=(0, 2)", output.getvalue())
            self.assertFalse(root.exists())


class StudyExecutionTests(unittest.TestCase):
    """Exercise real ResNet training with synthetic images and shared same-seed source checkpoints."""

    def test_two_conditions_share_sources_per_seed_and_resume_without_retraining(self):
        from pfhmm_vision.pipeline import run
        from pfhmm_vision.pretrain import pretrain
        from smoke import SyntheticCIFAR

        with tempfile.TemporaryDirectory(prefix="pfhmm_study_test_") as temporary:
            config = tiny_config(temporary)
            conditions = {"first": {}, "second": {"bandit": {"learning_rate": 0.0002}}}
            root = Path(temporary) / "study"
            with patch("pfhmm_vision.data.CIFAR100", SyntheticCIFAR), \
                    patch("pfhmm_vision.pipeline.CIFAR100", SyntheticCIFAR), \
                    patch("pfhmm_vision.pipeline.pretrain", wraps=pretrain) as source_runner, \
                    redirect_stdout(io.StringIO()):
                result = run_study(config, conditions, root=root, device="cpu")
                self.assertEqual(source_runner.call_count, 4)
                self.assertIsNone(source_runner.call_args_list[0].args[-1])
                self.assertIsNone(source_runner.call_args_list[1].args[-1])
                self.assertIsNotNone(source_runner.call_args_list[2].args[-1])
                self.assertIsNotNone(source_runner.call_args_list[3].args[-1])
                resumed = run_study(config, conditions, root=root, device="cpu")
                second_only = run_study(config, {"second": conditions["second"]}, root=root, device="cpu")
                reversed_cases = run_study(config, dict(reversed(list(conditions.items()))), root=root, device="cpu")
                interrupted_conditions = {"interrupted": {"bandit": {"learning_rate": 0.0003}}}
                interrupted_case = plan_study(config, interrupted_conditions, root=root)[0]
                for seed in (0, 2):
                    first_root = Path(result["conditions"]["first"]["directory"]) / f"seed{seed}"
                    source = json.loads((first_root / "complete.json").read_text())["shared_base_checkpoint"]
                    with patch("pfhmm_vision.pipeline.feature_cache", side_effect=RuntimeError("interrupted cache")):
                        with self.assertRaisesRegex(RuntimeError, "interrupted cache"):
                            run(interrupted_case["config"], interrupted_case["directory"] / f"seed{seed}",
                                seed, "cpu", base_checkpoint=source)
                    seed_root = interrupted_case["directory"] / f"seed{seed}"
                    self.assertTrue((seed_root / "shared" / "source_checkpoint.json").is_file())
                    self.assertFalse((seed_root / "complete.json").exists())
                partial_resumed = run_study(config, interrupted_conditions, root=root, device="cpu")
            self.assertEqual(result["evaluation_status"], "complete")
            self.assertEqual(set(result["conditions"]), {"first", "second"})
            first = result["conditions"]["first"]["summary"]
            second = result["conditions"]["second"]["summary"]
            self.assertEqual(first["seeds"], [0, 2])
            self.assertEqual(first["run_count"], 4)
            first_sources = {item["seed"]: item["shared_base_sha256"] for item in first["provenance"]}
            second_sources = {item["seed"]: item["shared_base_sha256"] for item in second["provenance"]}
            self.assertEqual(first_sources, second_sources)
            self.assertNotEqual(first_sources[0], first_sources[2])
            partial_sources = {item["seed"]: item["shared_base_sha256"]
                               for item in partial_resumed["conditions"]["interrupted"]["summary"]["provenance"]}
            self.assertEqual(first_sources, partial_sources)
            for name in conditions:
                original = result["conditions"][name]["summary"]
                repeated = resumed["conditions"][name]["summary"]
                self.assertEqual(original["rows"], repeated["rows"])
                self.assertEqual(original["comparisons"], repeated["comparisons"])
                self.assertEqual(original["rows"], reversed_cases["conditions"][name]["summary"]["rows"])
                case_root = Path(result["conditions"][name]["directory"])
                for seed in (0, 2):
                    completion = json.loads((case_root / f"seed{seed}" / "complete.json").read_text())
                    self.assertEqual(completion["methods"], ["baseline", "pfhmm"])
                    for method in ("baseline", "pfhmm"):
                        self.assertTrue((case_root / f"seed{seed}" / method / "training_complete.json").is_file())
            self.assertEqual(second["rows"], second_only["conditions"]["second"]["summary"]["rows"])
            self.assertEqual(second["provenance"], second_only["conditions"]["second"]["summary"]["provenance"])


if __name__ == "__main__":
    unittest.main()

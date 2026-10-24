"""Run Python-defined conditions with two paired seeds and separate result directories."""

from dataclasses import asdict, replace
from pathlib import Path
import re

from .config import Config, configure, digest
from run_two_seed import validate_seeds


def plan_study(config, conditions, seeds=(0, 2), root="runs/cifar100_study"):
    """Resolve every condition before execution so invalid later cases cannot waste training."""
    if not isinstance(config, Config):
        raise TypeError("config must be a Config instance")
    config.validate()
    seeds = validate_seeds(seeds)
    if not isinstance(conditions, dict) or not conditions:
        raise ValueError("conditions must be a nonempty dictionary of names to nested overrides")
    root = Path(root).expanduser().resolve()
    plans = []
    for name, overrides in conditions.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError("Condition names must start with a letter/digit and use letters, digits, _, . or -")
        if not isinstance(overrides, dict):
            raise TypeError("Each condition must contain a dictionary of configuration overrides")
        resolved = configure(config, **overrides)
        data_root = Path(resolved.data.root).expanduser().resolve()
        resolved = replace(resolved, data=replace(resolved.data, root=str(data_root)))
        identity = digest({"config": resolved.as_dict(), "seeds": list(seeds)})
        plans.append({"name": name, "config": resolved, "seeds": seeds,
                      "directory": root / name / identity[:16], "config_identity": identity})
    return plans


def print_plan(plans, device):
    """Show resolved training budgets before downloads or GPU work start."""
    print(f"Study: {len(plans)} condition(s), device={device}", flush=True)
    for case in plans:
        config = case["config"]
        print(f"  {case['name']}: seeds={case['seeds']} methods=(baseline, pfhmm) "
              f"source_epochs={config.pretrain.epochs} "
              f"updates_per_arm={config.bandit.updates} "
              f"feedback_per_arm={config.bandit.updates * config.bandit.batch_size}\n"
              f"    bandit_lr={config.bandit.learning_rate} kl={config.bandit.kl_coefficient} "
              f"particles={config.filter.particles} output={case['directory']}", flush=True)


def _source_recipe(config, seed):
    """Identify candidate source reuse; the checkpoint loader also verifies actual split and code."""
    split_fields = ("name", "split_seed", "pretrain_per_class", "calibration_per_class",
                    "adaptation_per_class", "validation_per_class")
    return digest({"model": asdict(config.model), "pretrain": asdict(config.pretrain),
                   "data": {key: getattr(config.data, key) for key in split_fields}, "seed": seed})


def run_study(config, conditions=None, seeds=(0, 2), root="runs/cifar100_study", device="cuda"):
    """Run every condition and seed, reusing a verified same-seed source across compatible cases."""
    from .artifacts import atomic_json, code_hash, execution_lock
    from .progress import Progress
    from run_two_seed import run_two_seed
    import json

    conditions = {"default": {}} if conditions is None else conditions
    plans = plan_study(config, conditions, seeds, root)
    print_plan(plans, device)
    root = Path(root).expanduser().resolve()
    source_hash = code_hash()
    plan = {"seeds": list(plans[0]["seeds"]), "methods": ["baseline", "pfhmm"],
            "code_sha256": source_hash, "arm_count": 4 * len(plans),
            "conditions": [{"name": case["name"], "config": case["config"].as_dict(),
                            "directory": str(case["directory"])} for case in plans]}
    identity = digest(plan)
    with execution_lock(root):
        atomic_json(root / "plan.json", {**plan, "identity": identity})
        destination = root / "studies" / identity
        destination.mkdir(parents=True, exist_ok=True)
        atomic_json(destination / "plan.json", {**plan, "identity": identity})
        progress = Progress(len(plans), "study conditions")
        progress.report(0, force=True)
        atomic_json(root / "progress.json", {**progress.snapshot(), "study_identity": identity,
                                             "status": "running"})
        summaries, source_checkpoints = {}, {}
        for index, case in enumerate(plans):
            config = case["config"]
            checkpoints = {seed: source_checkpoints[_source_recipe(config, seed)]
                           for seed in case["seeds"] if _source_recipe(config, seed) in source_checkpoints}
            summaries[case["name"]] = run_two_seed(
                config, case["directory"], case["seeds"], device, base_checkpoints=checkpoints)
            for seed in case["seeds"]:
                completion = json.loads((case["directory"] / f"seed{seed}" / "complete.json").read_text())
                source_checkpoints[_source_recipe(config, seed)] = Path(completion["shared_base_checkpoint"])
            progress.report(index + 1, detail=f"finished condition={case['name']}", force=True)
            atomic_json(root / "progress.json", {**progress.snapshot(), "study_identity": identity,
                                                 "status": "complete" if index + 1 == len(plans) else "running",
                                                 "last_completed_condition": case["name"]})
        result = {"evaluation_status": "complete", "study_identity": identity,
                  "this_invocation_seconds": progress.snapshot()["elapsed_seconds"],
                  "conditions": {case["name"]: {"directory": str(case["directory"]),
                                                "summary": summaries[case["name"]]} for case in plans},
                  "selection": "All specified conditions are retained; no condition is selected using test scores."}
        atomic_json(destination / "summary.json", result)
        atomic_json(root / "summary.json", result)
        print(f"Study complete: {root / 'summary.json'}", flush=True)
        return result

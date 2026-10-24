"""Run a Python-configured baseline/PF-HMM study without changing training implementation."""

import argparse
from dataclasses import replace
from pathlib import Path
import runpy

from pfhmm_vision.config import Config, load_config
from pfhmm_vision.study import plan_study, print_plan, run_study


def load_settings(path):
    """Execute the user's Python settings file and require explicit experiment values."""
    values = runpy.run_path(str(path))
    required = ("SEEDS", "DEVICE", "RUNS_DIR", "BASE_CONFIG", "CONDITIONS")
    missing = [name for name in required if name not in values]
    if missing:
        raise ValueError(f"Missing Python experiment settings: {missing}")
    if not isinstance(values["BASE_CONFIG"], Config):
        raise TypeError("BASE_CONFIG must be constructed using configure() or Config()")
    if not isinstance(values["CONDITIONS"], dict) or not values["CONDITIONS"]:
        raise ValueError("CONDITIONS must be a nonempty dictionary")
    return {name: values[name] for name in required}


def main():
    """Load Python conditions, allow CLI overrides, and validate the entire plan before training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", default="configs/experiment.py")
    parser.add_argument("--config", help="Optional JSON base config replacing BASE_CONFIG")
    parser.add_argument("--seeds", type=int, nargs=2)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--runs-dir")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without downloads or training")
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    resolve = lambda value: Path(value).expanduser() if Path(value).expanduser().is_absolute() else project / value
    settings = load_settings(resolve(args.settings))
    config = settings["BASE_CONFIG"] if args.config is None else load_config(resolve(args.config))
    config = replace(config, data=replace(config.data, root=str(resolve(config.data.root).resolve())))
    conditions = {}
    for name, overrides in settings["CONDITIONS"].items():
        if isinstance(overrides, dict) and isinstance(overrides.get("data"), dict) and "root" in overrides["data"]:
            overrides = {**overrides, "data": {**overrides["data"],
                                               "root": str(resolve(overrides["data"]["root"]).resolve())}}
        conditions[name] = overrides
    seeds = settings["SEEDS"] if args.seeds is None else args.seeds
    device = settings["DEVICE"] if args.device is None else args.device
    if device not in ("cpu", "cuda"):
        raise ValueError("DEVICE must be cpu or cuda")
    root = resolve(settings["RUNS_DIR"] if args.runs_dir is None else args.runs_dir)
    if args.dry_run:
        print_plan(plan_study(config, conditions, seeds, root), device)
    else:
        run_study(config, conditions, seeds, root, device)


if __name__ == "__main__":
    main()

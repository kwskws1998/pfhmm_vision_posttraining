"""Set up Conda and launch baseline and PF-HMM at two seeds with one Python command."""

import argparse
import os
from pathlib import Path
import subprocess

from bootstrap import setup
from experiment import load_settings
from pfhmm_vision.config import load_config
from pfhmm_vision.study import plan_study
from run_two_seed import validate_seeds


def main():
    """Keep setup and experiment execution independent while providing one convenient entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--seed", type=int, help="Run the original single-seed entry point")
    selection.add_argument("--seeds", type=int, nargs=2)
    parser.add_argument("--settings", default="configs/experiment.py")
    parser.add_argument("--config")
    parser.add_argument("--runs-dir")
    parser.add_argument("--conda")
    parser.add_argument("--verify-only-environment", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    settings_path = Path(args.settings).expanduser()
    settings = load_settings(settings_path if settings_path.is_absolute() else root / settings_path)
    device = args.device or settings["DEVICE"]
    seeds = settings["SEEDS"] if args.seeds is None else args.seeds
    if args.seed is None:
        validate_seeds(seeds)
        config_path = Path(args.config).expanduser() if args.config else None
        config = settings["BASE_CONFIG"] if config_path is None else load_config(
            config_path if config_path.is_absolute() else root / config_path)
        plan_study(config, settings["CONDITIONS"], seeds, root / (args.runs_dir or settings["RUNS_DIR"]))
    elif not 0 <= args.seed < 2 ** 32:
        parser.error("--seed must be in [0, 2**32)")
    if device not in ("cpu", "cuda"):
        parser.error("DEVICE must be cpu or cuda")
    executable = setup(root, device, conda=args.conda, verify_only=args.verify_only_environment)
    entry = "experiment.py" if args.seed is None else "run_experiment.py"
    command = [str(executable), "-s", str(root / entry), "--device", device]
    command.extend(["--seeds", *map(str, seeds), "--settings", args.settings]
                   if args.seed is None else ["--seed", str(args.seed)])
    if args.config:
        command.extend(["--config", args.config])
    if args.runs_dir:
        command.extend(["--runs-dir", args.runs_dir])
    subprocess.run(command, check=True, cwd=root, env={**os.environ, "PYTHONNOUSERSITE": "1"})


if __name__ == "__main__":
    main()

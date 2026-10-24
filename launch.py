"""Set up Conda and launch the matched one-seed experiment with one Python command."""

import argparse
from pathlib import Path
import subprocess

from bootstrap import setup


def main():
    """Keep setup and experiment execution independent while providing one convenient entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default="configs/cifar100.json")
    parser.add_argument("--runs-dir")
    parser.add_argument("--conda")
    parser.add_argument("--verify-only-environment", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    executable = setup(root, args.device, conda=args.conda, verify_only=args.verify_only_environment)
    command = [str(executable), str(root / "run_experiment.py"), "--device", args.device,
               "--seed", str(args.seed), "--config", args.config]
    if args.runs_dir:
        command.extend(["--runs-dir", args.runs_dir])
    subprocess.run(command, check=True, cwd=root)


if __name__ == "__main__":
    main()

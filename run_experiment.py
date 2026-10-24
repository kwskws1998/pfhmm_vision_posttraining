"""Run one shared CIFAR-100 pretraining stage and exactly one seed per selected RL method."""

import argparse
from dataclasses import replace
from pathlib import Path


def main():
    """Resolve every relative path against this project, including invocation from /workspace."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cifar100.json")
    parser.add_argument("--runs-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--methods", nargs="+", choices=("baseline", "pfhmm"), default=["baseline", "pfhmm"])
    parser.add_argument("--base-checkpoint", default=None,
                        help="Optional completed source checkpoint produced by this exact recipe")
    parser.add_argument("--stage", choices=("prepare", "pretrain", "all"), default="all")
    args = parser.parse_args()
    from pfhmm_vision.config import load_config
    from pfhmm_vision.pipeline import run
    project = Path(__file__).resolve().parent
    resolve = lambda path: Path(path).expanduser() if Path(path).expanduser().is_absolute() else project / path
    config = load_config(resolve(args.config))
    config = replace(config, data=replace(config.data, root=str(resolve(config.data.root).resolve())))
    runs_dir = resolve(args.runs_dir or f"runs/cifar100_seed{args.seed}")
    checkpoint = None if args.base_checkpoint is None else resolve(args.base_checkpoint)
    run(config, runs_dir, args.seed, args.device, tuple(args.methods), checkpoint, args.stage)


if __name__ == "__main__":
    main()

"""Exercise the actual ResNet pipeline on labeled synthetic images, without downloads."""

import argparse
from dataclasses import replace
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
from PIL import Image

from pfhmm_vision.config import (BanditConfig, CalibrationConfig, Config, DataConfig,
                                 EvaluationConfig, FilterConfig, Phase, PretrainConfig, View)
from pfhmm_vision.pipeline import run


class SyntheticCIFAR:
    """Supply a tiny deterministic provider solely for pipeline verification."""

    def __init__(self, root, train=True, download=False):
        rng = np.random.default_rng(10 if train else 20)
        self.classes = ["synthetic_zero", "synthetic_one"]
        self.targets = np.repeat([0, 1], 12 if train else 4).tolist()
        self.data = rng.integers(0, 256, (len(self.targets), 32, 32, 3), dtype=np.uint8)

    def __len__(self):
        """Return the synthetic population size."""
        return len(self.targets)

    def __getitem__(self, index):
        """Return an image/label pair with the same provider interface as torchvision CIFAR."""
        return Image.fromarray(self.data[index]), self.targets[index]


def smoke(root, device):
    """Run source training, calibration, both policy updates, caches, resume, and final metrics."""
    config = replace(Config(),
                     data=DataConfig(root=str(Path(root) / "unused"), pretrain_per_class=6,
                                     calibration_per_class=2, adaptation_per_class=2,
                                     validation_per_class=2, workers=0, download=False,
                                     test_views=(View(), View("gaussian_noise", 0.15))),
                     pretrain=PretrainConfig(epochs=1, batch_size=4, warmup_epochs=0),
                     calibration=CalibrationConfig(steps=3),
                     bandit=BanditConfig(batch_size=4, checkpoint_every=1, log_every=1,
                                         phases=(Phase(updates=1), Phase("noisy", 1, 0.02, 0.3))),
                     filter=FilterConfig(particles=32),
                     evaluation=EvaluationConfig(batch_size=4))
    with patch("pfhmm_vision.data.CIFAR100", SyntheticCIFAR), patch("pfhmm_vision.pipeline.CIFAR100", SyntheticCIFAR):
        first = run(config, root, 0, device)
        second = run(config, root, 0, device)
    if first != second:
        raise AssertionError("Completed experiment reuse changed final predictions or metrics")
    print("Synthetic pipeline smoke passed. This is not CIFAR-100 efficacy evidence.")


def main():
    """Use a temporary directory by default so synthetic results cannot be mistaken for real runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--runs-dir")
    args = parser.parse_args()
    if args.runs_dir:
        smoke(Path(args.runs_dir).resolve(), args.device)
    else:
        with tempfile.TemporaryDirectory(prefix="pfhmm_synthetic_") as directory:
            smoke(directory, args.device)


if __name__ == "__main__":
    main()

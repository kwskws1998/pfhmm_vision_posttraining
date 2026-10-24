"""Disjoint CIFAR-100 partitions and deterministic, label-preserving visual contexts."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torchvision.transforms import functional as TF

from .artifacts import atomic_json
from .config import digest


def stratified_splits(labels, counts, seed):
    """Partition each class once; no official test labels participate in this operation."""
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    result = {name: [] for name in counts}
    for target in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == target))
        if sum(counts.values()) > len(indices):
            raise ValueError(f"Insufficient examples for class {target}")
        start = 0
        for name, count in counts.items():
            result[name].extend(indices[start:start + count].tolist())
            start += count
    return {name: np.sort(indices).astype(np.int64) for name, indices in result.items()}


def load_training_data(config, root):
    """Download only the official training partition and persist its disjoint split manifest."""
    dataset = CIFAR100(root=config.root, train=True, download=config.download)
    counts = {name: getattr(config, f"{name}_per_class")
              for name in ("pretrain", "calibration", "adaptation", "validation")}
    splits = stratified_splits(dataset.targets, counts, config.split_seed)
    manifest = {"dataset": config.name, "class_names": dataset.classes,
                "split_seed": config.split_seed,
                "indices": {key: value.tolist() for key, value in splits.items()},
                "label_fingerprint": digest(dataset.targets)}
    manifest["fingerprint"] = digest(manifest)
    path = Path(root) / "shared" / "splits.json"
    if path.exists():
        import json
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Stored data partitions differ from the requested experiment")
    else:
        atomic_json(path, manifest)
    return dataset, splits, manifest


def normalization(dataset):
    """Calculate channel statistics from the source images explicitly supplied by the caller."""
    sums, squares, count = np.zeros(3), np.zeros(3), 0
    for start in range(0, len(dataset), 1024):
        images = np.asarray(dataset[start:start + 1024], dtype=np.float64) / 255.0
        sums += images.sum(axis=(0, 1, 2))
        squares += np.square(images).sum(axis=(0, 1, 2))
        count += np.prod(images.shape[:3])
    if count == 0:
        raise ValueError("Source normalization requires nonempty training images")
    mean = sums / count
    std = np.sqrt(np.maximum(squares / count - mean ** 2, 0))
    if np.any(std == 0):
        raise ValueError("A source channel has zero variance")
    return mean.tolist(), std.tolist()


def apply_view(image, view, seed, mean, std):
    """Apply a stateless corruption in pixel space before source-training normalization."""
    image = TF.to_tensor(image)
    if view.kind == "gaussian_noise":
        generator = torch.Generator().manual_seed(int(seed))
        image = (image + torch.randn(image.shape, generator=generator) * view.severity).clamp(0, 1)
    elif view.kind == "contrast":
        center = image.mean(dim=(-2, -1), keepdim=True)
        image = ((image - center) * view.severity + center).clamp(0, 1)
    elif view.kind != "clean":
        raise ValueError(f"Unregistered visual context: {view.kind}")
    return TF.normalize(image, mean, std)


class PretrainingDataset(Dataset):
    """Expose labels only to the supervised source-pretraining stage."""

    def __init__(self, dataset, indices, mean, std, augment):
        self.dataset, self.indices = dataset, np.asarray(indices)
        operations = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if augment else []
        self.transform = transforms.Compose(operations + [transforms.ToTensor(),
                                                           transforms.Normalize(mean, std)])

    def __len__(self):
        """Return the number of source-pretraining examples."""
        return len(self.indices)

    def __getitem__(self, position):
        """Return one augmented source image and its source label."""
        image, target = self.dataset[int(self.indices[position])]
        return self.transform(image), target


class ContextDataset(Dataset):
    """Provide reproducible feature-cache views without exposing targets to the encoder loop."""

    def __init__(self, dataset, indices, view, mean, std, view_seed):
        self.dataset, self.indices, self.view = dataset, np.asarray(indices), view
        self.mean, self.std, self.view_seed = mean, std, view_seed

    def __len__(self):
        """Return the fixed cache population size."""
        return len(self.indices)

    def __getitem__(self, position):
        """Return a deterministic view; labels are retained outside the training interface."""
        index = int(self.indices[position])
        image, _ = self.dataset[index]
        seed = np.random.SeedSequence([self.view_seed, index]).generate_state(1)[0]
        return apply_view(image, self.view, seed, self.mean, self.std)


def view_key(view):
    """Name an evaluation condition using its actual configured parameters."""
    return f"{view.kind}_{view.severity}"


def view_fingerprint(view):
    """Identify a corruption definition independently of its display label."""
    return digest(asdict(view))

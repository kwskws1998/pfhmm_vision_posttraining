"""One shared supervised source checkpoint, independent of post-training arm choice."""

from dataclasses import asdict
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .artifacts import atomic_torch, load_checkpoint, restore_rng, rng_state, seed_everything
from .config import digest
from .data import PretrainingDataset, normalization
from .models import make_backbone
from .progress import Progress


def pretraining_identity(config, manifest, seed, source_hash):
    """Bind pretrained weights to their architecture, source split, training recipe, and seed."""
    return digest({"model": asdict(config.model), "training": asdict(config.pretrain),
                   "splits": manifest["fingerprint"], "seed": seed, "code": source_hash})


def _epoch_learning_rate(config, epoch):
    """Warm up once and then use an epoch-based cosine schedule that resumes exactly."""
    if epoch < config.warmup_epochs:
        return config.learning_rate * (epoch + 1) / config.warmup_epochs
    progress = (epoch - config.warmup_epochs) / max(1, config.epochs - config.warmup_epochs - 1)
    return config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))


def _source_accuracy(model, loader, device, progress=None, completed=0):
    """Measure source-validation accuracy without modifying frozen evaluation statistics."""
    model.eval()
    correct, total = 0, 0
    with torch.inference_mode():
        for batch, (images, targets) in enumerate(loader, 1):
            predictions = model(images.to(device)).argmax(1).cpu()
            correct += int((predictions == targets).sum())
            total += len(targets)
            if progress is not None:
                progress.report(completed + batch, detail="source validation")
    return correct / total


def pretrain(config, dataset, splits, manifest, root, seed, device, source_hash, checkpoint=None):
    """Train or resume the common base; reuse only a completed checkpoint with matched provenance."""
    seed_everything(seed)
    identity = pretraining_identity(config, manifest, seed, source_hash)
    destination = Path(root) / "shared" / "pretrain.pt"
    existing = Path(checkpoint) if checkpoint else destination
    model = make_backbone(config.model, len(dataset.classes)).to(device)
    state = load_checkpoint(existing) if existing.exists() else None
    if checkpoint and state is None:
        raise FileNotFoundError(checkpoint)
    if state is not None and state.get("identity") != identity:
        raise ValueError("Base checkpoint does not match this source split, recipe, seed, or code")
    if state is not None:
        model.load_state_dict(state["model"], strict=True)
        mean, std = state["normalization"]
        if state["epoch"] == config.pretrain.epochs:
            print(f"Reusing completed shared pretraining: {existing}", flush=True)
            return model.eval(), state, existing
        if checkpoint:
            raise ValueError("An external base checkpoint must be complete")
    else:
        mean, std = normalization(dataset.data[splits["pretrain"]])
    optimizer = torch.optim.SGD(model.parameters(), lr=config.pretrain.learning_rate,
                                momentum=config.pretrain.momentum,
                                weight_decay=config.pretrain.weight_decay)
    start = 0
    history, elapsed = [], 0.0
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])
        start, history, elapsed = state["epoch"], state["history"], state["elapsed_seconds"]
        restore_rng(state["rng"])
    source = PretrainingDataset(dataset, splits["pretrain"], mean, std, augment=True)
    validation = PretrainingDataset(dataset, splits["validation"], mean, std, augment=False)
    validation_loader = DataLoader(validation, batch_size=config.evaluation.batch_size,
                                   shuffle=False, num_workers=config.data.workers,
                                   generator=torch.Generator().manual_seed(seed + 19))
    criterion = nn.CrossEntropyLoss()
    training_batches = math.ceil(len(source) / config.pretrain.batch_size)
    epoch_batches = training_batches + len(validation_loader)
    progress = Progress(config.pretrain.epochs * epoch_batches, "pretrain batches",
                        completed=start * epoch_batches, elapsed_seconds=elapsed)
    progress.report(force=True, detail=f"epoch={start}/{config.pretrain.epochs}")
    for epoch in range(start, config.pretrain.epochs):
        learning_rate = _epoch_learning_rate(config.pretrain, epoch)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        loader = DataLoader(source, batch_size=config.pretrain.batch_size, shuffle=True,
                            num_workers=config.data.workers, pin_memory=device.type == "cuda",
                            generator=torch.Generator().manual_seed(seed + epoch), drop_last=False)
        model.train()
        loss_sum, correct, count = 0.0, 0, 0
        for batch, (images, targets) in enumerate(loader, 1):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite source-pretraining loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.pretrain.gradient_clip,
                                     error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(targets)
            correct += int((logits.argmax(1) == targets).sum())
            count += len(targets)
            progress.report(epoch * epoch_batches + batch,
                            detail=f"epoch={epoch + 1}/{config.pretrain.epochs} "
                                   f"batch={batch}/{training_batches}")
        accuracy = _source_accuracy(model, validation_loader, device, progress,
                                    epoch * epoch_batches + training_batches)
        elapsed = progress.snapshot()["elapsed_seconds"]
        record = {"epoch": epoch + 1, "loss": loss_sum / count,
                  "training_accuracy": correct / count, "validation_accuracy": accuracy,
                  "learning_rate": learning_rate}
        history.append(record)
        state = {"format_version": 1, "identity": identity, "model": model.state_dict(),
                 "epoch": epoch + 1, "optimizer": optimizer.state_dict(), "rng": rng_state(),
                 "normalization": [mean, std], "history": history, "elapsed_seconds": elapsed,
                 "classes": dataset.classes, "source_split": manifest["fingerprint"]}
        atomic_torch(destination, state)
        progress.report(force=True, detail=f"epoch={epoch + 1}/{config.pretrain.epochs} "
                        f"loss={record['loss']:.4f} validation_accuracy={accuracy:.4f}")
    return model.eval(), state, destination

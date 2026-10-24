"""Content-verified frozen-encoder caches; no target labels are stored with adaptation features."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .artifacts import atomic_json, atomic_torch, file_hash, load_checkpoint
from .config import digest
from .data import ContextDataset, view_fingerprint
from .models import encode
from .progress import Progress


def feature_cache(model, dataset, indices, views, normalization, config, path, identity, device, seed):
    """Cache fixed views with verified model/split/view identities and a checksum sidecar."""
    path = Path(path)
    specification = {"identity": identity, "indices": digest(np.asarray(indices).tolist()),
                     "views": [view_fingerprint(v) for v in views], "view_seed": seed,
                     "normalization": normalization, "format_version": 1}
    expected = digest(specification)
    sidecar = path.with_suffix(".json")
    if path.exists() and sidecar.exists():
        import json
        metadata = json.loads(sidecar.read_text())
        if metadata.get("specification") != specification or metadata.get("sha256") != file_hash(path):
            raise ValueError(f"Feature-cache provenance or checksum mismatch: {path}")
        cached = load_checkpoint(path)
        if cached.get("fingerprint") != expected:
            raise ValueError("Feature-cache contents have a different identity")
        print(f"Reusing completed feature cache: {path}", flush=True)
        return cached
    if path.exists() != sidecar.exists():
        print(f"Rebuilding an interrupted feature-cache commit: {path}", flush=True)
    model.eval()
    features, logits = [], []
    progress = Progress(len(views) * len(indices), f"cache {path.name} examples")
    progress.report(force=True)
    for context, view in enumerate(views):
        source = ContextDataset(dataset, indices, view, *normalization, view_seed=seed)
        loader = DataLoader(source, batch_size=config.evaluation.batch_size, shuffle=False,
                            num_workers=config.data.workers,
                            generator=torch.Generator().manual_seed(seed),
                            pin_memory=device.type == "cuda")
        context_features, context_logits = [], []
        with torch.inference_mode():
            for images in loader:
                hidden, predictions = encode(model, images.to(device))
                context_features.append(hidden.cpu().float())
                context_logits.append(predictions.cpu().float())
                progress.report(progress.completed + len(images),
                                detail=f"context={context + 1}/{len(views)}")
        features.append(torch.cat(context_features))
        logits.append(torch.cat(context_logits))
        progress.report(force=True, detail=f"context={context + 1}/{len(views)} complete")
    cached = {"fingerprint": expected, "features": torch.stack(features),
              "logits": torch.stack(logits), "indices": np.asarray(indices, dtype=np.int64)}
    atomic_torch(path, cached)
    atomic_json(sidecar, {"specification": specification, "sha256": file_hash(path)})
    return cached

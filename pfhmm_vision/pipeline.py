"""Ordered experiment stages with one shared base, one seed, and final-test separation."""

from dataclasses import asdict
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
import torchvision
from torchvision.datasets import CIFAR100

from .artifacts import (atomic_json, code_hash, execution_lock, file_hash,
                        resolve_device, seed_everything)
from .cache import feature_cache
from .calibration import fit_temperatures
from .config import View, digest
from .data import load_training_data
from .evaluate import evaluate, print_comparison
from .pretrain import pretrain
from .progress import format_duration
from .train import train_arm


def runtime_metadata(device):
    """Record observed runtime details rather than treating GPU specifications as measurements."""
    result = {"python": platform.python_version(), "torch": torch.__version__,
              "torchvision": torchvision.__version__, "numpy": np.__version__,
              "device": str(device), "cuda_runtime": torch.version.cuda}
    if device.type == "cuda":
        result.update({"gpu": torch.cuda.get_device_name(device),
                       "total_vram_bytes": torch.cuda.get_device_properties(device).total_memory})
    return result


def run(config, root, seed=0, device_name="cuda", methods=("baseline", "pfhmm"),
        base_checkpoint=None, stage="all"):
    """Execute one seed per requested arm, automatically reusing strictly matched completed stages."""
    config.validate()
    if not 0 <= seed < 2 ** 32 or stage not in ("prepare", "pretrain", "all"):
        raise ValueError("Invalid seed or experiment stage")
    if not methods or len(set(methods)) != len(methods) or set(methods) - {"baseline", "pfhmm"}:
        raise ValueError("Methods must be a unique nonempty selection of baseline and pfhmm")
    device = resolve_device(device_name)
    root = Path(root).resolve()
    source_hash = code_hash()
    specification = {"config": config.as_dict(), "seed": seed, "code_sha256": source_hash,
                     "methods": sorted(methods),
                     "format_version": 1}
    identity = digest(specification)
    with execution_lock(root):
        manifest_path = root / "experiment.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous["identity"] != identity:
                raise ValueError("This run directory belongs to a different config, seed, or source version")
        else:
            atomic_json(manifest_path, {**specification, "identity": identity,
                                        "runtime": runtime_metadata(device),
                                        "label_access": {
                                            "pretrain": "source pretraining partition only",
                                            "calibration": "dedicated labeled calibration partition only",
                                            "posttraining": "one observed bit per chosen action",
                                            "validation": "logging only; fixed final update is evaluated",
                                            "test": "opened after all requested training arms finish"}})
        dataset, splits, split_manifest = load_training_data(config.data, root)
        if stage == "prepare":
            print(f"Prepared disjoint source partitions: {root / 'shared' / 'splits.json'}")
            return None
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        print(f"Experiment seed={seed} methods={methods} -> {root}", flush=True)
        print("Stage 1/5: source pretraining", flush=True)
        source_reference = root / "shared" / "source_checkpoint.json"
        if source_reference.exists():
            reference = json.loads(source_reference.read_text())
            established_path = Path(reference["path"])
            if not established_path.is_absolute():
                established_path = root / established_path
            if reference["experiment_identity"] != identity or file_hash(established_path) != reference["sha256"]:
                raise ValueError("The established source checkpoint has incompatible provenance or checksum")
            base_checkpoint = established_path
        backbone, source_state, source_path = pretrain(
            config, dataset, splits, split_manifest, root, seed, device, source_hash, base_checkpoint)
        source_checksum = file_hash(source_path)
        source_path = Path(source_path).resolve()
        stored_source_path = str(source_path.relative_to(root)) if source_path.is_relative_to(root) else str(source_path)
        atomic_json(source_reference, {"experiment_identity": identity, "path": stored_source_path,
                                       "sha256": source_checksum})
        source_peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        if stage == "pretrain":
            return None
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        shared_identity = digest({"source_checkpoint": source_checksum, "code": source_hash})
        normalization = source_state["normalization"]
        caches = {}
        print(f"Stage 2/5: frozen feature caches; elapsed={format_duration(time.monotonic() - started)}", flush=True)
        for name in ("calibration", "adaptation", "validation"):
            views = config.data.contexts if name != "validation" else (View(),)
            caches[name] = feature_cache(
                backbone, dataset, splits[name], views, normalization, config,
                root / "shared" / f"{name}_features.pt", shared_identity, device, seed + 4001)
        calibration_path = root / "shared" / "calibration.json"
        print(f"Stage 3/5: calibration; elapsed={format_duration(time.monotonic() - started)}", flush=True)
        calibration_identity = digest({"cache": caches["calibration"]["fingerprint"],
                                       "config": asdict(config.calibration)})
        if calibration_path.exists():
            calibration = json.loads(calibration_path.read_text())
            if calibration["identity"] != calibration_identity:
                raise ValueError("Stored calibration has incompatible provenance")
            if calibration["contents_sha256"] != digest({key: value for key, value in calibration.items()
                                                        if key != "contents_sha256"}):
                raise ValueError("Stored calibration values failed their content checksum")
            temperatures = torch.tensor(calibration["temperatures"])
        else:
            calibration_targets = torch.tensor(np.asarray(dataset.targets)[splits["calibration"]])
            temperatures, diagnostics = fit_temperatures(
                caches["calibration"]["logits"], calibration_targets, config.calibration)
            calibration = {"identity": calibration_identity, "temperatures": temperatures.tolist(),
                           "diagnostics": diagnostics,
                           "labeled_examples": len(calibration_targets),
                           "context_identifiability_warning":
                               "Similar temperature-conditioned probabilities can make causes unidentifiable."}
            calibration["contents_sha256"] = digest(calibration)
            atomic_json(calibration_path, calibration)
        del caches["calibration"]
        summaries, models = {}, {}
        training_identity = digest({"experiment": identity, "source": source_checksum,
                                    "calibration": digest(calibration),
                                    "adaptation": caches["adaptation"]["fingerprint"]})
        reference_probability = caches["adaptation"]["logits"].softmax(-1)
        from .calibration import correctness_probabilities
        separations = []
        for logits, policy in zip(caches["adaptation"]["logits"], reference_probability):
            conditional = correctness_probabilities(logits, temperatures)
            separations.append(float(((conditional[0] - conditional[1]).abs() * policy).sum(-1).mean()))
        atomic_json(root / "shared" / "likelihood_diagnostics.json", {
            "temperatures": temperatures.tolist(),
            "mean_action_weighted_probability_gap_by_actual_context": separations,
            "interpretation": "A zero gap makes latent visual context unidentifiable from this likelihood; a positive gap does not establish identifiability."})
        for name in methods:
            print(f"Stage 4/5: {name} training; elapsed={format_duration(time.monotonic() - started)}", flush=True)
            models[name], summaries[name] = train_arm(
                name, config, caches["adaptation"], np.asarray(dataset.targets)[splits["adaptation"]],
                caches["validation"], torch.tensor(np.asarray(dataset.targets)[splits["validation"]]),
                temperatures, root, seed, device, training_identity)
        if len(models) == 2 and len({s["initial_adapter_hash"] for s in summaries.values()}) != 1:
            raise RuntimeError("The two methods did not share the exact adapter initialization")
        if file_hash(source_path) != source_checksum:
            raise RuntimeError("The frozen source checkpoint was modified during post-training")
        atomic_json(root / "test_access.json", {
            "identity": training_identity,
            "completed_methods": {name: summary["checkpoint_sha256"] for name, summary in summaries.items()},
            "selection": "fixed final update", "seed": seed})
        test = CIFAR100(root=config.data.root, train=False, download=config.data.download)
        print(f"Stage 5/5: final test cache and evaluation; elapsed={format_duration(time.monotonic() - started)}", flush=True)
        test_cache = feature_cache(
            backbone, test, np.arange(len(test)), config.data.test_views, normalization, config,
            root / "shared" / "test_features.pt", shared_identity, device, seed + 9001)
        result = evaluate(models, test_cache, test.targets, config.data.test_views,
                          config, root, device, summaries)
        peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        atomic_json(root / "complete.json", {
            "identity": identity, "seed": seed, "methods": list(methods),
            "shared_base_checkpoint": str(Path(source_path).resolve()),
            "shared_base_sha256": source_checksum, "results_sha256": file_hash(root / "results.json"),
            "runtime": runtime_metadata(device), "this_invocation_seconds": time.monotonic() - started,
            "this_invocation_peak_cuda_bytes": max(source_peak_memory, peak_memory,
                                                    *(s["peak_cuda_bytes"] for s in summaries.values())),
            "total_feedback_per_method": config.bandit.updates * config.bandit.batch_size,
            "supervised_pretraining_epochs": config.pretrain.epochs,
            "calibration_labeled_examples": config.data.calibration_per_class * len(dataset.classes)})
        print(f"Experiment seed={seed} complete; elapsed={format_duration(time.monotonic() - started)}", flush=True)
        return result

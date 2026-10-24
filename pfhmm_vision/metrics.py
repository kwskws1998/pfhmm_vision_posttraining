"""Classifier metrics with explicit units and no checkpoint selection on final-test data."""

import numpy as np
import torch
from torch.nn import functional as F


def classification_metrics(logits, targets, bins=15):
    """Report accuracy, NLL, Brier score, and confidence ECE on one complete population."""
    logits = torch.as_tensor(logits, dtype=torch.float64)
    targets = torch.as_tensor(targets, dtype=torch.long)
    if logits.ndim != 2 or len(logits) != len(targets) or len(targets) == 0:
        raise ValueError("Metrics require a nonempty aligned example/class matrix")
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Nonfinite evaluation predictions")
    probability = logits.softmax(1)
    confidence, predictions = probability.max(1)
    correct = predictions.eq(targets)
    membership = (confidence * bins).long().clamp(max=bins - 1)
    ece = 0.0
    for index in range(bins):
        mask = membership == index
        if mask.any():
            ece += float(mask.double().mean() *
                         (confidence[mask].mean() - correct[mask].double().mean()).abs())
    one_hot = F.one_hot(targets, logits.shape[1]).double()
    return {"examples": len(targets), "accuracy": float(correct.double().mean()),
            "nll": float(F.cross_entropy(logits, targets)), "ece": ece,
            "brier": float((probability - one_hot).square().sum(1).mean()),
            "predicted_classes": int(predictions.unique().numel())}


def predict_adapter(model, features, reference_logits, batch_size, device):
    """Apply an adapter to one cache in bounded batches without changing its state."""
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            end = start + batch_size
            outputs.append(model(features[start:end].to(device),
                                 reference_logits[start:end].to(device)).cpu())
    return torch.cat(outputs)


def audit_metrics(audit, trace):
    """Compute privileged mechanism diagnostics only after a completed training run."""
    result = {"feedback_count": len(audit["actions"]), "phases": []}
    for phase in np.unique(audit["phase"]):
        mask = audit["phase"] == phase
        row = {"phase": int(phase), "examples": int(mask.sum()),
               "clean_accuracy": float(audit["clean_reward"][mask].mean()),
               "observed_reward": float(audit["observed_reward"][mask].mean()),
               "true_feedback_flip": float(audit["feedback_flip"][mask].mean()),
               "true_volatility": float(audit["volatility"][mask].mean()),
               "training_reward_mse": float(np.mean((trace["training_reward"][mask] -
                                                        audit["clean_reward"][mask]) ** 2))}
        if "noise" in trace:
            row.update({"estimated_noise": float(trace["noise"][mask].mean()),
                        "noise_mae": float(np.abs(trace["noise"][mask] -
                                                  audit["feedback_flip"][mask]).mean()),
                        "estimated_volatility": float(trace["volatility"][mask].mean()),
                        "volatility_mae": float(np.abs(trace["volatility"][mask] -
                                                       audit["volatility"][mask]).mean()),
                        "context_brier": float(np.mean((trace["context_one"][mask] -
                                                          audit["context"][mask]) ** 2))})
        result["phases"].append(row)
    return result

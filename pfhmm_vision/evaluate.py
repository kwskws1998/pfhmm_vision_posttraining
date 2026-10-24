"""Final evaluation and direct one-seed comparisons after every requested arm has finished."""

import csv
import io
from pathlib import Path

import numpy as np
import torch

from .artifacts import atomic_json, atomic_npz, _atomic_write
from .data import view_key
from .metrics import classification_metrics, predict_adapter
from .progress import Progress


def evaluate(models, test_cache, targets, views, config, root, device, training_summaries):
    """Evaluate fixed final policies; do not select steps or hyperparameters using these results."""
    if not all(training_summaries.get(name, {}).get("complete") for name in models):
        raise RuntimeError("Final-test evaluation requires completed training for every requested arm")
    target = torch.as_tensor(targets)
    rows, predictions = [], {}
    progress = Progress(len(views) * (len(models) + 1), "evaluation policies/conditions")
    progress.report(force=True)
    for index, view in enumerate(views):
        feature, reference = test_cache["features"][index], test_cache["logits"][index]
        baseline_metrics = classification_metrics(reference, target, config.evaluation.ece_bins)
        rows.append({"method": "frozen_base", "condition": view_key(view), **baseline_metrics,
                     "accuracy_gain_pp": 0.0})
        predictions[f"base_{index}"] = reference.numpy()
        progress.report(progress.completed + 1,
                        detail=f"condition={view_key(view)} method=frozen_base")
        for name, model in models.items():
            logits = predict_adapter(model, feature, reference, config.evaluation.batch_size, device)
            metrics = classification_metrics(logits, target, config.evaluation.ece_bins)
            rows.append({"method": name, "condition": view_key(view), **metrics,
                         "accuracy_gain_pp": 100 * (metrics["accuracy"] - baseline_metrics["accuracy"])})
            predictions[f"{name}_{index}"] = logits.numpy()
            progress.report(progress.completed + 1,
                            detail=f"condition={view_key(view)} method={name}", force=True)
    comparisons = []
    for view in views:
        condition = view_key(view)
        by_method = {row["method"]: row for row in rows if row["condition"] == condition}
        if "baseline" in by_method and "pfhmm" in by_method:
            difference = 100 * (by_method["pfhmm"]["accuracy"] - by_method["baseline"]["accuracy"])
            comparisons.append({"condition": condition, "pfhmm_minus_baseline_pp": difference,
                                "winner": "tie" if difference == 0 else ("pfhmm" if difference > 0 else "baseline")})
    result = {"evaluation_status": "complete", "selection": "fixed final update; no test selection",
              "rows": rows, "comparisons": comparisons,
              "interpretation": "One seed is a pilot; no between-seed uncertainty estimate is available."}
    atomic_json(Path(root) / "results.json", result)
    _atomic_write(Path(root) / "results.csv", lambda path: _write_csv(path, rows))
    atomic_npz(Path(root) / "test_predictions.npz", targets=np.asarray(targets), **predictions)
    print_comparison(result)
    return result


def _write_csv(path, rows):
    """Write a flat table with explicit accuracy units in the column names."""
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_comparison(result):
    """Display matched comparisons without inferring performance from training rewards."""
    print("\nFinal test accuracy (%)", flush=True)
    print(f"{'condition':24s} {'frozen':>9s} {'baseline':>9s} {'pfhmm':>9s} {'delta pp':>10s}")
    conditions = dict.fromkeys(row["condition"] for row in result["rows"])
    for condition in conditions:
        values = {row["method"]: row["accuracy"] * 100 for row in result["rows"]
                  if row["condition"] == condition}
        display = lambda key: f"{values[key]:9.3f}" if key in values else "      n/a"
        delta = (f"{values['pfhmm'] - values['baseline']:10.3f}"
                 if "pfhmm" in values and "baseline" in values else "       n/a")
        print(f"{condition:24s} {display('frozen_base')} {display('baseline')} {display('pfhmm')} {delta}")

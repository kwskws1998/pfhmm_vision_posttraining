"""Run baseline and PF-HMM at two matched seeds and report descriptive aggregate metrics."""

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import statistics


def validate_seeds(seeds):
    """Require exactly two distinct unsigned 32-bit integer experiment seeds."""
    seeds = tuple(seeds)
    if (len(seeds) != 2 or any(type(seed) is not int or not 0 <= seed < 2 ** 32
                               for seed in seeds) or len(set(seeds)) != 2):
        raise ValueError("Provide exactly two distinct integer seeds in [0, 2**32)")
    return seeds


def summarize(seed_results):
    """Aggregate complete matched runs using sample standard deviations and paired differences."""
    seeds = validate_seeds(seed_results)
    tables = {}
    metrics = ("accuracy", "nll", "ece", "brier", "accuracy_gain_pp")
    for seed, result in seed_results.items():
        if result.get("evaluation_status") != "complete":
            raise ValueError(f"Seed {seed} has not completed final evaluation")
        table = {}
        for row in result["rows"]:
            key = (row["method"], row["condition"])
            if key in table:
                raise ValueError("Duplicate method/condition in seed results")
            if any(not math.isfinite(row[metric]) for metric in metrics):
                raise ValueError("Aggregate metrics must be finite")
            table[key] = row
        conditions = {condition for _, condition in table}
        expected = {(method, condition) for method in ("frozen_base", "baseline", "pfhmm")
                    for condition in conditions}
        if not conditions or set(table) != expected:
            raise ValueError("Every condition requires frozen_base, baseline, and pfhmm results")
        tables[seed] = table
    if set(tables[seeds[0]]) != set(tables[seeds[1]]):
        raise ValueError("The two seeds have different evaluation conditions")
    rows = []
    for method, condition in tables[seeds[0]]:
        values = [tables[seed][method, condition] for seed in seeds]
        if values[0]["examples"] != values[1]["examples"]:
            raise ValueError("The two seeds evaluated different population sizes")
        row = {"method": method, "condition": condition, "seed_count": 2,
               "examples_per_seed": values[0]["examples"]}
        for metric in metrics:
            observed = [value[metric] for value in values]
            row[f"{metric}_mean"] = statistics.mean(observed)
            row[f"{metric}_std"] = statistics.stdev(observed)
        rows.append(row)
    comparisons = []
    for condition in sorted({condition for _, condition in tables[seeds[0]]}):
        differences = {
            str(seed): 100 * (tables[seed]["pfhmm", condition]["accuracy"] -
                             tables[seed]["baseline", condition]["accuracy"])
            for seed in seeds}
        comparisons.append({"condition": condition,
                            "pfhmm_minus_baseline_pp_mean": statistics.mean(differences.values()),
                            "pfhmm_minus_baseline_pp_std": statistics.stdev(differences.values()),
                            "per_seed": differences})
    return {"evaluation_status": "complete", "seeds": list(seeds),
            "methods": ["baseline", "pfhmm"], "run_count": 4,
            "source_pretraining_runs": 2, "rows": rows, "comparisons": comparisons,
            "standard_deviation": "Sample standard deviation across seeds (ddof=1)",
            "interpretation": "Two-seed descriptive pilot; no claim of statistically established superiority."}


def _write_csv(path, rows):
    """Write one aggregate row per method and evaluation condition."""
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary):
    """Display accuracy percentages and paired gains with their across-seed standard deviations."""
    print("\nTwo-seed final test accuracy: mean +/- sample SD (%)", flush=True)
    for comparison in summary["comparisons"]:
        condition = comparison["condition"]
        values = {row["method"]: row for row in summary["rows"] if row["condition"] == condition}
        parts = [condition]
        for method in ("baseline", "pfhmm"):
            row = values[method]
            parts.append(f"{method}={100 * row['accuracy_mean']:.3f} +/- "
                         f"{100 * row['accuracy_std']:.3f}")
        parts.append(f"delta={comparison['pfhmm_minus_baseline_pp_mean']:.3f} +/- "
                     f"{comparison['pfhmm_minus_baseline_pp_std']:.3f} pp")
        print("  ".join(parts), flush=True)


def run_two_seed(config, root, seeds=(0, 2), device="cuda", base_checkpoints=None):
    """Run both arms sequentially per seed and publish aggregates only after both seeds complete."""
    from pfhmm_vision.artifacts import atomic_json, code_hash, execution_lock, file_hash, _atomic_write
    from pfhmm_vision.config import digest
    from pfhmm_vision.pipeline import run
    from pfhmm_vision.progress import Progress

    seeds = validate_seeds(seeds)
    base_checkpoints = {} if base_checkpoints is None else dict(base_checkpoints)
    if set(base_checkpoints) - set(seeds):
        raise ValueError("Base checkpoint keys must belong to the requested seeds")
    config.validate()
    root = Path(root).resolve()
    specification = {"format_version": 1, "config": config.as_dict(), "seeds": list(seeds),
                     "methods": ["baseline", "pfhmm"], "code_sha256": code_hash()}
    identity = digest(specification)
    with execution_lock(root):
        manifest = root / "suite.json"
        if manifest.exists() and json.loads(manifest.read_text())["identity"] != identity:
            raise ValueError("This suite directory belongs to different seeds, config, or source code")
        atomic_json(manifest, {**specification, "identity": identity,
                               "pretraining": "Independent per seed; shared between methods within each seed"})
        results, provenance = {}, []
        progress = Progress(len(seeds), "suite seeds")
        progress.report(0, force=True)
        atomic_json(root / "progress.json", {**progress.snapshot(), "status": "running"})
        for index, seed in enumerate(seeds):
            seed_root = root / f"seed{seed}"
            print(f"\nRunning seed {seed}: baseline and pfhmm -> {seed_root}", flush=True)
            results[seed] = run(config, seed_root, seed, device, ("baseline", "pfhmm"),
                                base_checkpoint=base_checkpoints.get(seed))
            completion = json.loads((seed_root / "complete.json").read_text())
            checksum = file_hash(seed_root / "results.json")
            if completion["results_sha256"] != checksum:
                raise ValueError("A seed result failed its completion checksum")
            provenance.append({"seed": seed, "directory": str(seed_root),
                               "experiment_identity": completion["identity"],
                               "results_sha256": checksum,
                               "shared_base_sha256": completion["shared_base_sha256"]})
            progress.report(index + 1, detail=f"finished seed={seed}", force=True)
            atomic_json(root / "progress.json", {**progress.snapshot(), "last_completed_seed": seed,
                                                 "status": "complete" if index + 1 == len(seeds) else "running"})
        summary = summarize(results)
        summary.update({"suite_identity": identity, "provenance": provenance})
        summary["this_invocation_seconds"] = progress.snapshot()["elapsed_seconds"]
        _atomic_write(root / "summary.csv", lambda path: _write_csv(path, summary["rows"]))
        atomic_json(root / "summary.json", summary)
        print_summary(summary)
        print(f"\nSaved aggregate results: {root / 'summary.json'}", flush=True)
        return summary


def main():
    """Resolve project-relative paths and launch exactly two full matched experiments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cifar100.json")
    parser.add_argument("--runs-dir", default="runs/cifar100_two_seed")
    parser.add_argument("--seeds", type=int, nargs=2, default=[0, 2], metavar=("SEED0", "SEED1"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    seeds = validate_seeds(args.seeds)
    from pfhmm_vision.config import load_config

    project = Path(__file__).resolve().parent
    resolve = lambda path: Path(path).expanduser() if Path(path).expanduser().is_absolute() else project / path
    config = load_config(resolve(args.config))
    config = replace(config, data=replace(config.data, root=str(resolve(config.data.root).resolve())))
    run_two_seed(config, resolve(args.runs_dir), seeds, args.device)


if __name__ == "__main__":
    main()

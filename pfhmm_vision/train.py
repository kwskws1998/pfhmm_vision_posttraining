"""Matched on-policy adapter training from observed or inferred binary feedback."""

from pathlib import Path
import time

import numpy as np
import torch

from .artifacts import (atomic_json, atomic_npz, atomic_torch, file_hash, load_checkpoint,
                        restore_rng, rng_state, seed_everything)
from .calibration import correctness_probabilities
from .environment import FeedbackOracle
from .methods import make_method
from .metrics import audit_metrics, classification_metrics, predict_adapter
from .models import make_adapter


def sample_actions(logits, uniforms):
    """Couple arms using the same uniforms while respecting each arm's current policy."""
    probabilities = logits.detach().double().softmax(-1).cpu().numpy()
    cumulative = np.cumsum(probabilities, axis=1)
    cumulative[:, -1] = 1.0
    uniforms = np.asarray(uniforms)
    if (uniforms.shape != (len(logits),) or not np.isfinite(uniforms).all() or
            not np.isfinite(probabilities).all() or np.any(uniforms < 0) or np.any(uniforms >= 1)):
        raise ValueError("Expected one sampling uniform in [0, 1) per example")
    return np.sum(cumulative <= uniforms[:, None], axis=1).astype(np.int64)


def actor_loss(logits, reference_logits, actions, rewards, baseline, config):
    """Use detached advantages and a full categorical KL to the shared reference policy."""
    log_probability = logits.log_softmax(1)
    probabilities = log_probability.exp()
    reference_log_probability = reference_logits.detach().log_softmax(1)
    selected = log_probability.gather(1, actions[:, None]).squeeze(1)
    advantage = rewards.detach() - baseline.detach()
    policy_loss = -(advantage * selected).mean()
    kl = (probabilities * (log_probability - reference_log_probability)).sum(1).mean()
    entropy = -(probabilities * log_probability).sum(1).mean()
    total = policy_loss + config.kl_coefficient * kl - config.entropy_coefficient * entropy
    return total, {"loss": float(total.detach()), "policy_loss": float(policy_loss.detach()),
                   "kl": float(kl.detach()), "entropy": float(entropy.detach())}


def train_arm(name, config, adaptation_cache, adaptation_labels, validation_cache,
              validation_labels, temperatures, root, seed, device, identity, stop_after=None):
    """Resume a single arm and commit all stochastic state before publishing completion."""
    seed_everything(seed)
    destination = Path(root) / name
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_path = destination / "checkpoint.pt"
    model = make_adapter(config.model, adaptation_cache["features"][0],
                         adaptation_cache["logits"][0]).to(device)
    initial_hash = _parameter_fingerprint(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.bandit.learning_rate,
                                  weight_decay=config.bandit.weight_decay)
    processor = make_method(name, config.filter, seed + 3109)
    oracle = FeedbackOracle(adaptation_labels, config.bandit.phases,
                            config.bandit.batch_size, seed + 1987)
    start, history, trace, elapsed = 0, [], {}, 0.0
    peak_memory = 0
    if checkpoint_path.exists():
        saved = load_checkpoint(checkpoint_path)
        if saved.get("identity") != identity or saved.get("method") != name:
            raise ValueError("Training checkpoint provenance does not match the requested arm")
        if saved["step"] < config.bandit.updates and saved.get("runtime") != _runtime(device):
            raise ValueError("Exact continuation requires the same device type and library versions")
        if saved["initial_adapter_hash"] != initial_hash:
            raise ValueError("The shared adapter initialization has changed")
        model.load_state_dict(saved["adapter"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        processor.load_state_dict(saved["processor"])
        oracle.restore(saved["oracle"])
        start, history, trace = saved["step"], saved["history"], saved["trace"]
        elapsed, peak_memory = saved["elapsed_seconds"], saved["peak_cuda_bytes"]
        restore_rng(saved["rng"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    call_started = time.monotonic()
    committed_elapsed = elapsed
    for step in range(start, config.bandit.updates):
        if stop_after is not None and step >= stop_after:
            break
        features, reference_logits, uniforms = oracle.observation(step, adaptation_cache)
        features, reference_logits = features.to(device), reference_logits.to(device)
        model.train()
        logits = model(features, reference_logits)
        actions_np = sample_actions(logits, uniforms)
        rewards = oracle.feedback(step, actions_np)
        actions = torch.from_numpy(actions_np).to(device)
        with torch.no_grad():
            all_correctness = correctness_probabilities(reference_logits, temperatures.to(device))
            expected_correctness = all_correctness.mean(0)
            baseline = (logits.detach().softmax(1) * expected_correctness).sum(1)
            action_correctness = all_correctness.gather(
                2, actions[None, :, None].expand(len(temperatures), -1, 1)).squeeze(2).T.cpu().numpy()
        processed = processor.process(rewards, action_correctness)
        for key, values in processed.items():
            if not np.isfinite(values).all():
                raise FloatingPointError(f"Nonfinite reward-processing output: {key}")
            trace[key] = np.concatenate((trace.get(key, np.empty(0)), values))
        target_reward = torch.tensor(processed["training_reward"], device=device, dtype=logits.dtype)
        loss, record = actor_loss(logits, reference_logits, actions, target_reward, baseline, config.bandit)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite policy objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.bandit.gradient_clip, error_if_nonfinite=True)
        optimizer.step()
        if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise FloatingPointError("Nonfinite adapter parameter after optimizer step")
        record.update({"step": step + 1, "feedback_count": (step + 1) * config.bandit.batch_size,
                       "observed_reward": float(rewards.mean()),
                       "training_reward": float(target_reward.mean()),
                       "gradient_norm": float(gradient_norm)})
        if (step + 1) % config.bandit.checkpoint_every == 0 or step + 1 == config.bandit.updates:
            prediction = predict_adapter(model, validation_cache["features"][0],
                                         validation_cache["logits"][0], config.evaluation.batch_size, device)
            record["clean_validation"] = classification_metrics(
                prediction, validation_labels, config.evaluation.ece_bins)
        history.append(record)
        if (step + 1) % config.bandit.log_every == 0 or step == start:
            print(f"{name} step={step + 1}/{config.bandit.updates} "
                  f"feedback={record['feedback_count']} observed_reward={record['observed_reward']:.4f} "
                  f"kl={record['kl']:.5f}", flush=True)
        needs_save = ((step + 1) % config.bandit.checkpoint_every == 0 or
                      step + 1 == config.bandit.updates or
                      (stop_after is not None and step + 1 >= stop_after))
        if needs_save:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(device))
            committed_elapsed = elapsed + time.monotonic() - call_started
            atomic_torch(checkpoint_path, {
                "format_version": 1, "identity": identity, "method": name, "seed": seed,
                "step": step + 1, "adapter": model.state_dict(), "optimizer": optimizer.state_dict(),
                "processor": processor.state_dict(), "oracle": oracle.committed_state(record["feedback_count"]),
                "trace": trace, "history": history, "rng": rng_state(),
                "elapsed_seconds": committed_elapsed, "peak_cuda_bytes": peak_memory,
                "runtime": _runtime(device),
                "initial_adapter_hash": initial_hash,
            })
            atomic_json(destination / "history.json", history)
    completed = len(history) == config.bandit.updates
    if not completed:
        return model.eval(), {"complete": False, "updates": len(history), "identity": identity}
    audit = oracle.audit()
    atomic_npz(destination / "feedback_audit.npz", **audit, **{f"inferred_{k}": v for k, v in trace.items()})
    summary = {"complete": True, "method": name, "seed": seed, "identity": identity,
               "updates": config.bandit.updates, "initial_adapter_hash": initial_hash,
               "elapsed_seconds": committed_elapsed, "peak_cuda_bytes": peak_memory,
               "checkpoint_sha256": file_hash(checkpoint_path),
               "mechanism": audit_metrics(audit, trace)}
    for phase, specification in zip(summary["mechanism"]["phases"], config.bandit.phases):
        phase["name"] = specification.name
    atomic_json(destination / "training_complete.json", summary)
    return model.eval(), summary


def _runtime(device):
    """Record the numerical runtime whose checkpoint continuation is supported."""
    return {"torch": torch.__version__, "numpy": np.__version__, "device_type": device.type,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}


def _parameter_fingerprint(model):
    """Verify equal complete adapter initialization, including its hidden random projection."""
    import hashlib
    value = hashlib.sha256()
    for name, parameter in model.state_dict().items():
        value.update(name.encode())
        value.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()

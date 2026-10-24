"""Separate labeled calibration for an action-conditioned, frozen reference reward model."""

import torch
from torch.nn import functional as F


def fit_temperatures(logits, targets, config):
    """Fit one bounded temperature per context using only the dedicated calibration labels."""
    if logits.ndim != 3 or targets.shape != (logits.shape[1],):
        raise ValueError("Calibration requires context/example/class logits and aligned targets")
    temperatures, diagnostics = [], []
    for values in logits:
        initial_temperature = min(max(1.0, config.minimum_temperature), config.maximum_temperature)
        log_temperature = torch.tensor(initial_temperature).log().requires_grad_()
        optimizer = torch.optim.Adam([log_temperature], lr=config.learning_rate)
        initial = float(F.cross_entropy(values, targets))
        best_loss = float(F.cross_entropy(values / initial_temperature, targets))
        best_temperature = initial_temperature
        for _ in range(config.steps):
            optimizer.zero_grad()
            temperature = log_temperature.exp()
            loss = F.cross_entropy(values / temperature, targets)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                log_temperature.clamp_(
                    torch.tensor(config.minimum_temperature).log().item(),
                    torch.tensor(config.maximum_temperature).log().item())
                candidate = float(log_temperature.exp())
                candidate_loss = float(F.cross_entropy(values / candidate, targets))
            if candidate_loss < best_loss:
                best_loss, best_temperature = candidate_loss, candidate
        temperatures.append(best_temperature)
        diagnostics.append({"uncalibrated_nll": initial, "calibrated_nll": best_loss,
                            "temperature": best_temperature})
    return torch.tensor(temperatures), diagnostics


def correctness_probabilities(reference_logits, temperatures):
    """Predict all candidate-action correctness probabilities under each possible context."""
    return (reference_logits[None, :, :] / temperatures[:, None, None]).softmax(dim=-1)

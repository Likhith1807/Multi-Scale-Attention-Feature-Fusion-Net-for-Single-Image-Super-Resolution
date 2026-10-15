"""Losses and diagnostics for the controlled scale-selection experiment."""
from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class ResearchResult:
    total: torch.Tensor
    main_l1: torch.Tensor
    auxiliary_l1: torch.Tensor
    gate_cross_entropy: torch.Tensor
    gate_weight: float
    diagnostics: dict


def scheduled_gate_weight(update, maximum, warmup, ramp):
    """Zero through warmup, then linearly reach maximum over ``ramp`` updates."""
    if update <= warmup:
        return 0.0
    if ramp == 0:
        return float(maximum)
    return float(maximum) * min(1.0, (update - warmup) / ramp)


def local_branch_errors(candidates, target, scale):
    """Map candidate RGB L1 errors from HR pixels to the LR gate grid."""
    errors = torch.stack([
        (candidate.float() - target.float()).abs().mean(dim=1)
        for candidate in candidates
    ], dim=1)
    errors = F.avg_pool2d(errors, kernel_size=scale, stride=scale)
    return F.avg_pool2d(
        errors, kernel_size=3, stride=1, padding=1, count_include_pad=False
    )


def preference_targets(errors, temperature=1.0, epsilon=1e-6):
    """Detached, scale-normalized soft targets favoring lower-error branches."""
    if temperature <= 0 or epsilon <= 0:
        raise ValueError("temperature and epsilon must be positive")
    detached = errors.detach().float()
    normalizer = detached.mean(dim=1, keepdim=True)
    return (-detached / (temperature * normalizer + epsilon)).softmax(dim=1)


def research_objective(main, candidates, gate_logits, target, scale, experiment,
                       auxiliary_weight, maximum_gate_weight, gate_warmup,
                       gate_ramp, temperature, update):
    if experiment not in ("auxiliary", "error_supervised"):
        raise ValueError("research_objective requires a research experiment")
    if gate_logits.ndim == 4:
        # Backward-compatible support for the Version 1 final-block interface.
        gate_logits = gate_logits.unsqueeze(1)
    if gate_logits.ndim != 5:
        raise ValueError("gate_logits must have shape N x depth x branches x H x W")
    if len(candidates) != gate_logits.shape[2]:
        raise ValueError("Candidate count and gate branch count differ")
    main_l1 = F.l1_loss(main.float(), target.float())
    branch_losses = torch.stack([
        F.l1_loss(candidate.float(), target.float()) for candidate in candidates
    ])
    auxiliary_l1 = branch_losses.mean()
    errors = local_branch_errors(candidates, target, scale)
    preferences = preference_targets(errors, temperature)
    log_weights = F.log_softmax(gate_logits.float(), dim=2)
    gate_ce = -(preferences.unsqueeze(1) * log_weights).sum(dim=2).mean()
    current_gate_weight = (
        scheduled_gate_weight(update, maximum_gate_weight, gate_warmup, gate_ramp)
        if experiment == "error_supervised" else 0.0
    )
    total = main_l1 + auxiliary_weight * auxiliary_l1
    if current_gate_weight:
        total = total + current_gate_weight * gate_ce
    weights = log_weights.exp()
    with torch.no_grad():
        gate_dims = (0, 1, 3, 4)
        target_dims = (0, 2, 3)
        gate_entropy = -(weights * log_weights).sum(dim=2).mean()
        target_logs = preferences.clamp_min(1e-12).log()
        target_entropy = -(preferences * target_logs).sum(dim=1).mean()
        mean_weights = weights.mean(dim=1)
        regret = ((mean_weights * errors).sum(dim=1) - errors.min(dim=1).values).mean()
        agreement = (
            weights.argmax(dim=2) == preferences.argmax(dim=1).unsqueeze(1)
        ).float().mean()
        diagnostics = {
            "branch_l1": [float(value) for value in branch_losses.detach().cpu()],
            "gate_usage": [float(value) for value in weights.mean(dim=gate_dims).cpu()],
            "target_usage": [float(value) for value in preferences.mean(dim=target_dims).cpu()],
            "gate_entropy": float(gate_entropy.cpu()),
            "target_entropy": float(target_entropy.cpu()),
            "branch_regret": float(regret.cpu()),
            "gate_target_agreement": float(agreement.cpu()),
        }
    return ResearchResult(
        total=total, main_l1=main_l1, auxiliary_l1=auxiliary_l1,
        gate_cross_entropy=gate_ce, gate_weight=current_gate_weight,
        diagnostics=diagnostics,
    )

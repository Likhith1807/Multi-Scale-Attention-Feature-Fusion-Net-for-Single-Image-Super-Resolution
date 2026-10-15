"""Diagnose forced-branch candidate diversity without retraining the model."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from evaluate import save_json
from pairs import crop_pair, find_pairs, read_pair


def summarize_errors(errors, temperature):
    """Summarize N x K x H x W local branch errors."""
    import torch
    from research import preference_targets
    targets = preference_targets(errors, temperature)
    winners = errors.argmin(dim=1)
    branch_count = errors.shape[1]
    dims = (0, 2, 3)
    entropy = -(targets * targets.clamp_min(1e-12).log()).sum(dim=1)
    spread = errors.max(dim=1).values - errors.min(dim=1).values
    relative_spread = spread / errors.mean(dim=1).clamp_min(1e-12)
    sorted_errors = errors.sort(dim=1).values
    relative_margin = (
        (sorted_errors[:, 1] - sorted_errors[:, 0]) /
        errors.mean(dim=1).clamp_min(1e-12)
    )
    uniform_error = errors.mean(dim=1)
    oracle_error = errors.min(dim=1).values
    horizontal = (winners[:, :, 1:] == winners[:, :, :-1]).float().mean()
    vertical = (winners[:, 1:, :] == winners[:, :-1, :]).float().mean()
    winner_usage = [
        float((winners == index).float().mean().cpu())
        for index in range(branch_count)
    ]
    return {
        "branch_mean_errors": [float(value) for value in errors.mean(dim=dims).cpu()],
        "mean_absolute_spread": float(spread.mean().cpu()),
        "mean_relative_spread": float(relative_spread.mean().cpu()),
        "relative_margin_values": relative_margin.detach().cpu().reshape(-1).numpy(),
        "target_usage": [float(value) for value in targets.mean(dim=dims).cpu()],
        "winner_usage": winner_usage,
        "target_entropy": float(entropy.mean().cpu()),
        "normalized_target_entropy": float((entropy.mean() / math.log(branch_count)).cpu()),
        "oracle_error": float(oracle_error.mean().cpu()),
        "uniform_error": float(uniform_error.mean().cpu()),
        "oracle_improvement": float((uniform_error - oracle_error).mean().cpu()),
        "winner_neighbor_agreement": float(((horizontal + vertical) / 2).cpu()),
    }


def local_gradient_errors(candidates, target, scale):
    """Map HR finite-difference errors to the LR grid."""
    import torch
    from torch.nn import functional as F

    def gradients(value):
        dx = F.pad(value[:, :, :, 1:] - value[:, :, :, :-1], (0, 1, 0, 0))
        dy = F.pad(value[:, :, 1:, :] - value[:, :, :-1, :], (0, 0, 0, 1))
        return dx, dy

    target_dx, target_dy = gradients(target.float())
    maps = []
    for candidate in candidates:
        candidate_dx, candidate_dy = gradients(candidate.float())
        error = (
            (candidate_dx - target_dx).abs() +
            (candidate_dy - target_dy).abs()
        ).mean(dim=1) / 2
        maps.append(error)
    errors = F.avg_pool2d(
        torch.stack(maps, dim=1), kernel_size=scale, stride=scale
    )
    return F.avg_pool2d(
        errors, kernel_size=3, stride=1, padding=1, count_include_pad=False
    )


def finish_summary(accumulator, image_count):
    result = {}
    for domain in ("rgb", "gradient"):
        values = accumulator[domain]
        margins = np.concatenate(values.pop("relative_margin_values"))
        summary = {
            key: (
                np.mean(value, axis=0).tolist()
                if isinstance(value[0], list) else float(np.mean(value))
            )
            for key, value in values.items()
        }
        summary["relative_margin_quantiles"] = {
            str(percent): float(np.percentile(margins, percent))
            for percent in (50, 75, 90, 95, 99)
        }
        summary["fraction_relative_margin_above"] = {
            str(threshold): float(np.mean(margins > threshold))
            for threshold in (.001, .005, .01, .05)
        }
        result[f"{domain}_error_diagnostics"] = summary
    pairwise = accumulator["pairwise_output_l1"]
    result["pairwise_output_l1"] = {
        "pairs": ["branch_1_vs_2", "branch_1_vs_3", "branch_2_vs_3"],
        "mean": np.mean(pairwise, axis=0).tolist(),
        "overall_mean": float(np.mean(pairwise)),
    }
    result["candidate_global_l1"] = np.mean(
        accumulator["candidate_global_l1"], axis=0
    ).tolist()
    result["main_global_l1"] = float(np.mean(accumulator["main_global_l1"]))
    result["images"] = image_count
    result["branches"] = len(accumulator["candidate_global_l1"][0])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/research_baseline_3k_s2026/best.pt")
    parser.add_argument("--data", default="data")
    parser.add_argument("--depths", type=int, nargs="+", default=(1, 3, 6))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--crop", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", default="runs/candidate_depth_diagnostic_5060.json")
    args = parser.parse_args()
    if args.limit <= 0 or args.crop < 0 or args.temperature <= 0:
        parser.error("limit and temperature must be positive; crop must be nonnegative")
    if not args.depths or any(depth <= 0 for depth in args.depths):
        parser.error("depths must contain positive integers")

    import torch
    from model import CompactMSAF
    from research import local_branch_errors
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; run check_environment.py.")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        parser.error(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["model_config"]
    run_config = checkpoint.get("config", {})
    if config.get("fusion") != "learned":
        parser.error("Candidate routing diagnostics require learned fusion.")
    if run_config.get("experiment", "baseline") != "baseline":
        parser.error("Use the matched baseline checkpoint, not a research-variant checkpoint.")
    depths = tuple(dict.fromkeys(args.depths))
    if max(depths) > config["blocks"]:
        parser.error(f"Maximum depth is {config['blocks']} for this checkpoint.")

    device = torch.device(args.device)
    model = CompactMSAF(**config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    scale = config["scale"]
    root = Path(args.data)
    pairs = find_pairs(
        root / f"DIV2K_valid_LR_bicubic/X{scale}",
        root / "DIV2K_valid_HR", scale, args.limit,
    )
    accumulators = {}
    for depth in depths:
        accumulators[depth] = {
            "rgb": {}, "gradient": {}, "pairwise_output_l1": [],
            "candidate_global_l1": [], "main_global_l1": [],
        }

    with torch.inference_mode():
        for image_index, pair in enumerate(pairs, 1):
            low, high = read_pair(pair, scale)
            low, high = crop_pair(low, high, scale, args.crop)
            low_tensor = torch.from_numpy(
                low.transpose(2, 0, 1).copy()
            ).float()[None].to(device) / 255
            high_tensor = torch.from_numpy(
                high.transpose(2, 0, 1).copy()
            ).float()[None].to(device) / 255
            main_prediction = model(low_tensor)
            for depth in depths:
                candidates = model.forward_forced_paths(low_tensor, depth)
                if not all(torch.isfinite(value).all().item() for value in candidates):
                    raise RuntimeError(f"Non-finite candidate for {pair.name}, depth {depth}")
                accumulator = accumulators[depth]
                accumulator["main_global_l1"].append(float(
                    torch.nn.functional.l1_loss(main_prediction, high_tensor).cpu()
                ))
                accumulator["candidate_global_l1"].append([
                    float(torch.nn.functional.l1_loss(value, high_tensor).cpu())
                    for value in candidates
                ])
                accumulator["pairwise_output_l1"].append([
                    float((candidates[a] - candidates[b]).abs().mean().cpu())
                    for a, b in ((0, 1), (0, 2), (1, 2))
                ])
                diagnostics = {
                    "rgb": summarize_errors(
                        local_branch_errors(candidates, high_tensor, scale), args.temperature
                    ),
                    "gradient": summarize_errors(
                        local_gradient_errors(candidates, high_tensor, scale), args.temperature
                    ),
                }
                for domain, values in diagnostics.items():
                    for key, value in values.items():
                        accumulator[domain].setdefault(key, []).append(value)
            print(f"Analyzed {image_index}/{len(pairs)}: {pair.name}", flush=True)

    report = {
        "purpose": "candidate-depth diversity diagnostic; no training or superiority claim",
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint.get("step"),
        "model_config": config,
        "protocol": {
            "dataset": "DIV2K validation subset",
            "image_ids": [pair.name for pair in pairs],
            "region": f"center_{args.crop}_lr" if args.crop else "full_image",
            "depths": list(depths),
            "temperature": args.temperature,
            "precision": "FP32 inference",
            "candidate_definition": (
                "candidate k forces dilation branch k through every one of the "
                "last depth blocks; prefix and decoder are shared"
            ),
            "aggregation": "arithmetic mean of per-image diagnostics",
        },
        "maximum_target_entropy": math.log(3),
        "depth_results": {
            str(depth): finish_summary(accumulators[depth], len(pairs))
            for depth in depths
        },
        "interpretation_rules": [
            "Normalized target entropy near 1 means branch-error targets are uninformative.",
            "Larger pairwise output differences alone are insufficient; oracle error improvement must also grow.",
            "Winner usage and neighbor agreement describe preferences, not proof of semantic experts.",
            "This diagnostic chooses a mechanism for later matched training; it is not benchmark evidence.",
        ],
    }
    save_json(args.output, report)
    print(json.dumps(report, indent=2))
    print(f"\nDiagnostic saved to {args.output}")


if __name__ == "__main__":
    main()

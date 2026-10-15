"""Check matched experiment runs and summarize their development validation."""
import argparse
import json
from pathlib import Path

from evaluate import save_json


def load_run(path):
    path = Path(path)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (path / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    training = [row for row in rows if row.get("type") == "train"]
    validation = [row for row in rows if row.get("type") == "validation"]
    if not training or not validation:
        raise ValueError(f"Missing training or validation records in {path}")
    expected_step = config["steps"]
    if training[-1].get("step") != expected_step or validation[-1].get("step") != expected_step:
        raise ValueError(
            f"Run {path} is incomplete: expected final update {expected_step}, "
            f"found training {training[-1].get('step')} and validation "
            f"{validation[-1].get('step')}"
        )
    if not (path / "last.pt").is_file():
        raise ValueError(f"Completed checkpoint is missing from {path}")
    return path, config, manifest, training, validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="runs/research_baseline_3k_s2026")
    parser.add_argument("--auxiliary", default="runs/research_auxiliary_3k_s2026")
    parser.add_argument("--proposed", default="runs/research_error_3k_s2026")
    parser.add_argument("--output", default="runs/research_3k_comparison.json")
    args = parser.parse_args()
    loaded = {
        "baseline": load_run(args.baseline),
        "auxiliary": load_run(args.auxiliary),
        "error_supervised": load_run(args.proposed),
    }
    expected = {name: value[1]["experiment"] for name, value in loaded.items()}
    if expected != {"baseline": "baseline", "auxiliary": "auxiliary",
                    "error_supervised": "error_supervised"}:
        raise ValueError(f"Experiment labels do not match folders: {expected}")
    common_keys = ("scale", "features", "blocks", "fusion", "steps", "micro_batch",
            "accumulate", "patch", "train_limit", "valid_limit", "eval_crop",
            "seed", "learning_rate", "validate_every", "fixed_patch", "amp",
            "device", "workers")
    reference = {key: loaded["baseline"][1][key] for key in common_keys}
    mismatches = {
        name: {key: config[key] for key in common_keys if config[key] != reference[key]}
        for name, (_, config, _, _, _) in loaded.items()
    }
    mismatches = {name: values for name, values in mismatches.items() if values}
    if mismatches:
        raise ValueError(f"Runs are not matched: {mismatches}")
    research_keys = ("aux_weight", "gate_weight", "gate_warmup", "gate_ramp",
                     "temperature", "research_depth")
    auxiliary_config = loaded["auxiliary"][1]
    proposed_config = loaded["error_supervised"][1]
    research_reference = {
        key: auxiliary_config.get(key, 1 if key == "research_depth" else None)
        for key in research_keys
    }
    research_mismatches = {
        key: proposed_config.get(key, 1 if key == "research_depth" else None)
        for key in research_keys
        if proposed_config.get(key, 1 if key == "research_depth" else None)
        != research_reference[key]
    }
    if research_mismatches:
        raise ValueError(
            "Auxiliary and proposed research settings differ: "
            f"{research_mismatches}"
        )
    manifests = {name: (value[2]["train"]["sha256"], value[2]["validation"]["sha256"])
                 for name, value in loaded.items()}
    if len(set(manifests.values())) != 1:
        raise ValueError("Runs use different selected dataset bytes")
    summary = {}
    for name, (path, config, _, training, validation) in loaded.items():
        best = max(validation, key=lambda row: row["psnr_y"])
        final = validation[-1]
        summary[name] = {
            "path": str(path), "best": best, "final": final,
            "mean_train_seconds_per_update": sum(r["seconds"] for r in training) / len(training),
            "aux_weight": config.get("aux_weight"),
            "maximum_gate_weight": config.get("gate_weight"),
            "gate_warmup": config.get("gate_warmup"),
            "gate_ramp": config.get("gate_ramp"),
            "temperature": config.get("temperature"),
            "research_depth": config.get("research_depth", 1),
        }
    summary["best_psnr_deltas"] = {
        "auxiliary_minus_baseline": summary["auxiliary"]["best"]["psnr_y"] - summary["baseline"]["best"]["psnr_y"],
        "error_supervised_minus_auxiliary": summary["error_supervised"]["best"]["psnr_y"] - summary["auxiliary"]["best"]["psnr_y"],
        "error_supervised_minus_baseline": summary["error_supervised"]["best"]["psnr_y"] - summary["baseline"]["best"]["psnr_y"],
    }
    summary["best_ssim_deltas"] = {
        "auxiliary_minus_baseline": summary["auxiliary"]["best"]["ssim_y"] - summary["baseline"]["best"]["ssim_y"],
        "error_supervised_minus_auxiliary": summary["error_supervised"]["best"]["ssim_y"] - summary["auxiliary"]["best"]["ssim_y"],
        "error_supervised_minus_baseline": summary["error_supervised"]["best"]["ssim_y"] - summary["baseline"]["best"]["ssim_y"],
    }
    result = {
        "status": "matched", "matched_config": {
            "common": reference, "research_variants": research_reference,
        },
        "manifest_hashes": manifests, "summary": summary,
        "limitations": [
            "One development seed and short schedules cannot establish superiority.",
            "Best-checkpoint selection uses the same development images being summarized.",
            "Full-image per-source results and final benchmark sets remain required.",
        ],
    }
    save_json(args.output, result)
    print(json.dumps(summary, indent=2))
    print(f"\nComparison saved to {args.output}")


if __name__ == "__main__":
    main()

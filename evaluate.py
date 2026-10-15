"""Evaluate a saved CompactMSAF model on explicitly paired PNG folders."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from metrics import psnr_ssim
from pairs import crop_pair, find_pairs, read_pair


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def evaluate_model(model, pairs, scale, device, crop=0, output=None, save_images=3):
    import torch
    from torch.nn import functional as F
    was_training = model.training
    model.eval()
    rows = []
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
    try:
        with torch.inference_mode():
            for index, pair in enumerate(pairs):
                low, high = read_pair(pair, scale)
                low, high = crop_pair(low, high, scale, crop)
                tensor = torch.from_numpy(low.transpose(2, 0, 1).copy()).float()[None].to(device) / 255
                prediction = model(tensor)
                if not torch.isfinite(prediction).all().item():
                    raise RuntimeError(f"Non-finite output for {pair.name}")
                bicubic = F.interpolate(tensor, scale_factor=scale, mode="bicubic", align_corners=False)

                def rgb(array):
                    return np.rint(array[0].float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

                sr, base = rgb(prediction), rgb(bicubic)
                score = psnr_ssim(sr, high, scale)
                interpolation_score = psnr_ssim(base, high, scale)
                rows.append({
                    "id": pair.name, **score,
                    "torch_bicubic_psnr_y": interpolation_score["psnr_y"],
                    "torch_bicubic_ssim_y": interpolation_score["ssim_y"],
                })
                if output is not None and index < save_images:
                    Image.fromarray(sr).save(output / f"{pair.name}_sr.png")
                    Image.fromarray(high).save(output / f"{pair.name}_hr.png")
                    Image.fromarray(base).save(output / f"{pair.name}_torch_bicubic.png")
    finally:
        model.train(was_training)
    if not rows:
        raise ValueError("No evaluation images")
    keys = ("psnr_y", "ssim_y", "torch_bicubic_psnr_y", "torch_bicubic_ssim_y")
    result = {
        "count": len(rows),
        "mean": {key: float(np.mean([r[key] for r in rows])) for key in keys},
        "protocol": {
            "region": f"center_{crop}_lr" if crop else "full_image",
            "scale": scale, "border_hr_pixels": scale,
            "color": "BT.601 limited-range Y from quantized uint8 RGB",
            "ssim": "11x11 Gaussian, sigma 1.5, population moments, valid windows",
            "aggregation": "arithmetic mean of per-image metrics",
            "interpolation_reference": "PyTorch bicubic; not MATLAB benchmark interpolation",
            "precision": "FP32 inference",
        },
        "per_image": rows,
    }
    if output is not None:
        save_json(output / "metrics.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--lr-dir", required=True)
    parser.add_argument("--hr-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--crop", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", default="runs/evaluation")
    args = parser.parse_args()
    import torch
    from model import CompactMSAF
    if args.crop < 0 or args.limit < 0:
        parser.error("crop and limit must be nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; run check_environment.py.")
    # This package stores only tensors and ordinary Python primitives.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = CompactMSAF(**checkpoint["model_config"]).to(args.device)
    model.load_state_dict(checkpoint["model"])
    scale = checkpoint["model_config"]["scale"]
    pairs = find_pairs(args.lr_dir, args.hr_dir, scale, args.limit)
    destination = Path(args.output)
    if destination.exists() and any(destination.iterdir()):
        parser.error("Choose an empty output folder so evaluation results are not mixed.")
    result = evaluate_model(model, pairs, scale, args.device, args.crop, destination)
    print(json.dumps(json_safe(result["mean"]), indent=2))


if __name__ == "__main__":
    main()

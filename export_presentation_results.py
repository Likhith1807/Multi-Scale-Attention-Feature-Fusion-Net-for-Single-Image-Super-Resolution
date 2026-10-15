"""Export presentation-ready qualitative comparisons and result charts.

The qualitative panels use the pilot checkpoint because the reported
29.4766 dB / 0.79723 pilot result was produced by that checkpoint.  A zoom
region is selected deterministically from Bicubic-vs-HR error only; the model
output is never used to choose the displayed region.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from metrics import psnr_ssim
from pairs import find_pairs, read_pair


BACKGROUND = "#f6f9fc"
INK = "#17324f"
MUTED = "#587086"
BLUE = "#2b7aaa"
GREEN = "#24905a"
ORANGE = "#d98200"
GRID = "#d8e4ed"


def font(size, bold=False):
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
             else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def text_center(draw, xy, value, selected_font, fill=INK):
    x, y = xy
    box = draw.textbbox((0, 0), value, font=selected_font)
    draw.text((x - (box[2] - box[0]) / 2, y), value, font=selected_font, fill=fill)


def uint8_rgb(tensor):
    return np.rint(
        tensor[0].float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255
    ).astype(np.uint8)


def candidate_positions(length, patch, stride):
    values = list(range(0, max(1, length - patch + 1), stride))
    final = max(0, length - patch)
    if not values or values[-1] != final:
        values.append(final)
    return values


def bicubic_error_crop(bicubic, high, patch=128, stride=32):
    """Select a deterministic high-error crop without looking at model SR."""
    height, width = high.shape[:2]
    patch = min(patch, height, width)
    error = np.abs(bicubic.astype(np.float32) - high.astype(np.float32)).mean(axis=2)
    integral = np.pad(error, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    best_score, best = -1.0, (0, 0)
    for top in candidate_positions(height, patch, stride):
        bottom = top + patch
        for left in candidate_positions(width, patch, stride):
            right = left + patch
            total = integral[bottom, right] - integral[top, right]
            total -= integral[bottom, left] - integral[top, left]
            score = float(total) / (patch * patch)
            if score > best_score:
                best_score, best = score, (left, top)
    return (*best, patch)


def paste_contained(canvas, array, box, crop=None):
    x, y, width, height = box
    source = Image.fromarray(array)
    ratio = min(width / source.width, height / source.height)
    resized = source.resize(
        (max(1, round(source.width * ratio)), max(1, round(source.height * ratio))),
        Image.Resampling.LANCZOS,
    )
    px = x + (width - resized.width) // 2
    py = y + (height - resized.height) // 2
    canvas.paste(resized, (px, py))
    if crop is not None:
        left, top, patch = crop
        draw = ImageDraw.Draw(canvas)
        x0 = px + round(left * ratio)
        y0 = py + round(top * ratio)
        x1 = px + round((left + patch) * ratio)
        y1 = py + round((top + patch) * ratio)
        draw.rectangle((x0, y0, x1, y1), outline=ORANGE, width=4)


def presentation_panel(image_id, low, bicubic, sr, high, score, base_score, output):
    width, height = 1920, 1080
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    title_font, label_font = font(37, True), font(25, True)
    body_font, small_font = font(22), font(18)
    text_center(draw, (width // 2, 22),
                f"DIV2K {image_id}: ×4 Super-Resolution Comparison", title_font)

    labels = ("LR Input", "Bicubic ×4", "Compact MSAF", "Ground-Truth HR")
    arrays = (
        np.asarray(Image.fromarray(low).resize(
            (high.shape[1], high.shape[0]), Image.Resampling.NEAREST)),
        bicubic,
        sr,
        high,
    )
    crop = bicubic_error_crop(bicubic, high)
    margin, gap = 60, 20
    cell_width = (width - 2 * margin - 3 * gap) // 4
    full_top, full_height = 120, 350
    zoom_top, zoom_height = 550, 330

    for index, (label, array) in enumerate(zip(labels, arrays, strict=True)):
        left = margin + index * (cell_width + gap)
        text_center(draw, (left + cell_width // 2, 82), label, label_font)
        draw.rounded_rectangle(
            (left, full_top, left + cell_width, full_top + full_height),
            radius=12, fill="white", outline=GRID, width=2,
        )
        paste_contained(canvas, array, (left + 10, full_top + 10,
                                        cell_width - 20, full_height - 20), crop)
        text_center(draw, (left + cell_width // 2, 505), "Identical zoomed region", small_font, MUTED)
        crop_left, crop_top, crop_size = crop
        zoom = array[crop_top:crop_top + crop_size, crop_left:crop_left + crop_size]
        zoom_size = min(cell_width - 20, zoom_height - 20)
        zoom_image = Image.fromarray(zoom).resize(
            (zoom_size, zoom_size), Image.Resampling.NEAREST
        )
        draw.rounded_rectangle(
            (left, zoom_top, left + cell_width, zoom_top + zoom_height),
            radius=12, fill="white", outline=GRID, width=2,
        )
        canvas.paste(
            zoom_image,
            (left + (cell_width - zoom_size) // 2,
             zoom_top + (zoom_height - zoom_size) // 2),
        )

    footer_y = 915
    draw.rounded_rectangle((190, footer_y, 1730, 1030), radius=16,
                           fill="white", outline=GRID, width=2)
    text_center(draw, (630, footer_y + 20),
                f"Bicubic: {base_score['psnr_y']:.4f} dB | SSIM {base_score['ssim_y']:.5f}",
                body_font, MUTED)
    text_center(draw, (1290, footer_y + 20),
                f"Compact MSAF: {score['psnr_y']:.4f} dB | SSIM {score['ssim_y']:.5f}",
                body_font, GREEN)
    text_center(draw, (960, footer_y + 62),
                "Zoom chosen using Bicubic–HR error only; the model output was not used for selection.",
                small_font, MUTED)
    canvas.save(output)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validation_history(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("type") == "validation":
            rows.append(row)
    if not rows:
        raise ValueError(f"No validation rows in {path}")
    return rows


def line_chart(rows, output):
    canvas = Image.new("RGB", (1600, 900), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    title_font, axis_font, value_font = font(38, True), font(23), font(20, True)
    text_center(draw, (800, 30), "Baseline Validation Convergence", title_font)
    left, top, right, bottom = 150, 120, 1520, 730
    xs = [row["step"] for row in rows]
    ys = [row["psnr_y"] for row in rows]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys) - 0.03, max(ys) + 0.03

    def point(x, y):
        px = left + (x - x_min) / max(1, x_max - x_min) * (right - left)
        py = bottom - (y - y_min) / (y_max - y_min) * (bottom - top)
        return px, py

    for index in range(6):
        value = y_min + index * (y_max - y_min) / 5
        py = point(x_min, value)[1]
        draw.line((left, py, right, py), fill=GRID, width=2)
        draw.text((35, py - 12), f"{value:.3f}", font=axis_font, fill=MUTED)
    draw.line((left, top, left, bottom), fill=INK, width=3)
    draw.line((left, bottom, right, bottom), fill=INK, width=3)
    points = [point(x, y) for x, y in zip(xs, ys, strict=True)]
    draw.line(points, fill=BLUE, width=6)
    for (px, py), step, value in zip(points, xs, ys, strict=True):
        draw.ellipse((px - 8, py - 8, px + 8, py + 8), fill=BLUE)
        text_center(draw, (px, py - 38), f"{value:.4f}", value_font)
        text_center(draw, (px, bottom + 18), f"{step:,}", axis_font, MUTED)
    text_center(draw, (835, 805), "Optimizer updates", axis_font)
    draw.text((22, 80), "Y-PSNR (dB)", font=axis_font, fill=INK)
    gain = ys[-1] - ys[0]
    text_center(draw, (800, 850),
                f"Stable improvement: +{gain:.4f} dB from update {xs[0]:,} to {xs[-1]:,}",
                value_font, GREEN)
    canvas.save(output)


def ablation_chart(path, output):
    data = load_json(path)
    summary = data.get("summary", data)
    labels = ("Baseline", "Auxiliary depth 6", "Error-supervised depth 6")
    values = (
        summary["baseline"]["best"]["psnr_y"],
        summary["auxiliary"]["best"]["psnr_y"],
        summary["error_supervised"]["best"]["psnr_y"],
    )
    canvas = Image.new("RGB", (1600, 900), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    title_font, axis_font, value_font = font(38, True), font(23), font(22, True)
    text_center(draw, (800, 30), "Matched Depth-6 Ablation: Best Validation Y-PSNR", title_font)
    left, top, right, bottom = 160, 130, 1520, 720
    y_min, y_max = min(values) - 0.003, max(values) + 0.003
    for index in range(6):
        value = y_min + index * (y_max - y_min) / 5
        py = bottom - (value - y_min) / (y_max - y_min) * (bottom - top)
        draw.line((left, py, right, py), fill=GRID, width=2)
        draw.text((35, py - 12), f"{value:.4f}", font=axis_font, fill=MUTED)
    draw.line((left, top, left, bottom), fill=INK, width=3)
    draw.line((left, bottom, right, bottom), fill=INK, width=3)
    centers = (420, 840, 1260)
    colors = (GREEN, BLUE, ORANGE)
    for center, label, value, color in zip(centers, labels, values, colors, strict=True):
        py = bottom - (value - y_min) / (y_max - y_min) * (bottom - top)
        draw.rounded_rectangle((center - 115, py, center + 115, bottom),
                               radius=8, fill=color)
        text_center(draw, (center, py - 42), f"{value:.4f} dB", value_font)
        text_center(draw, (center, bottom + 28), label, axis_font)
    text_center(draw, (800, 815),
                "Zoomed axis shown transparently; neither routing variant exceeds the baseline.",
                value_font, MUTED)
    canvas.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/baseline_x4_pilot_5060/best.pt")
    parser.add_argument("--lr-dir", default="data/DIV2K_valid_LR_bicubic/X4")
    parser.add_argument("--hr-dir", default="data/DIV2K_valid_HR")
    parser.add_argument("--history", default="runs/research_baseline_3k_s2026/history.jsonl")
    parser.add_argument("--ablation", default="runs/research_depth6_3k_comparison.json")
    parser.add_argument("--output", default="runs/presentation_results_5060")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")

    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        parser.error(f"{output} is not empty. Use --overwrite or choose another --output.")
    output.mkdir(parents=True, exist_ok=True)

    import torch
    from torch.nn import functional as F
    from model import CompactMSAF
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu or check the environment.")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = CompactMSAF(**checkpoint["model_config"]).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    scale = checkpoint["model_config"]["scale"]
    pairs = find_pairs(args.lr_dir, args.hr_dir, scale, args.count)
    records = []
    with torch.inference_mode():
        for pair in pairs:
            low, high = read_pair(pair, scale)
            tensor = torch.from_numpy(low.transpose(2, 0, 1).copy()).float()[None]
            tensor = tensor.to(args.device) / 255
            prediction = model(tensor)
            bicubic_tensor = F.interpolate(
                tensor, scale_factor=scale, mode="bicubic", align_corners=False
            )
            sr, bicubic = uint8_rgb(prediction), uint8_rgb(bicubic_tensor)
            score = psnr_ssim(sr, high, scale)
            base_score = psnr_ssim(bicubic, high, scale)
            destination = output / f"{pair.name}_comparison.png"
            presentation_panel(pair.name, low, bicubic, sr, high,
                               score, base_score, destination)
            records.append({
                "id": pair.name,
                "comparison": destination.name,
                "model": score,
                "bicubic": base_score,
            })
            print(f"Saved {destination}")

    history_path = Path(args.history)
    if history_path.is_file():
        destination = output / "training_convergence.png"
        line_chart(validation_history(history_path), destination)
        print(f"Saved {destination}")
    else:
        print(f"Skipped convergence chart: missing {history_path}")

    ablation_path = Path(args.ablation)
    if ablation_path.is_file():
        destination = output / "depth6_ablation_psnr.png"
        ablation_chart(ablation_path, destination)
        print(f"Saved {destination}")
    else:
        print(f"Skipped ablation chart: missing {ablation_path}")

    report = {
        "checkpoint": str(Path(args.checkpoint)),
        "model_config": checkpoint["model_config"],
        "selection": "first sorted DIV2K validation image IDs",
        "zoom_selection": "maximum Bicubic-vs-HR RGB MAE; model output excluded",
        "images": records,
    }
    (output / "presentation_results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Presentation results saved to {output.resolve()}")


if __name__ == "__main__":
    main()

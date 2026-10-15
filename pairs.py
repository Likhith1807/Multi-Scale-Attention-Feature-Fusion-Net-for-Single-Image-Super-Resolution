"""Paired PNG handling, independent of PyTorch so data can be checked first."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Pair:
    name: str
    lr: Path
    hr: Path


def find_pairs(lr_dir, hr_dir, scale=4, limit=0):
    lr_dir, hr_dir = Path(lr_dir), Path(hr_dir)
    if not lr_dir.is_dir() or not hr_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {lr_dir} or {hr_dir}")
    low = {}
    suffix = f"x{scale}"
    for path in sorted(lr_dir.glob("*.png")):
        key = path.stem[:-len(suffix)] if path.stem.endswith(suffix) else path.stem
        if key in low:
            raise ValueError(f"Duplicate LR image ID: {key}")
        low[key] = path
    high = {p.stem: p for p in sorted(hr_dir.glob("*.png"))}
    if not high or not low:
        raise ValueError("Both image folders must contain PNG files.")
    if set(low) != set(high):
        raise ValueError(
            f"Unmatched image IDs. Missing LR: {sorted(set(high)-set(low))[:5]}; "
            f"missing HR: {sorted(set(low)-set(high))[:5]}"
        )
    pairs = [Pair(k, low[k], high[k]) for k in sorted(high)]
    return pairs[:limit] if limit > 0 else pairs


def check_disjoint(train, valid):
    overlap = {p.name for p in train} & {p.name for p in valid}
    paths = {p.hr.resolve() for p in train} & {p.hr.resolve() for p in valid}
    if overlap or paths:
        raise ValueError(f"Training and validation overlap: {sorted(overlap)[:5]}")


def read_pair(pair, scale):
    with Image.open(pair.lr) as image:
        low = np.asarray(image.convert("RGB")).copy()
    with Image.open(pair.hr) as image:
        high = np.asarray(image.convert("RGB")).copy()
    height, width = low.shape[:2]
    dh, dw = high.shape[0] - height * scale, high.shape[1] - width * scale
    if not (0 <= dh < scale and 0 <= dw < scale):
        raise ValueError(
            f"Unaligned pair {pair.name}: LR={low.shape}, HR={high.shape}, scale={scale}"
        )
    # Only trim the <scale HR remainder at bottom/right; never resize targets.
    return low, high[:height * scale, :width * scale]


def crop_pair(low, high, scale, patch, rng=None, augment=False):
    height, width = low.shape[:2]
    if high.shape[:2] != (height * scale, width * scale):
        raise ValueError("crop_pair expects aligned images.")
    if patch < 0 or (patch and min(height, width) < patch):
        raise ValueError(f"Patch {patch} does not fit LR image {height}x{width}.")
    if patch:
        if rng is None:
            top, left = (height - patch) // 2, (width - patch) // 2
        else:
            top = int(rng.integers(height - patch + 1))
            left = int(rng.integers(width - patch + 1))
        low = low[top:top + patch, left:left + patch]
        high = high[top * scale:(top + patch) * scale,
                    left * scale:(left + patch) * scale]
    if augment:
        if rng is None:
            raise ValueError("Augmentation requires an explicit generator.")
        if rng.random() < .5:
            low, high = low[:, ::-1], high[:, ::-1]
        if rng.random() < .5:
            low, high = low[::-1], high[::-1]
        if rng.random() < .5:
            low, high = low.transpose(1, 0, 2), high.transpose(1, 0, 2)
    return np.ascontiguousarray(low), np.ascontiguousarray(high)


def fingerprint(pairs):
    rows = []
    for pair in pairs:
        row = {"id": pair.name}
        for key, path in (("lr_sha256", pair.lr), ("hr_sha256", pair.hr)):
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            row[key] = digest.hexdigest()
        rows.append(row)
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "images": rows}


class PatchDataset:
    """Index-derived sampling stays stable across worker counts and resumes."""
    def __init__(self, pairs, scale, patch, seed, samples, fixed_patch=False):
        self.pairs, self.scale, self.patch = pairs, scale, patch
        self.seed, self.samples = seed, samples
        self.fixed_patch = fixed_patch

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        # Imported only when training: data/metric checks work without torch.
        import torch
        rng = np.random.default_rng(np.random.SeedSequence([
            self.seed, 0 if self.fixed_patch else int(index)
        ]))
        pair = self.pairs[0 if self.fixed_patch else int(rng.integers(len(self.pairs)))]
        low, high = read_pair(pair, self.scale)
        low, high = crop_pair(low, high, self.scale, self.patch, rng, not self.fixed_patch)
        return (
            torch.from_numpy(low.transpose(2, 0, 1).copy()).float() / 255,
            torch.from_numpy(high.transpose(2, 0, 1).copy()).float() / 255,
        )

"""Meaningful local checks. GPU checks still require a CUDA runtime."""
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image

from download_div2k import extract_pngs
from metrics import psnr_ssim, y_channel
from pairs import Pair, check_disjoint, crop_pair, find_pairs, read_pair


HAS_TORCH = importlib.util.find_spec("torch") is not None
ROOT = Path(__file__).resolve().parent


class DataAndMetricTests(unittest.TestCase):
    def test_aligned_crop_and_augmentation(self):
        low = np.random.default_rng(5).integers(0, 256, (17, 19, 3), dtype=np.uint8)
        high = np.repeat(np.repeat(low, 4, axis=0), 4, axis=1)
        for seed in range(10):
            lr, hr = crop_pair(low, high, 4, 8, np.random.default_rng(seed), True)
            np.testing.assert_array_equal(hr, np.repeat(np.repeat(lr, 4, 0), 4, 1))

    def test_pair_matching_and_misalignment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "lr").mkdir()
            (root / "hr").mkdir()
            Image.new("RGB", (12, 12)).save(root / "lr/0001x4.png")
            Image.new("RGB", (48, 48)).save(root / "hr/0001.png")
            pairs = find_pairs(root / "lr", root / "hr", 4)
            self.assertEqual(len(pairs), 1)
            self.assertEqual(read_pair(pairs[0], 4)[1].shape, (48, 48, 3))
            with self.assertRaises(ValueError):
                check_disjoint(pairs, pairs)
            Image.new("RGB", (40, 48)).save(root / "hr/0001.png")
            with self.assertRaises(ValueError):
                read_pair(pairs[0], 4)
            Image.new("RGB", (48, 48)).save(root / "hr/0002.png")
            with self.assertRaises(ValueError):
                find_pairs(root / "lr", root / "hr", 4)

    def test_perfect_and_constant_metrics(self):
        a = np.full((32, 35, 3), 80, dtype=np.uint8)
        b = np.full_like(a, 120)
        perfect = psnr_ssim(a, a, 4)
        self.assertTrue(math.isinf(perfect["psnr_y"]))
        self.assertAlmostEqual(perfect["ssim_y"], 1, places=10)
        ya, yb = float(y_channel(a)[0, 0]), float(y_channel(b)[0, 0])
        result = psnr_ssim(a, b, 4)
        self.assertAlmostEqual(result["psnr_y"], 20 * math.log10(255 / abs(ya - yb)), places=10)
        expected = (2 * ya * yb + 2.55 ** 2) / (ya * ya + yb * yb + 2.55 ** 2)
        self.assertAlmostEqual(result["ssim_y"], expected, places=10)

    def test_ssim_against_direct_two_dimensional_windows(self):
        rng = np.random.default_rng(42)
        a = rng.integers(0, 256, (32, 35, 3), dtype=np.uint8)
        b = rng.integers(0, 256, (32, 35, 3), dtype=np.uint8)
        ya, yb = y_channel(a)[4:-4, 4:-4], y_channel(b)[4:-4, 4:-4]
        g = np.exp(-np.arange(-5, 6, dtype=float) ** 2 / (2 * 1.5 ** 2))
        window = np.outer(g, g)
        window /= window.sum()
        values = []
        for i in range(ya.shape[0] - 10):
            for j in range(ya.shape[1] - 10):
                x, y = ya[i:i+11, j:j+11], yb[i:i+11, j:j+11]
                mx, my = float((window * x).sum()), float((window * y).sum())
                vx, vy = (window * (x - mx) ** 2).sum(), (window * (y - my) ** 2).sum()
                cov = (window * (x - mx) * (y - my)).sum()
                values.append(((2*mx*my + 2.55**2) * (2*cov + 7.65**2)) /
                              ((mx**2 + my**2 + 2.55**2) * (vx + vy + 7.65**2)))
        self.assertAlmostEqual(psnr_ssim(a, b, 4)["ssim_y"], float(np.mean(values)), places=11)

    def test_border_crop_and_small_images(self):
        a = np.full((32, 32, 3), 100, dtype=np.uint8)
        b = a.copy()
        b[:4] = 0
        self.assertTrue(math.isinf(psnr_ssim(a, b, 4)["psnr_y"]))
        self.assertTrue(math.isfinite(psnr_ssim(a, b, 0)["psnr_y"]))
        with self.assertRaises(ValueError):
            psnr_ssim(a[:10, :10], b[:10, :10], 4)

    def test_zip_extraction_stays_in_data_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr("../outside.png", b"not an image")
            with self.assertRaises(ValueError):
                extract_pngs(archive, root / "data")
            self.assertFalse((root / "outside.png").exists())


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed: model/resume checks skipped")
class ModelAndResumeTests(unittest.TestCase):
    def test_forward_and_backward_both_scales_and_fusions(self):
        import torch
        from model import CompactMSAF
        torch.set_num_threads(1)
        for scale in (2, 4):
            for fusion in ("uniform", "learned"):
                model = CompactMSAF(scale=scale, features=8, blocks=1, fusion=fusion)
                x = torch.rand(1, 3, 11, 13)
                y = model(x)
                self.assertEqual(tuple(y.shape), (1, 3, 11 * scale, 13 * scale))
                self.assertTrue(torch.isfinite(y).all().item())
                y.square().mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all().item()
                                    for p in model.parameters()))

    def test_research_paths_targets_and_gradients(self):
        import torch
        from model import CompactMSAF
        from research import (local_branch_errors, preference_targets,
                              research_objective, scheduled_gate_weight)
        torch.set_num_threads(1)
        model = CompactMSAF(scale=4, features=8, blocks=2, fusion="learned")
        low = torch.rand(2, 3, 8, 9)
        high = torch.rand(2, 3, 32, 36)
        ordinary = model(low)
        main, candidates, logits = model.forward_research(low)
        torch.testing.assert_close(main, ordinary, rtol=0, atol=0)
        self.assertEqual(len(candidates), 3)
        self.assertEqual(tuple(logits.shape), (2, 1, 3, 8, 9))
        self.assertTrue(all(tuple(value.shape) == tuple(main.shape) for value in candidates))
        depth_one = model.forward_forced_paths(low, 1)
        for expected, actual in zip(candidates, depth_one):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        depth_two = model.forward_forced_paths(low, 2)
        self.assertTrue(all(tuple(value.shape) == tuple(main.shape) for value in depth_two))
        deep_main, deep_candidates, deep_logits = model.forward_research(low, 2)
        torch.testing.assert_close(deep_main, ordinary, rtol=0, atol=0)
        self.assertEqual(tuple(deep_logits.shape), (2, 2, 3, 8, 9))
        for expected, actual in zip(depth_two, deep_candidates):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            model.forward_forced_paths(low, 0)
        with self.assertRaises(ValueError):
            model.forward_forced_paths(low, 3)
        errors = local_branch_errors(candidates, high, 4)
        targets = preference_targets(errors)
        self.assertFalse(targets.requires_grad)
        self.assertTrue(torch.allclose(targets.sum(dim=1), torch.ones_like(targets[:, 0])))
        synthetic = torch.tensor([[[[.1]], [[.2]], [[.3]]]], requires_grad=True)
        preference = preference_targets(synthetic)
        self.assertGreater(float(preference[0, 0]), float(preference[0, 1]))
        self.assertGreater(float(preference[0, 1]), float(preference[0, 2]))
        self.assertEqual(scheduled_gate_weight(10, .05, 10, 10), 0)
        self.assertAlmostEqual(scheduled_gate_weight(15, .05, 10, 10), .025)
        self.assertAlmostEqual(scheduled_gate_weight(20, .05, 10, 10), .05)
        result = research_objective(
            main, candidates, logits, high, 4, "error_supervised",
            auxiliary_weight=.1, maximum_gate_weight=.05,
            gate_warmup=0, gate_ramp=0, temperature=1., update=1,
        )
        self.assertTrue(torch.isfinite(result.total).item())
        result.total.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all().item()
                            for p in model.parameters()))
        self.assertGreater(float(model.blocks[-1].gate.weight.grad.abs().sum()), 0)

    def test_resumed_cpu_training_matches_uninterrupted_training(self):
        import torch
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for split, name, seed in (("train", "0001", 1), ("valid", "0801", 2)):
                low = np.random.default_rng(seed).integers(0, 256, (16, 16, 3), dtype=np.uint8)
                high = np.repeat(np.repeat(low, 4, 0), 4, 1)
                hr = root / "data" / f"DIV2K_{split}_HR"
                lr = root / "data" / f"DIV2K_{split}_LR_bicubic/X4"
                hr.mkdir(parents=True)
                lr.mkdir(parents=True)
                Image.fromarray(low).save(lr / f"{name}x4.png")
                Image.fromarray(high).save(hr / f"{name}.png")
            base = [
                sys.executable, str(ROOT / "train.py"), "--data", str(root / "data"),
                "--device", "cpu", "--no-amp", "--features", "8", "--blocks", "1",
                "--steps", "4", "--patch", "8", "--micro-batch", "1",
                "--accumulate", "1", "--train-limit", "1", "--valid-limit", "1",
                "--eval-crop", "8", "--save-every", "1", "--validate-every", "2",
            ]
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            for experiment, experiment_args in (
                ("baseline", []),
                ("error", ["--experiment", "error_supervised", "--gate-warmup", "1",
                           "--gate-ramp", "1"]),
            ):
                resumed, full = root / f"{experiment}_resumed", root / f"{experiment}_full"
                for extra in (
                    ["--run", str(resumed), "--stop-after", "2"],
                    ["--run", str(resumed), "--resume"],
                    ["--run", str(full)],
                ):
                    result = subprocess.run(base + experiment_args + extra, env=env, text=True,
                                            capture_output=True, timeout=120)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                a = torch.load(resumed / "last.pt", map_location="cpu", weights_only=True)
                b = torch.load(full / "last.pt", map_location="cpu", weights_only=True)
                self.assertEqual(a["step"], 4)
                for key in a["model"]:
                    torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
                rows = [json.loads(line) for line in (resumed / "history.jsonl").read_text().splitlines()]
                self.assertEqual([r["step"] for r in rows if r["type"] == "train"], [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main(verbosity=2)

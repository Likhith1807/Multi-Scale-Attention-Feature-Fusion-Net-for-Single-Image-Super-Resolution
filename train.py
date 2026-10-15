"""Reproducible baseline and controlled scale-selection research trainer."""
import argparse
import json
from pathlib import Path
import platform
import time

from evaluate import evaluate_model, json_safe, save_json
from pairs import PatchDataset, check_disjoint, find_pairs, fingerprint, read_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    parser.add_argument("--run", default="runs/baseline_x4_pilot")
    parser.add_argument("--scale", type=int, choices=(2, 4), default=4)
    parser.add_argument("--features", type=int, default=48)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--fusion", choices=("learned", "uniform"), default="learned")
    parser.add_argument("--experiment", choices=("baseline", "auxiliary", "error_supervised"),
                        default="baseline")
    parser.add_argument("--aux-weight", type=float, default=0.1)
    parser.add_argument("--gate-weight", type=float, default=0.05,
                        help="Maximum error-supervision weight")
    parser.add_argument("--gate-warmup", type=int, default=1000,
                        help="Updates completed before gate supervision starts")
    parser.add_argument("--gate-ramp", type=int, default=1000,
                        help="Updates used to linearly reach --gate-weight; 0 = immediate")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--research-depth", type=int, default=1,
                        help="Number of final blocks used by forced research paths")
    parser.add_argument("--steps", type=int, default=1000, help="Total scheduled optimizer updates")
    parser.add_argument("--stop-after", type=int, default=0, help="Pause after this update without changing the schedule")
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--patch", type=int, default=48)
    parser.add_argument("--fixed-patch", action="store_true", help="Repeat one patch for the overfit sanity check")
    parser.add_argument("--train-limit", type=int, default=100, help="First N sorted image IDs; 0 = all")
    parser.add_argument("--valid-limit", type=int, default=10, help="First N sorted image IDs; 0 = all")
    parser.add_argument("--eval-crop", type=int, default=128, help="Center LR crop for development; 0 = full image")
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for key in ("steps", "micro_batch", "accumulate", "patch", "features", "blocks",
                "research_depth",
                "validate_every", "save_every"):
        if getattr(args, key) <= 0:
            parser.error(f"{key} must be positive")
    for key in ("train_limit", "valid_limit", "eval_crop", "workers", "seed", "stop_after",
                "gate_warmup", "gate_ramp"):
        if getattr(args, key) < 0:
            parser.error(f"{key} must be nonnegative")
    if (args.learning_rate <= 0 or args.stop_after > args.steps or
            args.aux_weight < 0 or args.gate_weight < 0 or args.temperature <= 0):
        parser.error("Check learning rate and stop-after.")
    if args.experiment != "baseline" and args.fusion != "learned":
        parser.error("Auxiliary and error-supervised experiments require --fusion learned.")
    if args.research_depth > args.blocks:
        parser.error("research-depth cannot exceed blocks")
    import torch
    from torch.utils.data import DataLoader
    from model import CompactMSAF
    from research import research_objective
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; run check_environment.py.")
    device = torch.device(args.device)
    use_amp = args.amp and device.type == "cuda"
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    root, run = Path(args.data), Path(args.run)
    if args.resume and not (run / "last.pt").is_file():
        parser.error("Resume requested but this run has no last.pt.")
    if not args.resume and run.exists() and any(run.iterdir()):
        parser.error("Run folder is not empty. Use --resume or choose a new --run.")
    training = find_pairs(root / f"DIV2K_train_LR_bicubic/X{args.scale}",
                          root / "DIV2K_train_HR", args.scale, args.train_limit)
    validation = find_pairs(root / f"DIV2K_valid_LR_bicubic/X{args.scale}",
                            root / "DIV2K_valid_HR", args.scale, args.valid_limit)
    check_disjoint(training, validation)
    # Inspect selected images before a long run; no silent target resampling.
    for pairs, patch in ((training, args.patch), (validation, args.eval_crop)):
        for pair in pairs:
            low, high = read_pair(pair, args.scale)
            if patch and min(low.shape[:2]) < patch:
                raise ValueError(f"{pair.name} is too small for the requested crop.")
            available = patch * args.scale if patch else min(high.shape[:2])
            if pairs is validation and available - 2 * args.scale < 11:
                raise ValueError("Validation crop is too small for the metric window.")
    print("Hashing selected pairs for a reproducible dataset manifest...", flush=True)
    manifests = {"train": fingerprint(training), "validation": fingerprint(validation)}
    train_hashes = {row["hr_sha256"] for row in manifests["train"]["images"]}
    if train_hashes & {row["hr_sha256"] for row in manifests["validation"]["images"]}:
        raise ValueError("A byte-identical HR file appears in both data splits.")
    model_config = {k: getattr(args, k) for k in ("scale", "features", "blocks", "fusion")}
    model = CompactMSAF(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(.9, .99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.learning_rate / 100
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    step, best = 0, float("-inf")
    config = vars(args).copy()
    # Data/run paths and worker count may change when moving to another machine.
    fixed = [k for k in config if k not in (
        "data", "run", "workers", "resume", "stop_after", "device"
    )]
    if args.resume:
        checkpoint = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
        old = checkpoint["config"]
        research_keys = {"experiment", "aux_weight", "gate_weight", "gate_warmup",
                         "gate_ramp", "temperature", "research_depth"}
        differences = [
            k for k in fixed
            if not (k in research_keys and k not in old and args.experiment == "baseline")
            and old.get(k) != config[k]
        ]
        if differences:
            raise ValueError(f"Resume configuration differs: {differences}")
        if checkpoint["amp_used"] != use_amp:
            raise ValueError("Resume requires the same AMP mode.")
        if any(checkpoint["manifest"][key]["sha256"] != manifests[key]["sha256"]
               for key in manifests):
            raise ValueError("Dataset contents differ from the checkpoint.")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        step, best = checkpoint["step"], checkpoint["best_psnr_y"]
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda" and checkpoint.get("cuda_rng") is not None:
            states = checkpoint["cuda_rng"]
            if len(states) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(states)
        # Discard log entries after the saved update if the previous process
        # stopped between checkpoints; those updates will be recomputed.
        history = run / "history.jsonl"
        if history.exists():
            retained = []
            for line in history.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("step", 0) <= step:
                    retained.append(json.dumps(row))
            temporary = history.with_suffix(".jsonl.tmp")
            temporary.write_text("\n".join(retained) + ("\n" if retained else ""), encoding="utf-8")
            temporary.replace(history)
    run.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        save_json(run / "config.json", config)
        save_json(run / "manifest.json", manifests)
    save_json(run / "environment_latest.json", {
        "python": platform.python_version(), "torch": str(torch.__version__),
        "cuda_build": torch.version.cuda, "amp": use_amp,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "parameters": sum(p.numel() for p in model.parameters()),
        "resume": args.resume,
    })
    effective_batch = args.micro_batch * args.accumulate
    dataset = PatchDataset(training, args.scale, args.patch, args.seed,
                           args.steps * effective_batch, args.fixed_patch)
    loader_options = {
        "dataset": dataset, "batch_size": args.micro_batch,
        "sampler": range(step * effective_batch, len(dataset)),
        "num_workers": args.workers, "pin_memory": device.type == "cuda",
        "generator": torch.Generator().manual_seed(args.seed),
    }
    if args.workers:
        loader_options.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(**loader_options)
    stream = iter(loader)
    stop = args.stop_after or args.steps
    if stop <= step:
        print(f"Already at update {step}; requested stopping update is {stop}.")
        return

    def checkpoint_to(name):
        payload = {
            "format_version": 2, "model_config": model_config, "config": config,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "amp_used": use_amp, "step": step, "best_psnr_y": best,
            "manifest": manifests, "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        }
        temporary = run / (name + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(run / name)

    def append_log(record):
        with (run / "history.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(json_safe(record), allow_nan=False) + "\n")

    session_start = step
    print(f"Experiment={args.experiment}; research_depth={args.research_depth}; "
          f"training {len(training)} images; "
          f"validating {len(validation)}; start update {step}.", flush=True)
    while step < stop:
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.
        running_total = 0.
        research_sums = {}
        # On an AMP overflow, retry these exact crops instead of losing a batch.
        batches = [next(stream) for _ in range(args.accumulate)]
        lr_used = optimizer.param_groups[0]["lr"]
        for attempt in range(10):
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.
            running_total = 0.
            research_sums = {}
            for low, high in batches:
                low, high = low.to(device, non_blocking=True), high.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    if args.experiment == "baseline":
                        prediction = model(low)
                    else:
                        prediction, candidates, gate_logits = model.forward_research(
                            low, args.research_depth
                        )
                if args.experiment == "baseline":
                    loss = torch.nn.functional.l1_loss(prediction.float(), high.float())
                    main_l1 = loss
                else:
                    objective = research_objective(
                        prediction, candidates, gate_logits, high, args.scale,
                        args.experiment, args.aux_weight, args.gate_weight,
                        args.gate_warmup, args.gate_ramp, args.temperature, step + 1,
                    )
                    loss, main_l1 = objective.total, objective.main_l1
                    values = {
                        "auxiliary_l1": float(objective.auxiliary_l1.detach().item()),
                        "gate_cross_entropy": float(objective.gate_cross_entropy.detach().item()),
                        "gate_weight": objective.gate_weight,
                        **objective.diagnostics,
                    }
                    for key, value in values.items():
                        if isinstance(value, list):
                            target = research_sums.setdefault(key, [0.] * len(value))
                            for index, item in enumerate(value):
                                target[index] += item / args.accumulate
                        else:
                            research_sums[key] = research_sums.get(key, 0.) + value / args.accumulate
                if not torch.isfinite(loss).item():
                    raise RuntimeError("Non-finite loss. Inspect data or retry a new run with --no-amp.")
                running_loss += float(main_l1.detach().item()) / args.accumulate
                running_total += float(loss.detach().item()) / args.accumulate
                scaler.scale(loss / args.accumulate).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
            if not use_amp and not torch.isfinite(norm).item():
                raise RuntimeError("Non-finite gradients.")
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                break
        else:
            raise RuntimeError("Repeated AMP overflow. Inspect data or use --no-amp in a new run.")
        scheduler.step()
        step += 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        record = {"type": "train", "step": step, "l1": running_loss,
                  "total_loss": running_total, "learning_rate": lr_used,
                  "seconds": time.perf_counter() - start, **research_sums}
        append_log(record)
        if step % 25 == 0 or step == session_start + 1:
            extra = ""
            if args.experiment != "baseline":
                extra = (f"  aux={record['auxiliary_l1']:.6f}"
                         f"  gate_lambda={record['gate_weight']:.4f}")
            print(f"update {step}/{args.steps}  L1={running_loss:.6f}"
                  f"  total={running_total:.6f}{extra}  {record['seconds']:.3f}s",
                  flush=True)
        if step % args.validate_every == 0 or step == args.steps:
            result = evaluate_model(model, validation, args.scale, device, args.eval_crop,
                                    run / "validation" / f"step_{step:06d}")
            score = result["mean"]["psnr_y"]
            append_log({"type": "validation", "step": step, **result["mean"],
                        "region": result["protocol"]["region"]})
            print(f"Validation Y-PSNR={score:.4f}, Y-SSIM={result['mean']['ssim_y']:.5f}", flush=True)
            if score > best:
                best = score
                checkpoint_to("best.pt")
            checkpoint_to("last.pt")
        elif step % args.save_every == 0 or step == stop:
            checkpoint_to("last.pt")
    print(f"Saved update {step} to {run / 'last.pt'}.", flush=True)


if __name__ == "__main__":
    main()

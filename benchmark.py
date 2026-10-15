"""Time synthetic optimizer updates for sizing the first experiment.

This is a compute benchmark, not dataset training or an image-quality test.
"""
import argparse
import json
from pathlib import Path
import platform
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--scale", type=int, choices=(2, 4), default=4)
    parser.add_argument("--features", type=int, default=48)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--fusion", choices=("learned", "uniform"), default="learned")
    parser.add_argument("--experiment", choices=("baseline", "auxiliary", "error_supervised"),
                        default="baseline")
    parser.add_argument("--research-depth", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--patch", type=int, default=48)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", default="runs/benchmark.json")
    args = parser.parse_args()
    for name in ("features", "blocks", "research_depth", "micro_batch", "accumulate",
                 "patch", "warmup", "steps"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    try:
        import torch
        from model import CompactMSAF
        from research import research_objective
    except ImportError:
        print("Install PyTorch first; follow README.md.", file=sys.stderr)
        return 1
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable. Run check_environment.py first.", file=sys.stderr)
        return 1
    if args.experiment != "baseline" and args.fusion != "learned":
        parser.error("Research experiments require --fusion learned.")
    if args.research_depth > args.blocks or args.temperature <= 0:
        parser.error("research-depth cannot exceed blocks; temperature must be positive")

    device = torch.device(args.device)
    use_amp = args.amp and device.type == "cuda"
    torch.manual_seed(2026)
    model = CompactMSAF(args.scale, args.features, args.blocks, args.fusion).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    lr = torch.rand(args.micro_batch, 3, args.patch, args.patch, device=device)
    hr = torch.rand(
        args.micro_batch, 3, args.patch * args.scale, args.patch * args.scale,
        device=device,
    )

    def update():
        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.accumulate):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                if args.experiment == "baseline":
                    prediction = model(lr)
                else:
                    prediction, candidates, gate_logits = model.forward_research(
                        lr, args.research_depth
                    )
            if args.experiment == "baseline":
                loss = torch.nn.functional.l1_loss(prediction.float(), hr)
            else:
                loss = research_objective(
                    prediction, candidates, gate_logits, hr, args.scale,
                    args.experiment, auxiliary_weight=.1, maximum_gate_weight=.05,
                    gate_warmup=0, gate_ramp=0, temperature=args.temperature, update=1,
                ).total
            scaler.scale(loss / args.accumulate).backward()
        scaler.step(optimizer)
        scaler.update()
        return loss

    for index in range(args.warmup):
        loss = update()
        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite warmup loss; try --no-amp.")
        if index == 0 and not any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
            and torch.count_nonzero(parameter.grad).item() > 0
            for parameter in model.parameters()
        ):
            raise RuntimeError("No finite, nonzero gradients were observed.")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    scale_before = scaler.get_scale()
    start = time.perf_counter()
    for _ in range(args.steps):
        loss = update()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    if not torch.isfinite(loss).item():
        raise RuntimeError("Non-finite final loss; try --no-amp.")
    if use_amp and scaler.get_scale() < scale_before:
        raise RuntimeError(
            "AMP reduced its scale during timing, so some updates may have been "
            "skipped. Rerun with --no-amp for a valid timing."
        )
    seconds_per_update = elapsed / args.steps
    report = {
        "purpose": "synthetic optimizer-update timing; no image-quality results",
        "config": vars(args),
        "python": platform.python_version(),
        "pytorch": str(torch.__version__),
        "pytorch_cuda_build": torch.version.cuda,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "effective_batch": args.micro_batch * args.accumulate,
        "amp_used": use_amp,
        "seconds_per_optimizer_update": seconds_per_update,
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / 1024 ** 3
            if device.type == "cuda" else None
        ),
        "peak_reserved_gib": (
            torch.cuda.max_memory_reserved(device) / 1024 ** 3
            if device.type == "cuda" else None
        ),
        "compute_only_hours": {
            str(steps): steps * seconds_per_update / 3600
            for steps in (5000, 25000, 100000)
        },
        "limitations": [
            "Random tensors stay on the device; image decoding and data loading are excluded.",
            "Validation, checkpoints, augmentation, image decoding, and data loading are excluded.",
            "Research variants include their training-only candidate reconstructions and losses.",
            "Long laptop runs may slow because of thermal or power limits.",
            "The CPU option is a functional check and cannot predict laptop GPU speed.",
        ],
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nReport saved to {destination}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            print(
                "GPU memory ran out. Retry --micro-batch 2 --accumulate 8; "
                "if needed also use --patch 32.",
                file=sys.stderr,
            )
            sys.exit(2)
        raise

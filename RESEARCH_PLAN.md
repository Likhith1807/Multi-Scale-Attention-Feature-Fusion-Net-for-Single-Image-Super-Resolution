# Research plan — 27 September 2026

Technical specification: docs/ARCHITECTURE_AND_METHOD.md.
Current implementation: baseline, matched auxiliary-only control,
error-supervised routing, schedules, diagnostics, run comparison, a completed
candidate-depth diagnostic, and a frozen depth-6 mechanism screen.
Synthetic blur/noise training and external benchmark ingestion remain pending.

| Stage | Work | Gate for proceeding |
|---|---|---|
| New laptop | RTX 5060 setup, self-tests, benchmark | CUDA and all checks pass |
| Data | Offline extraction and all-pair verification | 800/100 valid source pairs |
| Baseline | Fixed-patch check, real-data pilot, convergence measurements | Finite training and correct images/metrics |
| Prior art | Read closest supervision/routing/fusion work | Defensible specific distinction |
| Mechanism pilot V1 | Shared-decoder alternatives and gate supervision | Completed: rejected; candidate errors collapsed and quality did not improve |
| Mechanism diagnosis | Force each branch through 1, 3 and 6 final blocks | Completed: depth 6 selected; temperature 0.05 calibrated |
| Mechanism pilot V2 | Matched depth-6 auxiliary and error-supervised runs | First pass RTX cost gate, then beat both controls without collapse |
| Robustness | Implement a frozen synthetic blur/noise protocol | Reproducible effect under predefined shifts |
| Full study | Recent baselines, central three-seed repeats, ablations, latency | Consistent measured evidence |
| Paper | Method, results, limitations, exact configs and code | Supervisor review and scope-matched submission |

Plan roughly six to eight weeks of sustained work from a verified new-laptop
baseline, depending on experiments and revisions. This is a planning estimate,
not a promised acceptance/publication date. November can be a submission goal;
conference deadlines and review timelines need separate verification.

Keep data/steps/tuning budgets matched. Additional training computation must be
reported. Do not treat ordinary multi-scale attention, a small parameter count,
or a new model name as established novelty. Do not report synthetic random-tensor
benchmark timing as real-data training time or SR quality.

Uploaded G16 is the main architectural reference; G15 is low-light object
detection and only contextual reading, not a direct SR comparison.

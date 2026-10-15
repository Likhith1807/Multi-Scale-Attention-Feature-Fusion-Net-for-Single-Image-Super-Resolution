# Security Policy

## Supported Versions

Security fixes and dependency patches are applied to the latest release on the `main` branch.

| Version | Supported |
|---|---|
| `main` (v1.0.x) | :white_check_mark: |
| < 1.0.0 | :x: |

---

## Reporting a Vulnerability

If you discover a potential security vulnerability within this project (such as unsafe deserialization in checkpoint loading, dependency vulnerabilities, or command injection risks in scripts):

1. **Do not disclose the issue publicly** on GitHub Issues or discussions.
2. Please report the issue privately to the repository maintainer via GitHub Security Advisories or by contacting the maintainer:
   - Profile: [https://github.com/Likhith1807](https://github.com/Likhith1807)
   - Email: `Likhith1807@users.noreply.github.com`
3. Include:
   - A detailed description of the vulnerability.
   - Minimal reproducible proof-of-concept (PoC) code or commands.
   - Impact assessment and potential mitigation.

We will acknowledge receipt of your vulnerability report within 48 hours and work with you on a coordinated disclosure timeline.

---

## Model Weights & Checkpoint Security Notice

In PyTorch, loading arbitrary `.pt` or `.pth` checkpoint files using `torch.load()` without caution can execute arbitrary code if the file contains pickled Python objects.
- In this codebase, checkpoints are saved and loaded containing standard PyTorch `state_dict` structures.
- Always inspect the source of untrusted third-party checkpoint files before loading them into your environment (`weights_only=True` is strongly recommended for untrusted checkpoints).

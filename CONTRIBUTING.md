# Contributing to MSAF-Net

Thank you for your interest in contributing to **Multi-Scale Attention Feature Fusion Network for Single-Image Super-Resolution (MSAF-Net)**! We welcome contributions from computer vision researchers, deep learning engineers, and open-source practitioners.

---

## Code of Conduct

All contributors and maintainers are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/<your-username>/Multi-Scale-Attention-Feature-Fusion-Net-for-Single-Image-Super-Resolution.git
   cd Multi-Scale-Attention-Feature-Fusion-Net-for-Single-Image-Super-Resolution
   ```

2. **Environment Setup**
   Create an isolated virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   # Install torch for your CUDA environment or CPU:
   pip install torch --extra-index-url https://download.pytorch.org/whl/cu121
   ```

3. **Verify the Installation**
   Run the full unit test suite:
   ```bash
   python self_test.py
   ```
   All 9 tests should pass without skips or errors.

---

## Contribution Guidelines

### 1. Architectural Integrity
- The baseline $4\times$ model (`CompactMSAF`) has an established parameter count of **586,077 parameters**.
- Do not modify parameter names, tensor dimensions, or default arguments in `model.py` without clearly justifying the architectural rationale and impact on checkpoint compatibility.
- Ensure that the default inference graph remains deterministic, fast, and does not require auxiliary training-only heads.

### 2. Testing & Quality Standards
- Any new features, loss formulations, or model blocks must be accompanied by tests in `self_test.py` or new test files covering:
  - Shape consistency across batch sizes and resolutions.
  - Forward and backward gradient propagation.
  - State dict serialization and checkpoint resume fidelity.
- Run `python self_test.py` before submitting any pull request.

### 3. Coding Style
- Follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines.
- Keep docstrings and type annotations clear and descriptive.
- Avoid external dependencies beyond PyTorch, NumPy, Pillow, and SciPy unless strictly necessary.

### 4. Commits and Pull Requests
- Use clear, descriptive commit messages:
  - `feat(model): add multi-head attention variant`
  - `fix(metrics): correct YCbCr matrix precision for BT.601`
  - `docs: update experimental throughput on RTX 4090`
- Submit pull requests against the `main` branch. Ensure the CI checks pass.

---

## Reporting Issues

If you encounter bugs, unexpected divergences, or numerical instability:
- Open a GitHub Issue using our [Bug Report Template](.github/ISSUE_TEMPLATE/bug_report.yml).
- Include your operating system, GPU architecture, PyTorch version, and full traceback.

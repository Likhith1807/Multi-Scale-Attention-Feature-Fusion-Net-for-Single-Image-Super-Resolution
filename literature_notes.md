# Week 1 literature worksheet

Fill the open questions from papers and official code. A future-work paragraph
alone does not establish novelty.

| Paper | Known task/mechanism | Role | Open questions |
| --- | --- | --- | --- |
| G16: MHARN, IEEE Access 2026; DOI 10.1109/ACCESS.2026.3667149 | SISR; dilated PCG branches, hybrid attention fusion, dynamic kernels; reports 18.3M parameters. | Main reference. | Author code, channel widths, dense growth rate, crop size, attention windows, exact evaluation protocol? |
| G15: MSF-DET, IEEE Sensors Journal 2026; DOI 10.1109/JSEN.2026.3670619 | Low-light object detection with multiscale and FFT processing. | Contextual reading. | Could a mechanism motivate a separately justified SR experiment? |
| Han et al., Sensors 2023; DOI 10.3390/s23239458 | Multi-scale recursive attention feature fusion for SR. | Closely related architecture. | Exact fusion and overlap with our planned idea? |
| DASR, CVPR 2021; arXiv 2104.00416 | Degradation-representation learning for blind SR. | Degradation-aware reference. | Which degradation assumptions and supervision are relevant? |
| RFDN: https://github.com/njulj/RFDN | Lightweight SR with official code. | Engineering comparison. | Runnable training/evaluation, parameter budget, license? |
| Relevant 2025/2026 efficient SR paper | To select. | Recent comparison. | Is the exact error-supervised fusion mechanism already present? |

Proposed question: does branch reconstruction-error supervision improve scale
fusion under held-out degradations at small inference cost?

Search error-aware/uncertainty-aware SR, mixture-of-experts gating, branch
selection, auxiliary reconstruction supervision, and degradation-aware fusion.
For the closest paper record:

- Exact fusion signal and supervision:
- Spatial granularity:
- Training/test degradations:
- Inference overhead:
- Public implementation:
- Whether HR information is required at inference:
- Remaining difference from the proposed method:

## 25 September update
Read docs/ARCHITECTURE_AND_METHOD.md for the updated shared-decoder candidate
proposal and primary-source links. Add the following close work to the audit:
EMDN (PLOS One, 2025); Spectral Bayesian Uncertainty (CVPR 2023); Complexity
Experts (CVPR 2025). Exact novelty remains unresolved. Record supervision
signals and training/inference differences, not just architecture names.

# Architecture and methodology specification
Revision: 27 September 2026. Status: baseline and controlled research paths implemented.

## Purpose and research question
Reconstruct an RGB image at four times the input width and height from one LR
image. Missing detail is ambiguous; reconstructed textures are estimates.
Investigate whether supervising spatial branch preferences with training-time
reconstruction errors improves SR under specified blur/noise shifts while
preserving the deployed architecture.

Working research title: Error-Supervised Scale Selection for Lightweight
Single-Image Super-Resolution under Degradation Shift.
Novelty and effectiveness are not established. No conference acceptance is guaranteed.

## Implemented baseline
CompactMSAF in model.py has 586,077 parameters in the default learned-fusion x4
configuration, confirmed again by the user's RTX 5060 benchmark.

| Stage | Operation | Output for a 48x48 LR crop (C x H x W) |
|---|---|---|
| Input | RGB in [0,1] | 3 x 48 x 48 |
| Stem | 3x3 convolution, 3 to 48 channels | 48 x 48 x 48 |
| Trunk | Six MultiScaleBlocks | 48 x 48 x 48 |
| Body | 3x3 convolution + stem skip | 48 x 48 x 48 |
| Upsampler 1 | 3x3 conv 48 to 192, PixelShuffle(2), LeakyReLU | 48 x 96 x 96 |
| Upsampler 2 | Same operations | 48 x 192 x 192 |
| RGB head | 3x3 conv 48 to 3 | 3 x 192 x 192 |
| Reconstruction | Add PyTorch bicubic x4 input | 3 x 192 x 192 |

Within every block:
1. Three independent 3x3 convolutions, dilation 1, 2 and 3, with LeakyReLU.
2. Gate: a 1x1 convolution from 48 channels to three logits at each LR position.
3. Softmax over branches (computed in FP32); weighted feature sum.
4. Channel attention: global average pooling, 48-to-12 1x1 conv, ReLU,
   12-to-48 1x1 conv, sigmoid; multiply the fused features.
5. 1x1 projection and residual addition with scale 0.2.

Equations:
B_k = LeakyReLU(Conv_dk(F)), dk in {1,2,3}.
w_k(p) = softmax_k(g(F)(p)).
M(p) = sum_k w_k(p) B_k(p).
F_out = F + 0.2 P(a(M) * M).

Dilation gives 3x3, 5x5 and 7x7 spatial spans for these individual convolutions;
these are sparse sampling patterns, not dense large kernels. All feature grids
retain the same resolution. The names 'edge expert' or 'texture expert' are not
assigned in advance; actual specialization requires evidence.
The uniform option removes the gate and averages branches, changing parameters slightly.
There is no transformer, GAN, diffusion model, uncertainty estimator or dynamic
kernel generator in this baseline. It is not an exact reproduction of MHARN.

## Implemented research extension
Initially supervise only the final block. In addition to the normal mixed
prediction, make three forced branch-choice predictions during training.
For candidate k replace only the final block's mixed features M with B_k;
apply that block's channel attention, projection/residual, body convolution,
shallow skip and the SAME existing decoder/RGB head/bicubic skip.
Earlier five blocks are shared. The three decoder evaluations share parameters.
This is a concrete alternative to adding separate auxiliary reconstruction heads.

The branch candidates measure each branch within the downstream network.
They do not establish which feature mixture is globally optimal, because
channel attention and decoding are nonlinear.

Compute e_k by RGB mean absolute error against training HR, average-pool each
4x4 HR cell onto the LR grid, then apply the same 3x3 local averaging to all
branches. Fix border treatment consistently (e.g. exclude padding from means).
Let e_mean = mean_k(e_k). Form detached targets:
q_k = softmax_k(-stopgrad(e_k) / (tau * stopgrad(e_mean) + epsilon)).
Use epsilon=1e-6 for unit-range images as an initial numerical safeguard.
The existing LR-only gate predicts w. HR targets are never inputs at inference.

Losses:
L_SR = mean(abs(SR - HR)).
L_aux = mean_k mean(abs(candidate_k - HR)).
L_gate = mean_p [-sum_k q_k(p) log(w_k(p))].
L_total = L_SR + lambda_aux L_aux + lambda_gate(t) L_gate.
The implementation computes log weights with stable FP32 `log_softmax`.
Only q is detached; reconstruction losses train candidates/shared layers and
gate loss trains the gate and its upstream feature path.

Initial 3,000-update development settings (not tuned or validated): auxiliary
weight 0.1, maximum gate weight 0.05, tau 1.0, no gate loss through update 500,
then a linear ramp ending at update 1,000. The general CLI defaults remain a
1,000-update warm-up and 1,000-update ramp for longer schedules.

At inference, remove candidate evaluations and target/loss calculation.
All three feature branches still run, as they already do in the baseline.
This is soft fusion, not sparse execution or a promised reduction in FLOPs.
With the unchanged decoder/gate, parameter count and inference graph remain
unchanged. Measure added training time and memory; do not conceal that cost.
Gate weights are relative preferences, not calibrated correctness probabilities.

## Data and evaluation protocols
Implemented: official DIV2K bicubic x4 training and validation pairs. Source IDs
are separated before patch extraction. Default pilot uses only 100 training
and 10 validation images; full-data limits are 0 but choose a new run and an
explicit training schedule after convergence measurements.
Training: 48x48 LR / 192x192 HR crops, aligned flips/transposes, Adam
(1e-4, betas .9/.99), cosine decay, AMP, microbatch 4 with accumulation 4.

The proposed robustness study still requires new degradation-generation code. Define
LR = clip(D4(HR * K) + n, 0, 1), including exact kernel support, reflection
padding, antialiasing/downsampling routine, noise clipping and operation order.
Starting proposal: 21x21 Gaussian blur, training sigma 0.2-1.5 HR pixels,
Gaussian noise std 0-10/255 in LR RGB. Predeclare independent evaluation
samples in this range and held-out stronger blur (1.5-2.5), stronger noise
(10-20/255), and anisotropic kernels. Freeze exact anisotropic ranges and seeds
before final testing. Do not describe this as real-camera validation.
Both baseline and proposal train on identical degradation distributions.
Bicubic-only weights under blur/noise are a transfer diagnostic, not a matched
blind-SR comparison. Keep separate checkpoints for the two studies.

Development data selects settings; final benchmark data stays untouched until
protocols are locked. Candidate final sets: Set5, Set14, BSD100, Urban100 and
Manga109, subject to access terms and compatible official LR generation.
Implemented metrics: uint8-quantized output, BT.601 Y, HR border crop equal to
scale, per-image PSNR and 11x11 Gaussian SSIM. Record conventions and avoid
mixing protocols or PyTorch versus MATLAB bicubic comparator values.

## Required controls and diagnostics
| Variant | Role |
|---|---|
| Uniform fusion | Cost/benefit of learning weights |
| Ordinary learned fusion | Implemented baseline |
| Learned fusion + identical candidate reconstruction losses | Deep-supervision control |
| Same + error-derived gate supervision | Proposed intervention |
| Same + spatially shuffled targets | Test target information |
| Second scale or compatible backbone | Evidence for broader applicability |

The auxiliary-only versus error-supervised contrast is central. Match data,
initialization seeds, steps and tuning budgets; additionally report a training
compute comparison because ordinary baseline updates are cheaper. Run at least
three seeds for the central contrast. Compare per-source-image differences;
bootstrap source images, not correlated patches, if reporting intervals.

Report PSNR/SSIM, full-image examples and failures, parameter count, measured
latency at identical dimensions/precision, GPU memory and training cost.
Inspect branch utilization, candidate diversity and error/weight association.
Diagnostic branch regret = mean_p(sum_k w_k e_k - min_k e_k); lower is better
for individual-branch preferences. It is not a bound on fused-output error.
Naive tiled inference is not equivalent because channel attention globally pools.

## Related work and unresolved novelty
- Uploaded G16: MHARN (2026), DOI 10.1109/ACCESS.2026.3667149. Already includes
  parallel dilated processing, adaptive fusion and attention. Do not present
  our baseline components as new.
- EMDN (2025): https://doi.org/10.1371/journal.pone.0330615. Uses multi-scale
  processing and feature-difference error-driven fusion. The candidate
  distinction is HR-derived branch-preference supervision, not the word 'error'.
- Spectral Bayesian Uncertainty for Image Super-Resolution (CVPR 2023):
  https://openaccess.thecvf.com/content/CVPR2023/html/Liu_Spectral_Bayesian_Uncertainty_for_Image_Super-Resolution_CVPR_2023_paper.html
- Complexity Experts are Task-Discriminative Learners for Any Image Restoration
  (CVPR 2025): https://openaccess.thecvf.com/content/CVPR2025/html/Zamfir_Complexity_Experts_are_Task-Discriminative_Learners_for_Any_Image_Restoration_CVPR_2025_paper.html
- DASR: https://arxiv.org/abs/2104.00416
- RFDN official implementation: https://github.com/njulj/RFDN
- NTIRE 2026 efficient SR report: https://arxiv.org/abs/2604.03198

The audit must still check equivalent routing supervision, privileged training
information, expert selection and shared-decoder auxiliary training. No 'first'
claim is justified. If equivalent prior art or no reproducible benefit emerges,
revise the question before committing to full experiments.

## Version 1 outcome and candidate-depth diagnostic
The first matched 3,000-update development screen completed on 27 September
2026. Baseline reached 28.986740 dB, auxiliary-only 28.979654 dB and
error-supervised 28.982423 dB on ten fixed center crops. Error supervision did
not beat baseline. During the final 500 updates its target entropy was 1.098586,
nearly the three-branch maximum `ln(3)=1.098612`, and gate-target agreement was
near chance. Therefore Version 1 is retained as a negative ablation rather than
claimed as an improvement.

The next implemented diagnostic forces one dilation branch through the last
1, 3 or 6 blocks of the existing baseline checkpoint. It measures decoded RGB
and finite-difference error diversity, target entropy, preference margins,
oracle local improvement and winner spatial agreement. It performs no training,
adds no parameters and leaves normal inference unchanged. Its purpose is to
decide whether deeper forced paths can support an informative Version 2 target;
the diagnostic itself is not evidence of better super-resolution.

## Version 2 frozen mechanism screen
The development diagnostic selected depth 6: candidate k forces dilation
branch k through all six blocks, while the normal reconstruction follows the
ordinary learned mixtures. The candidate-derived soft preference map is
detached and supervises each of the six deployed gate logits during training.
Their cross-entropies are averaged, so gate-loss scale does not grow with
depth. The auxiliary control uses the identical candidate paths without gate
cross-entropy.

Temperature 0.05 was frozen after a no-training sweep because its normalized
RGB target entropy was 0.9204. Normal inference calls neither forced paths nor
target generation; parameters and inference topology remain unchanged. These
diagnostics justify a controlled screen only and are not reconstruction-quality
evidence.

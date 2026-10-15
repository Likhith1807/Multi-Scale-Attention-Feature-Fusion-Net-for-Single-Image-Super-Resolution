"""Classical SR metrics: uint8 RGB -> BT.601 limited-range Y; scale border crop.

SSIM uses an 11x11 Gaussian window (sigma 1.5), population moments, valid
windows, and data range 255. Inputs must already be quantized RGB images.
"""
import math

import numpy as np
from scipy.ndimage import correlate1d


def y_channel(rgb):
    return 16 + np.asarray(rgb, dtype=np.float64) @ np.array(
        [65.481, 128.553, 24.966], dtype=np.float64
    ) / 255


def psnr_ssim(prediction, reference, border=4):
    prediction, reference = np.asarray(prediction), np.asarray(reference)
    if prediction.shape != reference.shape or prediction.ndim != 3:
        raise ValueError("Expected equally sized HWC RGB arrays.")
    if prediction.shape[2] != 3 or prediction.dtype != np.uint8 or reference.dtype != np.uint8:
        raise ValueError("Metrics require RGB uint8 arrays.")
    if border < 0:
        raise ValueError("border must be nonnegative")
    a, b = y_channel(prediction), y_channel(reference)
    if border:
        a, b = a[border:-border, border:-border], b[border:-border, border:-border]
    if min(a.shape) < 11:
        raise ValueError("Image is too small for border cropping and 11x11 SSIM.")
    mse = float(np.mean((a - b) ** 2))
    psnr = math.inf if mse == 0 else 10 * math.log10(255 ** 2 / mse)
    coordinates = np.arange(-5, 6, dtype=np.float64)
    kernel = np.exp(-(coordinates ** 2) / (2 * 1.5 ** 2))
    kernel /= kernel.sum()

    def blur(image):
        result = correlate1d(image, kernel, axis=0, mode="reflect")
        return correlate1d(result, kernel, axis=1, mode="reflect")[5:-5, 5:-5]

    ma, mb = blur(a), blur(b)
    va, vb, covariance = blur(a * a) - ma * ma, blur(b * b) - mb * mb, blur(a * b) - ma * mb
    c1, c2 = (.01 * 255) ** 2, (.03 * 255) ** 2
    ssim_map = ((2 * ma * mb + c1) * (2 * covariance + c2)) / (
        (ma * ma + mb * mb + c1) * (va + vb + c2)
    )
    return {"psnr_y": psnr, "ssim_y": float(ssim_map.mean())}

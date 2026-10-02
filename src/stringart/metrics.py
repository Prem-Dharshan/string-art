"""Quality metrics: PSNR and SSIM on the direct render and after Gaussian blur (viewing distance)."""

import cv2
import numpy as np
from skimage.metrics import structural_similarity

DEFAULT_SIGMAS = (0, 1, 2, 4)


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    return img if sigma == 0 else cv2.GaussianBlur(img, (0, 0), sigma)


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = (a - b) if mask is None else (a - b)[mask]
    mse = float(np.mean(diff**2))
    return float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))


def ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    _, smap = structural_similarity(a, b, data_range=1.0, full=True)
    return float(smap.mean() if mask is None else smap[mask].mean())


def evaluate(target, render, mask=None, sigmas=DEFAULT_SIGMAS, roi=None) -> dict[str, float]:
    """Metrics keyed like `ssim_s2` (SSIM after blurring both images with sigma = 2 px).
    With a `roi` mask (e.g. the face), also `ssim_roi_s2` / `psnr_roi_s2` restricted to it."""
    out = {}
    use_roi = roi is not None and roi.any()
    for s in sigmas:
        t, r = _blur(target, s), _blur(render, s)
        out[f"psnr_s{s}"] = round(psnr(t, r, mask), 4)
        out[f"ssim_s{s}"] = round(ssim(t, r, mask), 4)
        if use_roi:
            out[f"psnr_roi_s{s}"] = round(psnr(t, r, roi), 4)
            out[f"ssim_roi_s{s}"] = round(ssim(t, r, roi), 4)
    return out

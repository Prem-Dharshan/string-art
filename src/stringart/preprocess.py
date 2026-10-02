"""Image loading and the baseline preprocessing chain (grayscale -> CLAHE -> Gaussian)."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SAMPLES = ("astronaut", "camera", "coffee", "chelsea")


def load_image(src: str) -> np.ndarray:
    """Load a BGR uint8 image from a path, or `sample:<name>` from scikit-image's test data."""
    if src.startswith("sample:"):
        from skimage import data

        name = src.split(":", 1)[1]
        if name not in SAMPLES:
            raise ValueError(f"unknown sample {name!r}; choose from {SAMPLES}")
        img = getattr(data, name)()
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2BGR)
    path = Path(src)
    if not path.is_file():
        raise FileNotFoundError(path)
    # imdecode handles non-ASCII Windows paths, which cv2.imread does not.
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not decode image {path}")
    return img


def center_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(h, w)
    y, x = (h - s) // 2, (w - s) // 2
    return img[y : y + s, x : x + s]


def frame_mask(frame: str, size: int) -> np.ndarray:
    """Boolean mask of pixels inside the frame."""
    if frame == "rect":
        return np.ones((size, size), dtype=bool)
    yy, xx = np.mgrid[:size, :size]
    c = (size - 1) / 2
    return (xx - c) ** 2 + (yy - c) ** 2 <= c**2


@dataclass
class PreprocessConfig:
    size: int = 600
    frame: str = "circle"
    clahe_clip: float = 2.0  # 0 disables CLAHE
    clahe_tiles: int = 8
    blur_sigma: float = 1.0  # 0 disables smoothing


def preprocess(img: np.ndarray, cfg: PreprocessConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return (target, mask): grayscale float in [0, 1] (1 = white), white outside the frame."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.resize(center_square(gray), (cfg.size, cfg.size), interpolation=cv2.INTER_AREA)
    if cfg.clahe_clip > 0:
        clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(cfg.clahe_tiles,) * 2)
        gray = clahe.apply(gray)
    target = gray.astype(np.float64) / 255.0
    if cfg.blur_sigma > 0:
        target = cv2.GaussianBlur(target, (0, 0), cfg.blur_sigma)
    mask = frame_mask(cfg.frame, cfg.size)
    target[~mask] = 1.0
    return target, mask

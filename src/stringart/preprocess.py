"""Image loading and preprocessing (PLAN.md I4).

    load -> crop (face-centred via YuNet, else centre) -> resize -> grayscale
         -> level stretch (if exposure is poor) -> bilateral smoothing -> CLAHE
         -> optional GrabCut background fade (when a face is found) -> white outside frame

`smooth="gaussian"` with `stretch="off", crop="center"` reproduces the M1/M2 chain
(grayscale -> CLAHE -> Gaussian), available as `PreprocessConfig.legacy()`.
"""

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .face import Face, detect_faces

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
    crop: str = "face"  # "face": centre on the largest detected face (fallback centre); "center"
    face_zoom: float = 1.8  # crop side = face height * zoom (tight portrait)
    face_shift: float = 0.15  # move crop centre down by this fraction of face height
    # 1-99 percentile level stretch. "auto" only stretches badly exposed photos: on
    # well-exposed ones stretching measurably hurt (M3 ablation). Poor exposure = narrow range,
    # or no black point (washed out), or no white point (underexposed). Thresholds are 0..255;
    # every natural photo in the M6 set passes all three, every synthetic fault fails one.
    stretch: str = "auto"
    stretch_below: float = 115.0  # range p99 - p1 under this
    black_point_above: float = 64.0  # p1 over this: no blacks
    white_point_below: float = 128.0  # p99 under this: no whites
    smooth: str = "bilateral"  # "bilateral" (edge-preserving), "gaussian", "none"
    blur_sigma: float = 1.0  # gaussian sigma
    bilateral_sigma_color: float = 25.0
    clahe_clip: float = 2.0  # 0 disables CLAHE
    clahe_tiles: int = 8
    background: str = "none"  # "fade": lighten GrabCut background (faces only); "none"
    background_keep: float = 0.35  # fraction of background darkness kept when fading

    @classmethod
    def legacy(cls, **kw) -> "PreprocessConfig":
        """The M1/M2 chain: centre crop, grayscale -> CLAHE -> Gaussian."""
        return cls(crop="center", stretch="off", smooth="gaussian", **kw)


@dataclass
class Prepared:
    target: np.ndarray  # grayscale [0, 1], 1 = white; white outside the frame
    mask: np.ndarray  # inside-frame mask
    plain: np.ndarray  # plain grayscale of the same crop (fidelity reference for metrics)
    canvas_bgr: np.ndarray  # the cropped, resized colour image
    faces: list[Face] = field(default_factory=list)  # in canvas coordinates
    crop: tuple[int, int, int] = (0, 0, 0)  # x0, y0, side in the source image
    fg: np.ndarray | None = None  # soft foreground mask when background fading is on


def _crop_box(shape, faces: list[Face], cfg: PreprocessConfig) -> tuple[int, int, int]:
    h, w = shape[:2]
    max_side = min(h, w)
    if cfg.crop == "face" and faces:
        f = faces[0]
        side = int(round(min(max_side, max(f.box[3] * cfg.face_zoom, 32))))
        cx, cy = f.center + np.array([0.0, cfg.face_shift * f.box[3]])
    elif cfg.crop in ("face", "center"):
        side, cx, cy = max_side, w / 2, h / 2
    else:
        raise ValueError(f"unknown crop {cfg.crop!r} (expected 'face' or 'center')")
    x0 = int(round(np.clip(cx - side / 2, 0, w - side)))
    y0 = int(round(np.clip(cy - side / 2, 0, h - side)))
    return x0, y0, side


def _grabcut_fg(bgr: np.ndarray, face: Face, mask: np.ndarray) -> np.ndarray:
    """Soft foreground probability via GrabCut, seeded by a head-and-shoulders box around the
    face. (Saliency seeding was tried for face-less images and faded most of the subject away,
    so background fading is only applied when a face is found.)"""
    size = bgr.shape[0]
    gc = np.full((size, size), cv2.GC_PR_BGD, np.uint8)
    x, y, w, h = face.box
    x0, x1 = int(max(0, x - 0.9 * w)), int(min(size, x + 1.9 * w))
    gc[int(max(0, y - 0.6 * h)) :, x0:x1] = cv2.GC_PR_FGD
    fx, fy = int(max(0, x + 0.15 * w)), int(max(0, y + 0.1 * h))
    gc[fy : int(y + 0.9 * h), fx : int(x + 0.85 * w)] = cv2.GC_FGD
    gc[~mask] = cv2.GC_BGD
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.grabCut(bgr, gc, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
    fg = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.float64)
    return cv2.GaussianBlur(fg, (0, 0), size / 100)


def prepare(img: np.ndarray, cfg: PreprocessConfig) -> Prepared:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if cfg.smooth not in ("bilateral", "gaussian", "none"):
        raise ValueError(f"unknown smooth {cfg.smooth!r}")
    if cfg.background not in ("none", "fade"):
        raise ValueError(f"unknown background {cfg.background!r}")
    faces = detect_faces(img) if cfg.crop == "face" or cfg.background == "fade" else []
    x0, y0, side = _crop_box(img.shape, faces, cfg)
    crop = img[y0 : y0 + side, x0 : x0 + side]
    interp = cv2.INTER_AREA if side >= cfg.size else cv2.INTER_CUBIC
    bgr = cv2.resize(crop, (cfg.size, cfg.size), interpolation=interp)
    scale = cfg.size / side
    faces = [f.transformed(scale, x0, y0) for f in faces]
    faces = [f for f in faces if 0 <= f.center[0] < cfg.size and 0 <= f.center[1] < cfg.size]
    mask = frame_mask(cfg.frame, cfg.size)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    plain = gray.astype(np.float64) / 255.0
    if cfg.stretch not in ("auto", "on", "off"):
        raise ValueError(f"unknown stretch {cfg.stretch!r}")
    lo, hi = np.percentile(gray[mask], (1, 99))
    poor = hi - lo < cfg.stretch_below or lo > cfg.black_point_above or hi < cfg.white_point_below
    if cfg.stretch == "on" or (cfg.stretch == "auto" and poor):
        if hi - lo >= 8:  # leave (near-)flat images alone
            gray = np.clip((gray.astype(np.float64) - lo) * 255.0 / (hi - lo), 0, 255)
            gray = (gray + 0.5).astype(np.uint8)
    if cfg.smooth == "bilateral":
        gray = cv2.bilateralFilter(gray, 0, cfg.bilateral_sigma_color, max(1.0, cfg.size / 300))
    if cfg.clahe_clip > 0:
        clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(cfg.clahe_tiles,) * 2)
        gray = clahe.apply(gray)
    target = gray.astype(np.float64) / 255.0
    if cfg.smooth == "gaussian" and cfg.blur_sigma > 0:
        target = cv2.GaussianBlur(target, (0, 0), cfg.blur_sigma)

    fg = None
    if cfg.background == "fade" and faces:
        fg = _grabcut_fg(bgr, faces[0], mask)
        keep = cfg.background_keep + (1 - cfg.background_keep) * fg
        target = 1.0 - (1.0 - target) * keep
    target[~mask] = 1.0
    plain[~mask] = 1.0
    return Prepared(target, mask, plain, bgr, faces, (x0, y0, side), fg)


def preprocess(img: np.ndarray, cfg: PreprocessConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return (target, mask): grayscale float in [0, 1] (1 = white), white outside the frame."""
    p = prepare(img, cfg)
    return p.target, p.mask

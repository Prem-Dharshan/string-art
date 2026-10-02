"""Automatic importance maps (PLAN.md I5).

    W = floor + (1 - floor) * clip(a * face + b * edges + c * saliency, 0, 1)

* face: from the 68 LBF landmarks when available (face oval at 0.5; eyes, brows, nose and
  mouth at 1.0), else from YuNet's 5 points (ellipse oval + discs). Feathered.
* edges: Scharr gradient magnitude of the target, normalized at its 99th percentile.
* saliency: OpenCV spectral-residual saliency. It gets a stronger weight when no face is
  found, so pets and objects still get a focus region.

`auto_weights` (what the CLI uses by default) returns W only when a face was found. On the
M3 ablation, edge/saliency-only weights did not help face-less images (coffee -0.04 SSIM,
cat about 0), while face weights raised face-region SSIM by +0.04 to +0.10.

W multiplies the per-pixel squared error in the greedy solver. Because that error is
symmetric, important *light* regions (eye whites, teeth) are protected from stray threads as
strongly as dark ones are filled. That is the job the LessWrong negative weights did by hand.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .face import Face
from .preprocess import Prepared

# iBUG 68-point layout
JAW, BROW_R, BROW_L = range(0, 17), range(17, 22), range(22, 27)
NOSE, EYE_R, EYE_L, MOUTH = range(27, 36), range(36, 42), range(42, 48), range(48, 60)


@dataclass
class ImportanceConfig:
    face: float = 1.0
    edges: float = 0.5
    saliency: float = 0.25
    saliency_no_face: float = 0.7
    floor: float = 0.1
    edge_sigma: float = 1.5


def _pts(lm: np.ndarray, idx) -> np.ndarray:
    return np.round(lm[list(idx)]).astype(np.int32)


def face_map(faces: list[Face], size: int) -> tuple[np.ndarray, np.ndarray]:
    """(importance in [0, 1], face-oval ROI mask) for all faces on a size x size canvas."""
    feat = np.zeros((size, size), np.float32)
    oval = np.zeros((size, size), np.float32)
    for f in faces:
        x, y, w, h = f.box
        thick = max(2, int(round(0.07 * w)))
        if f.landmarks is not None and len(f.landmarks) == 68:
            lm = f.landmarks
            brows = lm[list(BROW_R) + list(BROW_L)].copy()
            brows[:, 1] -= 0.25 * h  # extend the hull up over the forehead
            hull = cv2.convexHull(np.round(np.vstack([lm[list(JAW)], brows])).astype(np.int32))
            cv2.fillConvexPoly(oval, hull, 1.0)
            for idx in (EYE_R, EYE_L, MOUTH):
                p = _pts(lm, idx)
                cv2.fillPoly(feat, [p], 1.0)
                cv2.polylines(feat, [p], True, 1.0, thick)
            for idx in (BROW_R, BROW_L, NOSE):
                cv2.polylines(feat, [_pts(lm, idx)], False, 1.0, thick)
        else:
            c = (int(x + w / 2), int(y + h / 2))
            cv2.ellipse(oval, c, (int(0.5 * w), int(0.62 * h)), 0, 0, 360, 1.0, -1)
            eye_r, eye_l, nose, m_r, m_l = np.round(f.five).astype(int)
            for p, r in ((eye_r, 0.13), (eye_l, 0.13), (nose, 0.09)):
                cv2.circle(feat, tuple(p), int(r * w), 1.0, -1)
            mc = tuple((m_r + m_l) // 2)
            half = int(max(np.linalg.norm(m_l - m_r) / 2, 0.1 * w) * 1.15)
            cv2.ellipse(feat, mc, (half, int(0.08 * h)), 0, 0, 360, 1.0, -1)
    if faces:
        sigma = max(1.0, 0.03 * max(f.box[2] for f in faces))
        feat = cv2.GaussianBlur(feat, (0, 0), sigma)
        oval = cv2.GaussianBlur(oval, (0, 0), sigma)
    return np.maximum(0.5 * oval, feat).astype(np.float64), oval > 0.5


def edge_map(target: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    gx = cv2.Scharr(target, cv2.CV_64F, 1, 0)
    gy = cv2.Scharr(target, cv2.CV_64F, 0, 1)
    mag = np.hypot(gx, gy)
    # Ignore the artificial edge where the photo meets the white outside of the frame.
    inner = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    mag[~inner] = 0
    if not inner.any():
        return np.zeros_like(target)
    ref = np.percentile(mag[mask], 99)
    if ref <= 1e-9:
        return np.zeros_like(target)
    return cv2.GaussianBlur(np.clip(mag / ref, 0, 1), (0, 0), sigma)


def saliency_map(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    sal = cv2.saliency.StaticSaliencySpectralResidual.create()
    ok, sm = sal.computeSaliency(bgr)
    if not ok:
        return np.zeros(mask.shape)
    sm = cv2.GaussianBlur(sm.astype(np.float64), (0, 0), bgr.shape[0] / 60)
    sm[~mask] = 0
    hi = sm[mask].max()
    return sm / hi if hi > 0 else sm


def importance(prep: Prepared, cfg: ImportanceConfig = ImportanceConfig()):  # noqa: B008
    """Return (W, parts) where parts holds the component maps and the face ROI mask."""
    size = prep.target.shape[0]
    face, roi = face_map(prep.faces, size)
    edges = edge_map(prep.target, prep.mask, cfg.edge_sigma)
    sal = saliency_map(prep.canvas_bgr, prep.mask)
    c_sal = cfg.saliency if prep.faces else cfg.saliency_no_face
    combo = np.clip(cfg.face * face + cfg.edges * edges + c_sal * sal, 0, 1)
    w = cfg.floor + (1 - cfg.floor) * combo
    w[~prep.mask] = cfg.floor
    return w, {"face": face, "edges": edges, "saliency": sal, "face_roi": roi}


def auto_weights(prep: Prepared, mode: str = "auto",
                 cfg: ImportanceConfig = ImportanceConfig()):  # noqa: B008
    """(W or None, parts). mode: "auto" = weights only when a face was found; "on"; "off"."""
    if mode not in ("auto", "on", "off"):
        raise ValueError(f"unknown importance mode {mode!r}")
    w, parts = importance(prep, cfg)
    use = mode == "on" or (mode == "auto" and bool(prep.faces))
    return (w if use else None), parts

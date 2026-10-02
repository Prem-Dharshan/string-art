"""Face detection (YuNet) and 68-point landmarks (LBF), both from OpenCV. Both degrade
gracefully: no model file or no face -> empty result, and callers fall back to saliency."""

import functools
from dataclasses import dataclass

import cv2
import numpy as np

from .models import model_path


@dataclass
class Face:
    box: np.ndarray  # x, y, w, h
    score: float
    five: np.ndarray  # (5, 2): right eye, left eye, nose tip, right/left mouth corner
    landmarks: np.ndarray | None = None  # (68, 2) iBUG layout, if LBF is available

    def transformed(self, scale: float, dx: float, dy: float) -> "Face":
        """Map into another frame: p' = (p - (dx, dy)) * scale."""
        off = np.array([dx, dy])
        lm = None if self.landmarks is None else (self.landmarks - off) * scale
        box = np.array([(self.box[0] - dx) * scale, (self.box[1] - dy) * scale,
                        self.box[2] * scale, self.box[3] * scale])
        return Face(box, self.score, (self.five - off) * scale, lm)

    @property
    def center(self) -> np.ndarray:
        x, y, w, h = self.box
        return np.array([x + w / 2, y + h / 2])


@functools.cache
def _yunet():
    path = model_path("yunet")
    if not path.is_file():
        return None
    return cv2.FaceDetectorYN.create(str(path), "", (320, 320), 0.7, 0.3, 5000)


@functools.cache
def _lbf():
    path = model_path("lbf")
    if not path.is_file():
        return None
    fm = cv2.face.createFacemarkLBF()
    fm.loadModel(str(path))
    return fm


def detect_faces(img: np.ndarray, score_thresh: float = 0.7, max_side: int = 800,
                 landmarks: bool = True) -> list[Face]:
    """Faces in a BGR image, largest first. Returns [] when the model is missing."""
    det = _yunet()
    if det is None:
        return []
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    det.setInputSize((small.shape[1], small.shape[0]))
    det.setScoreThreshold(score_thresh)
    _, raw = det.detect(small)
    if raw is None:
        return []
    faces = [Face(r[:4] / s, float(r[14]), r[4:14].reshape(5, 2) / s) for r in raw]
    faces.sort(key=lambda f: f.box[2] * f.box[3], reverse=True)
    fm = _lbf() if landmarks else None
    if fm is not None:
        rects = np.array([f.box for f in faces], dtype=np.int32)
        ok, lms = fm.fit(img, rects)
        if ok:
            for f, lm in zip(faces, lms, strict=True):
                f.landmarks = np.asarray(lm, dtype=np.float64).reshape(-1, 2)
    return faces

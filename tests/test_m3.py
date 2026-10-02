import json

import cv2
import numpy as np
import pytest

from stringart import face as face_mod
from stringart.cli import main
from stringart.face import Face, detect_faces
from stringart.importance import ImportanceConfig, auto_weights, edge_map, face_map, importance
from stringart.metrics import evaluate
from stringart.models import model_path
from stringart.preprocess import PreprocessConfig, _crop_box, load_image, prepare

needs_models = pytest.mark.skipif(
    not (model_path("yunet").is_file() and model_path("lbf").is_file()),
    reason="face models not downloaded (stringart fetch-models)",
)


def _fake_face(x=40, y=30, w=40, h=50, landmarks=False):
    five = np.array([[x + 12, y + 18], [x + 28, y + 18], [x + 20, y + 28], [x + 13, y + 38],
                     [x + 27, y + 38]], float)
    return Face(np.array([x, y, w, h], float), 0.9, five, None)


@needs_models
def test_astronaut_face_and_landmarks():
    faces = detect_faces(load_image("sample:astronaut"))
    assert len(faces) == 1
    f = faces[0]
    assert f.landmarks.shape == (68, 2)
    x, y, w, h = f.box
    inside = ((f.landmarks[:, 0] > x - 0.2 * w) & (f.landmarks[:, 0] < x + 1.2 * w)
              & (f.landmarks[:, 1] > y - 0.2 * h) & (f.landmarks[:, 1] < y + 1.2 * h))
    assert inside.mean() > 0.9


@needs_models
def test_face_crop_centres_face():
    p = prepare(load_image("sample:astronaut"), PreprocessConfig(size=300))
    assert len(p.faces) == 1
    cx, cy = p.faces[0].center
    assert abs(cx - 150) < 30 and abs(cy - 150) < 60
    assert p.faces[0].box[3] > 300 / 2.5  # face fills a good part of the frame


def test_missing_models_degrade_to_no_faces(tmp_path, monkeypatch):
    monkeypatch.setenv("STRINGART_MODELS", str(tmp_path))
    face_mod._yunet.cache_clear()
    face_mod._lbf.cache_clear()
    try:
        assert detect_faces(load_image("sample:astronaut")) == []
        p = prepare(load_image("sample:astronaut"), PreprocessConfig(size=128))
        assert p.faces == [] and p.crop[2] == 512  # centre crop of the full 512px image
    finally:
        face_mod._yunet.cache_clear()
        face_mod._lbf.cache_clear()


def test_crop_box_clamped_inside_image():
    f = _fake_face(x=2, y=1, w=30, h=30)
    x0, y0, side = _crop_box((200, 300, 3), [f], PreprocessConfig(face_zoom=2.0))
    assert x0 == 0 and y0 == 0 and side == 60
    x0, y0, side = _crop_box((200, 300, 3), [f], PreprocessConfig(face_zoom=50))
    assert side == 200 and 0 <= x0 <= 100


def test_legacy_config_matches_m2_chain():
    img = load_image("sample:camera")
    p = prepare(img, PreprocessConfig.legacy(size=128))
    gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (128, 128),
                      interpolation=cv2.INTER_AREA)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    ref = cv2.GaussianBlur(gray / 255.0, (0, 0), 1.0)
    ref[~p.mask] = 1.0
    assert np.allclose(p.target, ref)


def test_flat_image_is_stable():
    p = prepare(np.full((90, 120, 3), 128, np.uint8), PreprocessConfig(size=64))
    assert np.isfinite(p.target).all()
    assert np.all(edge_map(p.target, p.mask, 1.0) == 0)
    w, _ = importance(p)
    assert np.isfinite(w).all()


def test_auto_stretch_only_for_poor_exposure():
    rng = np.random.default_rng(0)
    dark = rng.integers(10, 70, (100, 100, 3), dtype=np.uint8)  # narrow, underexposed
    ramp = np.tile(np.linspace(0, 255, 100), (100, 1)).astype(np.uint8)
    full = np.dstack([ramp] * 3)  # full 0..255 range
    kw = {"size": 64, "crop": "center", "smooth": "none", "clahe_clip": 0}
    washed = np.dstack([np.tile(np.linspace(100, 250, 100), (100, 1)).astype(np.uint8)] * 3)
    for img, should_change in ((dark, True), (full, False), (washed, True)):
        auto = prepare(img, PreprocessConfig(stretch="auto", **kw)).target
        off = prepare(img, PreprocessConfig(stretch="off", **kw)).target
        assert (not np.allclose(auto, off)) == should_change
    with pytest.raises(ValueError):
        prepare(full, PreprocessConfig(stretch="maybe", **kw))


def test_auto_weights_follow_faces():
    img = np.random.default_rng(0).integers(0, 255, (100, 100, 3), dtype=np.uint8)
    p = prepare(img, PreprocessConfig(size=64, crop="center"))
    assert auto_weights(p, "auto")[0] is None
    assert auto_weights(p, "off")[0] is None
    assert auto_weights(p, "on")[0].shape == (64, 64)
    p.faces = [_fake_face(x=10, y=10, w=30, h=30)]
    assert auto_weights(p, "auto")[0] is not None


def test_background_fade_needs_a_face():
    img = np.random.default_rng(0).integers(0, 255, (100, 100, 3), dtype=np.uint8)
    a = prepare(img, PreprocessConfig(size=64, crop="center"))
    b = prepare(img, PreprocessConfig(size=64, crop="center", background="fade"))
    assert b.fg is None and np.array_equal(a.target, b.target)


def test_five_point_face_map():
    imp, roi = face_map([_fake_face()], 128)
    eye = imp[48, 52]  # right-eye point (x=52, y=48)
    assert eye > 0.6 and imp[5, 5] == 0
    assert roi[55, 60] and not roi[5, 5]


def test_importance_range_and_focus():
    img = load_image("sample:astronaut")
    p = prepare(img, PreprocessConfig(size=200))
    cfg = ImportanceConfig(floor=0.1)
    w, parts = importance(p, cfg)
    assert w.shape == p.target.shape
    assert w.min() >= 0.1 - 1e-9 and w.max() <= 1 + 1e-9
    if p.faces:  # face pixels matter more than the average pixel
        assert w[parts["face_roi"]].mean() > w[p.mask].mean()


def test_evaluate_roi_keys():
    a = np.random.default_rng(1).random((64, 64))
    roi = np.zeros((64, 64), bool)
    roi[10:30, 10:30] = True
    m = evaluate(a, a * 0.9, sigmas=(2,), roi=roi)
    assert {"ssim_s2", "psnr_s2", "ssim_roi_s2", "psnr_roi_s2"} <= m.keys()
    assert "ssim_roi_s2" not in evaluate(a, a, sigmas=(2,), roi=np.zeros((64, 64), bool))


def test_cli_run_with_importance(tmp_path):
    out = tmp_path / "run"
    main(["run", "sample:astronaut", "--size", "128", "--pins", "64", "--min-gap", "5",
          "--out", str(out), "--quiet"])
    assert (out / "importance.png").is_file()
    meta = json.loads((out / "metrics.json").read_text())
    assert {"vs_target", "vs_photo"} <= meta["metrics"].keys()

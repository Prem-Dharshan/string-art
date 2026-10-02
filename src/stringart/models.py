"""Pretrained OpenCV face models: download on demand into `models/` (gitignored)."""

import hashlib
import os
import urllib.request
from pathlib import Path

MODELS = {
    # name: (url, approx size MB)
    "yunet": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
        "face_detection_yunet_2023mar.onnx",
        0.3,
    ),
    "lbf": (
        "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml",
        54,
    ),
}
FILENAMES = {"yunet": "face_detection_yunet_2023mar.onnx", "lbf": "lbfmodel.yaml"}


def models_dir() -> Path:
    """`$STRINGART_MODELS`, else `<repo>/models` when running from a checkout, else ~/.stringart."""
    if env := os.environ.get("STRINGART_MODELS"):
        return Path(env)
    repo = Path(__file__).resolve().parents[2] / "models"
    return repo if repo.is_dir() else Path.home() / ".stringart" / "models"


def model_path(name: str) -> Path:
    return models_dir() / FILENAMES[name]


def fetch(name: str, force: bool = False) -> Path:
    url, size_mb = MODELS[name]
    dest = model_path(name)
    if dest.is_file() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {name} (~{size_mb} MB) -> {dest}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    if tmp.stat().st_size < 1000:  # an LFS pointer or error page, not a model
        tmp.unlink()
        raise RuntimeError(f"download of {name} from {url} looks invalid")
    tmp.replace(dest)
    return dest


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

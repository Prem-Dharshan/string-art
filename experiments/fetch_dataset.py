"""Download the evaluation set described in data/dataset.json from Wikimedia Commons.

* Images go to data/raw/<id>.jpg (gitignored), scaled to `width` px by Commons.
* Licence, author and source page of every image are written to data/raw/manifest.json and to
  data/ATTRIBUTION.md (committed: CC BY / BY-SA require attribution).
* `derived_from` entries are synthetic exposure variants of another image, for testing
  robustness against a known-good source.

    uv run python experiments/fetch_dataset.py [--force]
"""

import argparse
import html
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "data" / "dataset.json"
RAW = ROOT / "data" / "raw"
UA = "stringart-academic-project/0.1 (https://github.com/Prem-Dharshan/string-art)"
API = "https://commons.wikimedia.org/w/api.php?"
ALLOWED = re.compile(
    r"^(public domain|pd|cc0|cc by(-sa)?( [0-9.]+)?( [a-z]{2})?|no restrictions)", re.I
)


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(4):
        try:
            return urllib.request.urlopen(req, timeout=120).read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def _strip(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


def image_info(title: str, width: int) -> dict:
    q = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "imageinfo",
        "iiprop": "url|size|extmetadata",
        "iiurlwidth": width,
    }
    page = next(iter(json.loads(_get(API + urllib.parse.urlencode(q)))["query"]["pages"].values()))
    if "imageinfo" not in page:
        raise RuntimeError(f"{title}: not found on Commons")
    ii = page["imageinfo"][0]
    md = ii.get("extmetadata", {})
    return {
        "url": ii.get("thumburl") or ii["url"],
        "page": ii["descriptionurl"],
        "license": _strip(md.get("LicenseShortName", {}).get("value", "")),
        "license_url": md.get("LicenseUrl", {}).get("value", ""),
        "artist": _strip(md.get("Artist", {}).get("value", "")) or "unknown",
        "original_size": [ii["width"], ii["height"]],
    }


# Synthetic exposure degradations, applied in 8-bit RGB.
def underexpose(img):
    return np.clip(255.0 * (img / 255.0) ** 2.2 * 0.55, 0, 255).astype(np.uint8)


def low_contrast(img):
    return np.clip(95 + img * (65 / 255.0), 0, 255).astype(np.uint8)


def washed_out(img):
    return np.clip(255.0 * (img / 255.0) ** 0.4 * 0.9 + 25, 0, 255).astype(np.uint8)


TRANSFORMS = {"underexpose": underexpose, "low_contrast": low_contrast, "washed_out": washed_out}


def _write(path: Path, img) -> None:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError(f"could not encode {path}")
    buf.tofile(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    args = ap.parse_args()
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    RAW.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    )

    for item in spec["images"]:
        if "title" not in item:
            continue
        dest = RAW / f"{item['id']}.jpg"
        if dest.is_file() and item["id"] in manifest and not args.force:
            continue
        info = image_info(item["title"], spec["width"])
        if not ALLOWED.match(info["license"]):
            raise RuntimeError(f"{item['title']}: licence {info['license']!r} not allowed")
        img = cv2.imdecode(np.frombuffer(_get(info["url"]), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"{item['title']}: could not decode download")
        _write(dest, img)
        manifest[item["id"]] = {**item, **info, "size": [img.shape[1], img.shape[0]]}
        print(f"{item['id']:32s} {img.shape[1]}x{img.shape[0]}  {info['license']}")
        time.sleep(0.5)  # be polite to Commons

    for item in spec["images"]:
        if "derived_from" not in item:
            continue
        src = cv2.imdecode(
            np.fromfile(RAW / f"{item['derived_from']}.jpg", np.uint8), cv2.IMREAD_COLOR
        )
        _write(RAW / f"{item['id']}.jpg", TRANSFORMS[item["transform"]](src.astype(np.float64)))
        base = manifest[item["derived_from"]]
        manifest[item["id"]] = {
            **item,
            "license": base["license"],
            "artist": base["artist"],
            "page": base["page"],
            "license_url": base.get("license_url", ""),
        }
        print(f"{item['id']:32s} derived ({item['transform']}) from {item['derived_from']}")

    manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Image attribution",
        "",
        "Evaluation images from Wikimedia Commons. `h02`–`h04` are synthetic exposure "
        "variants of `f07` made by `experiments/fetch_dataset.py`.",
        "",
        "| id | author | licence | source |",
        "|---|---|---|---|",
    ]
    for item in spec["images"]:
        m = manifest[item["id"]]
        lic = f"[{m['license']}]({m['license_url']})" if m.get("license_url") else m["license"]
        lines.append(
            f"| {item['id']} | {m['artist'].replace('|', '/')} | {lic} | [Commons]({m['page']}) |"
        )
    (ROOT / "data" / "ATTRIBUTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(spec['images'])} images ready in {RAW}")


if __name__ == "__main__":
    main()

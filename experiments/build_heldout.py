"""Build data/heldout.json: a held-out set chosen by a fixed rule, never looked at for tuning.

Rule (no hand-picking): for each Commons category below, skip the first 60 members (the pool
browsed by hand when the M6 set was curated) and anything already in data/dataset.json, then
take the first members, in the API's own order, that pass the licence filter and, for face
categories, the same automatic face filter used before: exactly one YuNet face, score > 0.85,
face height >= 15% of the shorter side. Synthetic exposure faults are made from the first three
held-out faces, at strengths that differ from the M6 faults.

    uv run python experiments/build_heldout.py
    uv run python experiments/fetch_dataset.py --spec data/heldout.json --raw data/raw_heldout \
        --attribution data/ATTRIBUTION_heldout.md
"""

import json
import sys
import time
import urllib.parse
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_dataset import ALLOWED, API, ROOT, _get, _strip  # noqa: E402

from stringart.face import detect_faces  # noqa: E402

SKIP_FIRST = 60
QUOTAS = [  # (category, how many, needs a face, category label)
    ("Quality images of men", 4, True, "face"),
    ("Quality images of women", 4, True, "face"),
    ("Quality images of children", 2, True, "face"),
    ("Quality images of cats", 1, False, "animal"),
    ("Quality images of dogs", 2, False, "animal"),
    ("Quality images of horses", 2, False, "animal"),
    ("Quality images of lighthouses", 1, False, "object"),
]
FAULTS = [  # (transform, params) — different strengths from the M6 faults
    ("underexpose", {"gamma": 1.8, "gain": 0.7}),
    ("low_contrast", {"lo": 60, "span": 90}),
    ("washed_out", {"gamma": 0.5, "gain": 0.85, "lift": 35}),
    ("underexpose", {"gamma": 2.6, "gain": 0.45}),
    ("low_contrast", {"lo": 120, "span": 50}),
    ("washed_out", {"gamma": 0.35, "gain": 0.95, "lift": 15}),
]


def _api(**p):
    p.update(action="query", format="json")
    return json.loads(_get(API + urllib.parse.urlencode(p)))


def members(cat: str) -> list[str]:
    r = _api(list="categorymembers", cmtitle=f"Category:{cat}", cmtype="file", cmlimit=500)
    return [m["title"] for m in r["query"]["categorymembers"]]


def info(titles: list[str]) -> dict[str, dict]:
    out = {}
    for i in range(0, len(titles), 10):
        q = _api(
            titles="|".join(titles[i : i + 10]),
            prop="imageinfo",
            iiprop="url|extmetadata",
            iiurlwidth=640,
        )
        for pg in q["query"]["pages"].values():
            if "imageinfo" in pg:
                ii = pg["imageinfo"][0]
                lic = _strip(ii.get("extmetadata", {}).get("LicenseShortName", {}).get("value"))
                out[pg["title"]] = {"thumb": ii.get("thumburl") or ii["url"], "license": lic}
    return out


def main() -> None:
    used = {
        it["title"]
        for it in json.loads((ROOT / "data" / "dataset.json").read_text(encoding="utf-8"))["images"]
        if "title" in it
    }
    images, faces = [], []
    for cat, k, need_face, label in QUOTAS:
        pool = [t for t in members(cat)[SKIP_FIRST:] if t not in used]
        meta = info(pool[: 12 * k])
        taken = 0
        for title in pool[: 12 * k]:
            m = meta.get(title)
            if taken == k or m is None or not ALLOWED.match(m["license"]):
                continue
            if need_face:
                img = cv2.imdecode(np.frombuffer(_get(m["thumb"]), np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                det = detect_faces(img, landmarks=False)
                side = min(img.shape[:2])
                if not (len(det) == 1 and det[0].score > 0.85 and det[0].box[3] >= 0.15 * side):
                    continue
                time.sleep(0.2)
            taken += 1
            iid = f"x{len(images) + 1:02d}_{label}"
            images.append({"id": iid, "category": label, "title": title, "rule": cat})
            if need_face:
                faces.append(iid)
        print(f"{cat}: {taken}/{k}")
    for j, (transform, params) in enumerate(FAULTS):
        src = faces[j % 3]
        images.append(
            {
                "id": f"x{len(images) + 1:02d}_fault",
                "category": "hard",
                "derived_from": src,
                "transform": transform,
                "params": params,
            }
        )
    spec = {
        "description": "Held-out set (never used for tuning), chosen by the rule in "
        "experiments/build_heldout.py",
        "attribution_note": "Held-out images from Wikimedia Commons, chosen by a fixed rule. "
        "`*_fault` images are synthetic exposure faults of held-out faces.",
        "width": 1600,
        "images": images,
    }
    path = ROOT / "data" / "heldout.json"
    path.write_text(json.dumps(spec, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}: {len(images)} images")


if __name__ == "__main__":
    main()

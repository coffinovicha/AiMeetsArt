import csv, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
L = ROOT / "labels"
OUT = L / "palettes.json"


def load(path, crop=None):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if crop:
        w, h = im.size
        im = im.crop((int(crop[0] * w), int(crop[1] * h), int(crop[2] * w), int(crop[3] * h)))
    return im


def palette(im, k=6, side=200, seed=0):
    im = im.copy()
    im.thumbnail((side, side))
    px = np.asarray(im).reshape(-1, 3).astype(np.float32)
    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(px)
    counts = np.bincount(km.labels_, minlength=k)
    order = np.argsort(-counts)
    return [{"hex": "#%02x%02x%02x" % tuple(int(round(c)) for c in km.cluster_centers_[i]),
             "pct": round(100 * counts[i] / counts.sum(), 1)} for i in order]


if __name__ == "__main__":
    done = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    crops = {}
    if (L / "annotations.jsonl").exists():
        for line in (L / "annotations.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                a = json.loads(line)
                crops[a["id"]] = a.get("crop")
    with (L / "manifest.csv").open(encoding="utf-8") as f:
        items = list(csv.DictReader(f))
    n = 0
    for it in items:
        c = crops.get(it["id"])
        prev = done.get(it["id"])
        if isinstance(prev, dict) and prev.get("crop") == c:
            continue
        done[it["id"]] = {"crop": c, "colors": palette(load(ROOT / it["file"], c))}
        n += 1
    OUT.write_text(json.dumps(done, indent=1), encoding="utf-8")
    print(n, "computed,", len(done), "total")

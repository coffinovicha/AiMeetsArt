import csv, json, os, re
from collections import defaultdict
from functools import lru_cache
import numpy as np
import torch
from vlm_common import ROOT

DB = ROOT / "artificio_data"
EMB = "clip_ViT-H-14-quickgelu_dfn5b"
CLIP_NAME, CLIP_PRE = "ViT-H-14-quickgelu", "dfn5b"


DB_SET = os.environ.get("DB_SET", "full")
FULL = DB / "full"


def _year(v):
    m = re.search(r"(1[4-9]|20)\d\d", str(v or ""))
    return m.group() if m else ""


def _load_full():
    meta = []
    for r in csv.DictReader(open(FULL / "metadata.csv", encoding="utf-8")):
        meta.append({"uid": f"full_{r['row']}", "artist": (r["artist"] or "").strip(), "title": (r["title"] or "").strip(),
                     "year": _year(r["date"]), "date": r["date"], "style": r["style"], "genre": r["genre"], "file": ""})
    files = sorted(FULL.glob("emb_*.npy"), key=lambda f: (int(f.stem.split("_")[1]), int(f.stem.split("_")[2])))
    E = np.concatenate([np.load(f) for f in files]).astype("float32")
    assert len(E) == len(meta), (len(E), len(meta))
    return meta, E


@lru_cache(maxsize=1)
def load_db():
    import faiss
    if DB_SET == "full":
        meta, E = _load_full()
    else:
        rows = {r["uid"]: r for r in csv.DictReader(open(DB / "metadata.csv", encoding="utf-8"))}
        uids = (DB / f"{EMB}_uids.txt").read_text(encoding="utf-8").split()
        E = np.load(DB / f"{EMB}.npy").astype("float32")
        meta = [rows[u] for u in uids]
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(E.shape[1])
    index.add(E)
    by_artist = defaultdict(list)
    for k, m in enumerate(meta):
        by_artist[m["artist"]].append(k)
    return index, E, meta, dict(by_artist)


@lru_cache(maxsize=1)
def clip_model():
    import open_clip
    dev = os.environ.get("CLIP_DEVICE", "cuda")
    m, _, prep = open_clip.create_model_and_transforms(CLIP_NAME, pretrained=CLIP_PRE, device=dev, precision="fp16" if dev == "cuda" else "fp32")
    m.eval()
    return m, prep


@torch.no_grad()
def embed(images):
    m, prep = clip_model()
    p = m.visual.conv1.weight
    x = torch.stack([prep(im.convert("RGB")) for im in images]).to(p.device, p.dtype)
    v = m.encode_image(x).float()
    return (v / v.norm(dim=-1, keepdim=True)).cpu().numpy()


def search(vec, k=10):
    index, _, meta, _ = load_db()
    s, i = index.search(vec.reshape(1, -1).astype("float32"), k)
    return [{"row": int(j), "score": float(x), **{f: meta[j][f] for f in ("uid", "artist", "title", "year", "style", "genre", "file")}}
            for x, j in zip(s[0], i[0])]


if __name__ == "__main__":
    from palette import load
    from vlm_common import records
    index, E, meta, by_artist = load_db()
    out = ROOT / "outputs" / "gallery_vith.npy"
    recs = records()
    if not out.exists():
        V = np.concatenate([embed([load(ROOT / r["file"], r["crop"]) for r in recs[s:s + 16]]) for s in range(0, len(recs), 16)])
        np.save(out, V)
        (ROOT / "outputs" / "gallery_vith_ids.txt").write_text("\n".join(r["id"] for r in recs), encoding="utf-8")
    V = np.load(out)
    S, I = index.search(E, 2)
    nn_score, nn_row = S[:, 1], I[:, 1]
    same = np.array([meta[a]["artist"] == meta[b]["artist"] for a, b in enumerate(nn_row)])
    bins = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]
    table = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (nn_score >= lo) & (nn_score < hi)
        table.append({"from": lo, "to": hi, "n": int(m.sum()), "same_artist": float(same[m].mean()) if m.any() else None})
    above = []
    for t in [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        m = nn_score >= t
        above.append({"threshold": t, "share_db": float(m.mean()), "same_artist_precision": float(same[m].mean())})
    G, GI = index.search(V.astype("float32"), 1)
    g = {"min": float(G.min()), "median": float(np.median(G)), "max": float(G.max()),
         "share_above": {str(t): float((G[:, 0] >= t).mean()) for t in [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]}}
    res = {"db_nearest_neighbor_bins": table, "db_threshold_precision": above, "gallery_top1": g}
    (ROOT / "outputs" / "db_similarity.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    for r, s, j in sorted(zip(recs, G[:, 0], GI[:, 0]), key=lambda x: -x[1])[:12]:
        print(r["id"], round(float(s), 3), "|", r["known_artist"], "|", meta[j]["artist"], "-", meta[j]["title"])

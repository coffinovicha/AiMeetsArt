import json
from collections import Counter
import numpy as np
import torch
from db_index import load_db, DB_SET
from vlm_common import ROOT, records

_, E, meta, by_artist = load_db()
Et = torch.from_numpy(E).half().cuda()
nn_s, nn_i = [], []
for s in range(0, len(E), 2048):
    S = Et[s:s + 2048] @ Et.T
    S[torch.arange(S.shape[0]), torch.arange(s, s + S.shape[0])] = -1
    v, i = S.max(1)
    nn_s.append(v.float().cpu())
    nn_i.append(i.cpu())
nn_s, nn_i = torch.cat(nn_s).numpy(), torch.cat(nn_i).numpy()
same = np.array([meta[a]["artist"] == meta[b]["artist"] for a, b in enumerate(nn_i)])
bins = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]
table = []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (nn_s >= lo) & (nn_s < hi)
    table.append({"from": lo, "to": hi, "n": int(m.sum()), "same_artist": round(float(same[m].mean()), 3) if m.any() else None})
fine = []
for t in np.arange(0.70, 0.99, 0.01):
    m = (nn_s >= t) & (nn_s < t + 0.01)
    if m.sum() >= 30:
        fine.append((round(float(t), 2), int(m.sum()), round(float(same[m].mean()), 3)))
threshold = next((t for t, n, p in fine if p >= 0.5 and all(q >= 0.5 for tt, nn, q in fine if tt >= t)), None)
near_dup = next((t for t, n, p in fine if p >= 0.9 and all(q >= 0.9 for tt, nn, q in fine if tt >= t)), None)

ids = (ROOT / "outputs" / "gallery_vith_ids.txt").read_text(encoding="utf-8").split()
V = np.load(ROOT / "outputs" / "gallery_vith.npy").astype("float32")
G = (torch.from_numpy(V).half().cuda() @ Et.T).float().cpu().numpy()
top = G.max(1)
res = {"db_set": DB_SET, "artworks": len(E), "artists": len(by_artist),
       "nearest_neighbor_bins": table, "fine_bins": fine, "b_threshold": threshold, "near_duplicate_threshold": near_dup,
       "gallery_top1_vs_wikiart": {"median": round(float(np.median(top)), 3), "max": round(float(top.max()), 3),
                                    "share_above": {str(t): round(float((top >= t).mean()), 3) for t in (0.75, 0.8, 0.85, 0.9)}}}
best = G.argmax(1)
res["gallery_best_wikiart_matches"] = sorted([{"id": i, "score": round(float(top[k]), 3), "artist": meta[best[k]]["artist"], "title": meta[best[k]]["title"]}
                                              for k, i in enumerate(ids)], key=lambda x: -x["score"])[:15]
(ROOT / "outputs" / f"db_similarity_{DB_SET}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k not in ("fine_bins",)}, indent=1, ensure_ascii=False))

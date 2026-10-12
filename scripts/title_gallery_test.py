import json
import numpy as np
from titles import enc, title_cascade, year_of
from vlm_common import ROOT, load, records

DUP = {"F1-016": "F1-017", "F1-017": "F1-016", "F1-040": "F1-046", "F1-046": "F1-040", "F1-056": "F1-062", "F1-062": "F1-056"}
SETUPS = {"artist_known_grounded": dict(artist=True, year=False, use_era=False, evocative=True),
          "artist_unknown_grounded": dict(artist=False, year=False, use_era=False, evocative=True)}
out = ROOT / "outputs" / "title_gallery_test_val_top5_b070.json"
R = {r["id"]: r for r in records()}
P = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "outputs" / "preds_verified2_val.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
ids = (ROOT / "outputs" / "gallery_vith_ids.txt").read_text(encoding="utf-8").split()
V = np.load(ROOT / "outputs" / "gallery_vith.npy").astype("float32")
todo = [i for i in P if R[i]["known_title"]]
model, proc = load(str(ROOT / "models" / "qwen_lora_v2"))
model.eval()
rows = [x for x in json.loads(out.read_text(encoding="utf-8"))["rows"] if all(n in x for n in SETUPS)] if out.exists() else []
done = {x["id"] for x in rows}
for i in todo:
    if i in done:
        continue
    r, d = R[i], P[i]
    q = V[ids.index(i)]
    excl = [i] + ([DUP[i]] if i in DUP else [])
    row = {"id": i, "artist": r["known_artist"], "real": r["known_title"], "year": r["known_year"]}
    for name, s in SETUPS.items():
        res = title_cascade(model, proc, q, d["pred"], d["objects"], d.get("palette"),
                            artist=r["known_artist"] if s["artist"] else None, exclude_ids=excl,
                            year=year_of(r["known_year"]) if s["year"] else None, use_era=s["use_era"], evocative=s["evocative"], top=5)
        row[name] = {"titles": [t["title"] for t in res["titles"]], "path": res["path"], "why": res["why"],
                     "match": {k: res["match"][k] for k in ("artist", "title", "score", "source")} if res["match"] else None,
                     "era": res["era"], "rejected": [(x["title"], x["rejected"]) for x in res["rejected"]]}
    rows.append(row)
    out.write_text(json.dumps({"rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(i, "| real:", r["known_title"], "|", {n: (row[n]["path"], row[n]["titles"][:1]) for n in SETUPS}, flush=True)

from titles import norm
real = enc([x["real"] for x in rows])
summ = {}
for n in SETUPS:
    gen = enc([x[n]["titles"][0] if x[n]["titles"] else "" for x in rows])
    exact = close = 0
    best = []
    for k, x in enumerate(rows):
        ts = x[n]["titles"][:5]
        exact += any(norm(t) == norm(x["real"]) for t in ts)
        sims = (enc(ts) @ real[k]) if ts else np.array([0.0])
        best.append(float(sims.max()))
        close += bool(sims.max() >= 0.7)
    summ[n] = {"similarity_to_real_top1": round(float((real * gen).sum(1).mean()), 3),
               "top5_exact_hits": exact, "top5_close_hits_sim_0.7": close, "top5_best_similarity_mean": round(float(np.mean(best)), 3),
               "paths": {p: sum(x[n]["path"] == p for x in rows) for p in "abc"}}
perm = np.roll(np.arange(len(rows)), 1)
summ["baseline_other_painting"] = round(float((real * real[perm]).sum(1).mean()), 3)
out.write_text(json.dumps({"summary": summ, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summ, indent=1))

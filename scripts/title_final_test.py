import os
os.environ.setdefault("VLM_SIDE", "336")
os.environ.setdefault("DB_SET", "full")
import argparse, json, re
import numpy as np
from interpret import TITLE, adapter_titles
from titles import FRAMES, color_name, enc, norm, title_cascade, ungrounded_words
from vlm_common import ROOT, load, load_image, records

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test")
a = ap.parse_args()
DUP = {"F1-016": "F1-017", "F1-017": "F1-016", "F1-040": "F1-046", "F1-046": "F1-040", "F1-056": "F1-062", "F1-062": "F1-056"}
GENRE = {"untitled", "abstraction", "abstract", "still", "life", "portrait", "landscape", "composition"} | FRAMES
out = ROOT / "outputs" / f"title_final_{a.split}.json"
R = {r["id"]: r for r in records()}
P = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "outputs" / f"preds_verified2_{a.split}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
ids = (ROOT / "outputs" / "gallery_vith_ids.txt").read_text(encoding="utf-8").split()
V = np.load(ROOT / "outputs" / "gallery_vith.npy").astype("float32")
todo = [i for i in P if P[i].get("pred") and (a.split == "test" or R[i]["known_title"])]
model, proc = load(str(TITLE))
model.eval()


def ground(d):
    pred = d["pred"]
    g = {w.lower() for f in ("description", "theme", "mood", "lighting", "composition", "style_guess") for w in re.findall(r"[A-Za-z]+", str(pred.get(f) or ""))}
    g |= {w.lower() for o in d["objects"] if o["status"] == "confirmed" for w in re.findall(r"[A-Za-z]+", o["object"])}
    g |= {w for c in (d.get("palette") or [])[:6] for w in color_name(c["hex"]).split()}
    return g


rows = json.loads(out.read_text(encoding="utf-8"))["rows"] if out.exists() else []
done = {x["id"] for x in rows}
for i in todo:
    if i in done:
        continue
    r, d = R[i], P[i]
    img = load_image(ROOT / r["file"], d.get("crop") or r["crop"], side=336)
    cache = {}

    def extra(art):
        if art not in cache:
            cache[art] = adapter_titles(model, proc, img, art)
        return cache[art]

    row = {"id": i, "artist": r["known_artist"], "real": r["known_title"], "year": r["known_year"]}
    g = ground(d)
    for mode in (("artist_known", "artist_unknown") if r["known_title"] else ("artist_unknown",)):
        res = title_cascade(model, proc, V[ids.index(i)], d["pred"], d["objects"], d.get("palette"),
                            artist=r["known_artist"].strip() if mode == "artist_known" else None,
                            exclude_ids=[i] + ([DUP[i]] if i in DUP else []), use_era=False, evocative=True, top=5,
                            extra=extra, writer=True, artistic=True)
        ts = res["titles"]
        row[mode] = {"titles": [t["title"] for t in ts], "sources": [t.get("source", "fallback") for t in ts], "prefix": ts[0].get("prefix") if ts else None,
                     "pattern_margin": [t.get("pattern_margin") for t in ts], "pattern_rank": [t.get("pattern_rank") for t in ts],
                     "path": res["path"], "why": res["why"],
                     "match": {k: res["match"][k] for k in ("artist", "title", "score", "source")} if res["match"] else None,
                     "invented": {t["title"]: ungrounded_words(t["title"].replace(" — ", " "), g, GENRE) for t in ts},
                     "n_rejected": len(res["rejected"])}
    rows.append(row)
    out.write_text(json.dumps({"rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(i, "| real:", r["known_title"] or "-", "|", {m: (row[m]["path"], row[m]["titles"][:2]) for m in ("artist_known", "artist_unknown") if m in row}, flush=True)


def hit(t, real):
    return norm(t) == norm(real) or (" — " in t and norm(t.split(" — ")[0]) == norm(real))


summ = {}
for mode in ("artist_known", "artist_unknown"):
    rs = [x for x in rows if mode in x]
    titled = [x for x in rs if x["real"]]
    m = {"paintings": len(rs), "paths": {p: sum(x[mode]["path"] == p for x in rs) for p in "abc"},
         "titles_shown": sum(len(x[mode]["titles"]) for x in rs),
         "titles_with_invented_words": sum(bool(v) for x in rs for v in x[mode]["invented"].values()),
         "untitled_or_abstraction_named": sum(bool(x[mode]["prefix"]) for x in rs)}
    if titled:
        real = enc([x["real"] for x in titled])
        top1 = enc([x[mode]["titles"][0] if x[mode]["titles"] else "" for x in titled])
        close = 0
        for k, x in enumerate(titled):
            ts = x[mode]["titles"][:5]
            cands = ts + [t.split(" — ", 1)[1] for t in ts if " — " in t]
            close += bool(cands) and float((enc(cands) @ real[k]).max()) >= 0.7
        perm = np.roll(np.arange(len(titled)), 1)
        m.update({"titled": len(titled), "top5_exact_hits": sum(any(hit(t, x["real"]) for t in x[mode]["titles"][:5]) for x in titled),
                  "top5_close_hits_sim_0.7": int(close), "similarity_to_real_top1": round(float((real * top1).sum(1).mean()), 3),
                  "baseline_other_painting": round(float((real * real[perm]).sum(1).mean()), 3)})
        pm = [x[mode]["pattern_margin"][0] for x in titled if x[mode]["path"] in "ab" and x[mode]["pattern_margin"] and x[mode]["pattern_margin"][0] is not None]
        if pm:
            m["first_title_closer_to_own_artist"] = f"{sum(v > 0 for v in pm)} of {len(pm)}"
    summ[mode] = m
out.write_text(json.dumps({"summary": summ, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summ, indent=1))

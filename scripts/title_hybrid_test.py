import csv, glob, io, json, os
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from db_index import FULL, load_db
from titles import enc, norm, title_cascade
from vlm_common import ROOT, SIDE, load, load_image, records

DUP = {"F1-016": "F1-017", "F1-017": "F1-016", "F1-040": "F1-046", "F1-046": "F1-040", "F1-056": "F1-062", "F1-062": "F1-056"}
SETUPS = {"adapter_checked": dict(writer=False), "hybrid": dict(writer=True)}
if os.environ.get("ARTISTIC") == "1":
    SETUPS = {"hybrid_artistic": dict(writer=True, artistic=True)}
out = ROOT / "outputs" / f"title_hybrid_test{os.environ.get('TAG', '')}.json"
R = {r["id"]: r for r in records()}
P = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "outputs" / "preds_verified2_val.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
ids = (ROOT / "outputs" / "gallery_vith_ids.txt").read_text(encoding="utf-8").split()
V = np.load(ROOT / "outputs" / "gallery_vith.npy").astype("float32")
_, E, meta, by_artist = load_db()
src = list(csv.DictReader(open(FULL / "metadata.csv", encoding="utf-8")))
shards = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--Artificio--WikiArt/snapshots/*/data/*.parquet")))
full_row = {(m["artist"], norm(m["title"])): k for k, m in enumerate(meta)}


def wiki_image(row):
    r = src[row]
    col = pq.read_table(shards[int(r["shard"])], columns=["image"]).column("image")
    im = Image.open(io.BytesIO(col[int(r["index"])].as_py()["bytes"])).convert("RGB")
    im.thumbnail((SIDE, SIDE))
    return im


items = []
for i, d in P.items():
    r = R[i]
    if r["known_title"]:
        items.append({"set": "gallery_val", "id": i, "artist": r["known_artist"].strip(), "real": r["known_title"].strip(),
                      "pred": d["pred"], "objects": d["objects"], "palette": d.get("palette"), "q": V[ids.index(i)],
                      "excl_ids": [i] + ([DUP[i]] if i in DUP else []), "excl_rows": [], "modes": ("artist_known", "artist_unknown")})
for p in json.loads((ROOT / "outputs" / "title_pattern_test_run7.json").read_text(encoding="utf-8"))["rows"]:
    row = full_row[(p["artist"], norm(p["real"]))]
    objs = [{"object": o, "status": "confirmed"} for o in p["pred"].get("objects") or [] if isinstance(o, str)]
    items.append({"set": "wikiart_held_out", "id": row, "artist": p["artist"], "real": p["real"], "pred": p["pred"], "objects": objs,
                  "palette": None, "q": E[row], "excl_ids": [], "modes": ("artist_known",),
                  "excl_rows": [k for k in by_artist[p["artist"]] if norm(meta[k]["title"]) == norm(p["real"])]})

model, proc = load(str(ROOT / "models" / "qwen_title_test_ft"))
model.eval()


@torch.no_grad()
def adapter_titles(img, artist):
    q = f"Give this artwork a title in the style of {artist}. Answer with the title only." if artist else \
        "Give this artwork a title. Answer with the title only."
    m = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": q}]}]
    text = proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=[img], return_tensors="pt").to(model.device)
    ids_ = model.generate(**inp, max_new_tokens=24, num_beams=10, num_return_sequences=10, do_sample=False)
    outs = [t.strip().strip('"“”').strip() for t in proc.batch_decode(ids_[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)]
    return list(dict.fromkeys(t for t in outs if t))


rows = json.loads(out.read_text(encoding="utf-8"))["rows"] if out.exists() else []
done = {(x["set"], x["id"]) for x in rows}
for it in items:
    if (it["set"], it["id"]) in done:
        continue
    img = load_image(ROOT / R[it["id"]]["file"], R[it["id"]]["crop"]) if it["set"] == "gallery_val" else wiki_image(it["id"])
    cache = {}

    def extra(artist):
        if artist not in cache:
            cache[artist] = adapter_titles(img, artist)
        return cache[artist]

    row = {k: it[k] for k in ("set", "id", "artist", "real")}
    for mode in it["modes"]:
        for name, s in SETUPS.items():
            res = title_cascade(model, proc, it["q"], it["pred"], it["objects"], it["palette"],
                                artist=it["artist"] if mode == "artist_known" else None, exclude_rows=it["excl_rows"],
                                exclude_ids=it["excl_ids"], use_era=False, evocative=True, top=5, extra=extra, writer=s["writer"], artistic=s.get("artistic", False))
            row[f"{name}|{mode}"] = {"titles": [t["title"] for t in res["titles"]], "sources": [t.get("source", "fallback") for t in res["titles"]],
                                     "path": res["path"], "why": res["why"], "rejected": [(x["title"], x["rejected"]) for x in res["rejected"]]}
    row["adapter_candidates"] = {str(k): v for k, v in cache.items()}
    rows.append(row)
    out.write_text(json.dumps({"rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(it["set"], it["id"], "| real:", it["real"], "|", {k: (v["titles"][:2], v["sources"][:2]) for k, v in row.items() if "|" in k}, flush=True)

summ = {}
for s in ("gallery_val", "wikiart_held_out"):
    rs = [x for x in rows if x["set"] == s]
    real = enc([x["real"] for x in rs])
    summ[s] = {}
    for k in [k for k in rs[0] if "|" in k]:
        top1 = enc([x[k]["titles"][0] for x in rs])
        exact = close = 0
        for j, x in enumerate(rs):
            ts = x[k]["titles"][:5]
            sims = enc(ts) @ real[j]
            exact += any(norm(t) == norm(x["real"]) for t in ts)
            close += bool(sims.max() >= 0.7)
        summ[s][k] = {"top5_exact_hits": exact, "top5_close_hits_sim_0.7": close,
                      "similarity_to_real_top1": round(float((real * top1).sum(1).mean()), 3),
                      "top1_source": {src_: sum(x[k]["sources"][0] == src_ for x in rs) for src_ in ("adapter", "writer", "habit", "fallback")},
                      "paths": {p: sum(x[k]["path"] == p for x in rs) for p in "abc"}}
out.write_text(json.dumps({"summary": summ, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summ, indent=1))

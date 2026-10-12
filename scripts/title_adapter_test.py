import csv, glob, io, json, os, random, re
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from peft import PeftModel
from db_index import FULL, load_db
from titles import STOP, clean_titles, color_name, enc, norm, ungrounded_words
from vlm_common import ROOT, SIDE, load, load_image, records

ADAPTERS = {"wikiart_only": ROOT / "models" / "qwen_title_test", "wikiart_then_gallery": ROOT / "models" / os.environ.get("TITLE_FT", "qwen_title_test_ft")}
out = ROOT / "outputs" / f"title_adapter_test{os.environ.get('TITLE_TAG', '')}.json"
R = {r["id"]: r for r in records()}
P = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "outputs" / "preds_verified2_val.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
pat = json.loads((ROOT / "outputs" / "title_pattern_test_run7.json").read_text(encoding="utf-8"))["rows"]
_, _, meta, by_artist = load_db()
src = list(csv.DictReader(open(FULL / "metadata.csv", encoding="utf-8")))
shards = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--Artificio--WikiArt/snapshots/*/data/*.parquet")))


def wiki_image(row):
    r = src[row]
    col = pq.read_table(shards[int(r["shard"])], columns=["image"]).column("image")
    im = Image.open(io.BytesIO(col[int(r["index"])].as_py()["bytes"])).convert("RGB")
    im.thumbnail((SIDE, SIDE))
    return im


def ground(pred, objects, palette):
    g = {w.lower() for f in ("description", "theme", "mood", "lighting", "composition", "style_guess")
         for w in re.findall(r"[A-Za-z]+", str(pred.get(f) or ""))}
    g |= {w.lower() for o in objects for w in re.findall(r"[A-Za-z]+", o)}
    g |= {w for c in (palette or [])[:6] for w in color_name(c["hex"]).split()}
    return g


items = []
for i, d in P.items():
    r = R[i]
    if r["known_title"]:
        conf = [o["object"] for o in d["objects"] if o["status"] == "confirmed"]
        items.append({"set": "gallery_val", "id": i, "artist": r["known_artist"].strip(), "real": r["known_title"].strip(),
                      "img": (ROOT / r["file"], r["crop"]), "ground": ground(d["pred"], conf, d.get("palette"))})
full_row = {(m["artist"], norm(m["title"])): k for k, m in enumerate(meta)}
for p in pat:
    p["row"] = full_row[(p["artist"], norm(p["real"]))]
    objs = [o for o in p["pred"].get("objects") or [] if isinstance(o, str)]
    items.append({"set": "wikiart_held_out", "id": p["row"], "artist": p["artist"], "real": p["real"], "img": p["row"],
                  "ground": ground(p["pred"], objs, None), "grounded_writer": p["a"][:5]})

model, proc = load(str(ADAPTERS["wikiart_only"]))
model.load_adapter(str(ADAPTERS["wikiart_then_gallery"]), adapter_name="ft")
model.eval()


@torch.no_grad()
def titles(img, artist):
    q = f"Give this artwork a title in the style of {artist}. Answer with the title only." if artist else \
        "Give this artwork a title. Answer with the title only."
    m = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": q}]}]
    text = proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=[img], return_tensors="pt").to(model.device)
    ids = model.generate(**inp, max_new_tokens=24, num_beams=8, num_return_sequences=8, do_sample=False)
    outs = [t.strip().strip('"“”').strip() for t in proc.batch_decode(ids[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)]
    return list(dict.fromkeys(t for t in outs if t))[:5]


rows = json.loads(out.read_text(encoding="utf-8"))["rows"] if out.exists() else []
done = {(x["set"], x["id"]) for x in rows}
for it in items:
    if (it["set"], it["id"]) in done:
        continue
    img = load_image(*it["img"]) if it["set"] == "gallery_val" else wiki_image(it["img"])
    row = {k: it[k] for k in ("set", "id", "artist", "real")}
    if "grounded_writer" in it:
        row["grounded_writer"] = it["grounded_writer"]
    for name, adapter in (("wikiart_only", "default"), ("wikiart_then_gallery", "ft")):
        model.set_adapter(adapter)
        for mode, art in (("artist_known", it["artist"]), ("artist_unknown", None)):
            ts = titles(img, art)
            row[f"{name}|{mode}"] = {"titles": ts, "unsupported": {t: ungrounded_words(t, it["ground"]) for t in ts}}
    rows.append(row)
    out.write_text(json.dumps({"rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(it["set"], it["id"], "| real:", it["real"], "|", row["wikiart_then_gallery|artist_known"]["titles"][:3], flush=True)

ARTISTS = sorted({x["artist"] for x in rows if x["set"] == "wikiart_held_out"})
held_rows = {x["id"] for x in rows if x["set"] == "wikiart_held_out"}
C = []
for a in ARTISTS:
    v = enc(clean_titles([meta[k] for k in by_artist[a] if k not in held_rows])).mean(0)
    C.append(v / np.linalg.norm(v))
C = np.array(C)
summ = {}
for s in ("gallery_val", "wikiart_held_out"):
    rs = [x for x in rows if x["set"] == s]
    real = enc([x["real"] for x in rs])
    keys = [k for k in rs[0] if "|" in k] + (["grounded_writer"] if s == "wikiart_held_out" else [])
    summ[s] = {"n": len(rs)}
    for k in keys:
        get = (lambda x: x[k]["titles"]) if "|" in k else (lambda x: x[k])
        top1 = enc([get(x)[0] if get(x) else "" for x in rs])
        exact = close = unsup = 0
        best = []
        for j, x in enumerate(rs):
            ts = get(x)[:5]
            sims = enc(ts) @ real[j] if ts else np.array([0.0])
            exact += any(norm(t) == norm(x["real"]) for t in ts)
            close += bool(sims.max() >= 0.7)
            best.append(float(sims.max()))
            if ts:
                unsup += bool(x[k]["unsupported"][ts[0]]) if "|" in k else 0
        e = {"top5_exact_hits": exact, "top5_close_hits_sim_0.7": close, "similarity_to_real_top1": round(float((real * top1).sum(1).mean()), 3),
             "top5_best_similarity_mean": round(float(np.mean(best)), 3)}
        if "|" in k:
            e["top1_with_unsupported_words"] = unsup
        if s == "wikiart_held_out":
            e["top1_traced_to_artist"] = sum(ARTISTS[int(np.argmax(C @ top1[j]))] == x["artist"] for j, x in enumerate(rs))
        summ[s][k] = e
    perm = np.roll(np.arange(len(rs)), 1)
    summ[s]["baseline_other_painting"] = round(float((real * real[perm]).sum(1).mean()), 3)
    if s == "wikiart_held_out":
        summ[s]["real_titles_traced_to_artist"] = sum(ARTISTS[int(np.argmax(C @ real[j]))] == x["artist"] for j, x in enumerate(rs))
out.write_text(json.dumps({"summary": summ, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summ, indent=1))

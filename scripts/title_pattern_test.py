import json, random
import numpy as np
import torch
from db_index import load_db
from titles import clean_titles, enc, title_cascade
from vlm_common import ROOT, load, load_image, messages, parse

ARTISTS = ["Piet Mondrian", "Wassily Kandinsky", "Paul Klee", "Claude Monet", "Henri de Toulouse-Lautrec",
           "Pyotr Konchalovsky", "Martiros Saryan", "Gustav Klimt"]
PER = 3
import os
EVOC = os.environ.get("EVOC", "1") == "1"
out = ROOT / "outputs" / "title_pattern_test.json"
index, E, meta, by_artist = load_db()
random.seed(7)
held = {a: random.sample(by_artist[a], PER) for a in ARTISTS}
prev = {r["row"]: r for r in json.loads(out.read_text(encoding="utf-8"))["rows"]} if out.exists() else {}
prev_keep = lambda rows: rows + [r for k, r in prev.items() if k not in {x["row"] for x in rows}]
model, proc = load(str(ROOT / "models" / "qwen_lora_v2"))
model.eval()
rows = []
for a in ARTISTS:
    for j in held[a]:
        if j in prev and prev[j].get("pred"):
            pred = prev[j]["pred"]
        else:
            img = load_image(ROOT / "artificio_data" / meta[j]["file"].replace("\\", "/"))
            text = proc.apply_chat_template(messages(img), tokenize=False, add_generation_prompt=True)
            inp = proc(text=[text], images=[img], return_tensors="pt").to(model.device)
            with torch.no_grad():
                ids = model.generate(**inp, max_new_tokens=800, do_sample=False)
            pred = parse(proc.batch_decode(ids[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]) or {}
        objs = [{"object": o, "status": "confirmed"} for o in pred.get("objects") or [] if isinstance(o, str)]
        excl = [k for k in by_artist[a] if meta[k]["title"].strip().lower() == meta[j]["title"].strip().lower()]
        ra = title_cascade(model, proc, E[j], pred, objs, None, artist=a, exclude_rows=excl, use_era=False, evocative=EVOC)
        rc = title_cascade(model, proc, E[j], pred, objs, None, exclude_rows=excl, force="c", use_era=False, evocative=EVOC)
        rows.append({"artist": a, "row": j, "real": meta[j]["title"], "description": pred.get("description"), "pred": pred,
                     "a": [t["title"] for t in ra["titles"]], "a_path": ra["path"], "c": [t["title"] for t in rc["titles"]],
                     "a_rejected": ra["rejected"], "c_rejected": rc["rejected"]})
        out.write_text(json.dumps({"summary": None, "rows": list(prev_keep(rows))}, indent=1, ensure_ascii=False), encoding="utf-8")
        print(a, "|", meta[j]["title"], "| a:", rows[-1]["a"], "| c:", rows[-1]["c"], flush=True)

cent = {}
for a in ARTISTS:
    keep = [k for k in by_artist[a] if k not in held[a]]
    v = enc(clean_titles([meta[k] for k in keep])).mean(0)
    cent[a] = v / np.linalg.norm(v)
C = np.array([cent[a] for a in ARTISTS])


def hit(t, a):
    return ARTISTS[int(np.argmax(C @ enc([t])[0]))] == a


summ = {}
for key in ("real", "a", "c"):
    ts = [(r["real"] if key == "real" else (r[key][0] if r[key] else None), r["artist"]) for r in rows]
    ts = [(t, a) for t, a in ts if t]
    summ[key] = {"n": len(ts), "artist_identified": sum(hit(t, a) for t, a in ts), "chance": round(len(ts) / len(ARTISTS), 1)}
summ["copies"] = sum(any(t.lower() == r["real"].lower() for t in r["a"] + r["c"]) for r in rows)
out.write_text(json.dumps({"summary": summ, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summ, indent=1))

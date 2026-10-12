import csv, json, random, re
from collections import Counter, defaultdict
import numpy as np
import torch
from db_index import FULL, load_db, CLIP_NAME, CLIP_PRE
from titles import STOP, clean_titles, norm, text_model
from vlm_common import ROOT

PER_ARTIST = 15
KEEP_SHARE = 0.5
out = ROOT / "labels" / "wikiart_title_train.jsonl"
_, E, meta, by_artist = load_db()
src = list(csv.DictReader(open(FULL / "metadata.csv", encoding="utf-8")))
held = set()
for f in ("title_pattern_test_run4.json",):
    p = ROOT / "outputs" / f
    if p.exists():
        held |= {(r["artist"], norm(r["real"])) for r in json.loads(p.read_text(encoding="utf-8"))["rows"]}
vocab = text_model().tokenizer.vocab
mid = Counter()
cap = Counter()
for m in meta:
    ws = re.findall(r"[A-Za-z']+", m["title"])
    for w in ws[1:]:
        lw = w.lower()
        mid[lw] += 1
        cap[lw] += w[:1].isupper()
proper = {w for w, n in mid.items() if n >= 3 and cap[w] / n >= 0.95 and w not in STOP}
HONOR = {"sir", "emperor", "king", "queen", "saint", "st", "madame", "mademoiselle", "monsieur", "mr", "mrs", "miss", "lord", "lady",
         "prince", "princess", "count", "countess", "duke", "duchess", "baron", "general", "dr", "father", "mother", "brother", "sister"}
print("words treated as names:", len(proper), sorted(proper)[:30], flush=True)
cand = []
for k, m in enumerate(meta):
    t, a = m["title"], m["artist"]
    if not clean_titles([m]) or (a, norm(t)) in held or re.search(r"\d", t):
        continue
    words = re.findall(r"[A-Za-z']+", t)
    if not 1 <= len(words) <= 8 or any(len(p) > 3 and p.lower() in t.lower() for p in a.split()):
        continue
    if not all(w.lower().strip("'") in vocab or w.lower() in STOP for w in words):
        continue
    low = [w.lower().strip("'") for w in words]
    if any(w in proper for w in low[1:]) or (low[0] in proper and len(low) > 1):
        continue
    if any(w in HONOR and k + 1 < len(words) and words[k + 1][:1].isupper() for k, w in enumerate(low)):
        continue
    cand.append(k)
print("titles passing text rules:", len(cand), "of", len(meta), flush=True)

import open_clip
model, _, _ = open_clip.create_model_and_transforms(CLIP_NAME, pretrained=CLIP_PRE, device="cuda", precision="fp16")
tok = open_clip.get_tokenizer(CLIP_NAME)
sims = []
with torch.no_grad():
    for s in range(0, len(cand), 512):
        ks = cand[s:s + 512]
        T = model.encode_text(tok([meta[k]["title"] for k in ks]).cuda()).float()
        T = (T / T.norm(dim=-1, keepdim=True)).cpu().numpy()
        sims += list((T * E[ks]).sum(1))
sims = np.array(sims)
cut = float(np.quantile(sims, 1 - KEEP_SHARE))
good = [k for k, s in zip(cand, sims) if s >= cut]
print(f"titles that match their own image (CLIP score >= {cut:.3f}):", len(good), flush=True)
random.seed(0)
per = defaultdict(list)
for k in good:
    per[meta[k]["artist"]].append(k)
rows = []
for a, ks in per.items():
    for k in random.sample(ks, min(PER_ARTIST, len(ks))):
        r = src[k]
        rows.append({"row": k, "shard": int(r["shard"]), "index": int(r["index"]), "artist": meta[k]["artist"], "title": meta[k]["title"],
                     "style": meta[k]["style"], "clip_title_image": round(float(sims[cand.index(k)]), 3)})
random.shuffle(rows)
out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
print("training pairs:", len(rows), "artists:", len(per))
lo = sorted(zip(sims, cand))[:8]
hi = sorted(zip(sims, cand))[-8:]
print("lowest-matching titles (dropped):", [meta[k]["title"] for _, k in lo])
print("best-matching titles (kept):", [meta[k]["title"] for _, k in hi])
print("examples kept:", [r["title"] for r in rows[:15]])

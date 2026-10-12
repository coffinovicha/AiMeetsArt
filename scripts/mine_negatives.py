import json, re
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer
from verify_objects import checkable, head
from vlm_common import ROOT, records

PERSON = {"man", "men", "woman", "women", "girl", "boy", "child", "children", "lady", "dancer", "saint", "queen", "king", "pharaoh",
          "goddess", "god", "angel", "musician", "couple", "mother", "soldier", "rider", "fisherman", "farmer", "monk",
          "worker", "vendor", "herder", "jester", "warrior", "baby", "nude"}
GENERIC = {"figure", "people", "person", "human", "character", "face", "portrait"}
SAME, PAIR = 0.7, 0.3

st = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
enc = lambda xs: st.encode(list(xs), normalize_embeddings=True)
gold = {r["id"]: r for r in records() if r["split"] == "train"}
preds = {json.loads(l)["id"]: json.loads(l)["pred"] or {} for l in (ROOT / "outputs" / "preds_v2_train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}

per_image, pairs = {}, Counter()
for i, r in gold.items():
    g = [o for o in r["objects"] if isinstance(o, str)]
    p = [o for o in preds.get(i, {}).get("objects") or [] if checkable(o)]
    if not g or not p:
        continue
    gh = {head(o) for o in g}
    text = {head(w) for w in re.findall(r"[a-z]+", (r["description"] + " " + (r["explanation"] or "")).lower())}
    specific = gh & PERSON
    S = enc(p) @ enc(g).T
    found = []
    for k, o in enumerate(p):
        h = head(o)
        if h in gh or h in text or h in GENERIC or float(S[k].max()) >= SAME:
            continue
        if h in PERSON and not specific:
            continue
        j = int(S[k].argmax())
        mate = g[j] if float(S[k, j]) >= PAIR else None
        found.append({"wrong": o, "true": mate, "sim": round(float(S[k, j]), 3)})
        if mate:
            pairs[(head(mate), o)] += 1
    if found:
        per_image[i] = found

conf = defaultdict(set)
for (th, o), _ in pairs.items():
    conf[th].add(o)
hard = {}
for i, r in gold.items():
    g = [o for o in r["objects"] if isinstance(o, str)]
    gh = {head(o) for o in g}
    text = {head(w) for w in re.findall(r"[a-z]+", (r["description"] + " " + (r["explanation"] or "")).lower())}
    own = [(f["true"], f["wrong"]) for f in per_image.get(i, [])]
    borrowed = [(next(o for o in g if head(o) == th), w) for th in gh & set(conf) for w in conf[th]
                if head(w) not in gh and head(w) not in text and not (head(w) in PERSON and not (gh & PERSON))]
    items = list(dict.fromkeys(own + borrowed))
    if items and g:
        G = enc(g)
        items = [(t, w) for t, w in items if float((enc([w]) @ G.T).max()) < SAME]
    if items:
        hard[i] = [{"true": t, "wrong": w} for t, w in items]

out = {"pairs": [{"true": t, "wrong": w, "count": c} for (t, w), c in pairs.most_common()], "per_image": per_image, "hard": hard}
(ROOT / "labels" / "hard_negatives.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print("images with own mistakes", len(per_image), "mistakes", sum(len(v) for v in per_image.values()))
print("images with hard negatives", len(hard), "total", sum(len(v) for v in hard.values()))
print("top pairs", [(t, w, c) for (t, w), c in pairs.most_common(25)])

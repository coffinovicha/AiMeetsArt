import json, random, re
import numpy as np
import torch
from transformers import Owlv2ForObjectDetection, Owlv2Processor
from vlm_common import ROOT

OWL = "google/owlv2-base-patch16-ensemble"
ABSTRACT = {"line", "shape", "stroke", "brushstroke", "dot", "pattern", "field", "band", "color", "colour", "texture", "form",
            "mass", "patch", "splash", "drip", "circle", "square", "triangle", "rectangle", "stripe", "grid", "ground",
            "background", "surface", "plane", "area", "block", "spot", "swirl", "ornament", "mark", "scribble", "glaze",
            "reflection", "glare", "frame", "texture", "composition", "light", "shadow", "highlight", "layer", "wash"}
TEMPLATES = {"plain": "{}", "painting": "a painting of {}"}
CAL = ROOT / "outputs" / "verifier_calib.json"


def head(p):
    w = re.findall(r"[a-z]+", p.lower())
    if not w:
        return ""
    h = w[-1]
    return h[:-3] + "y" if h.endswith("ies") else h[:-2] if h.endswith(("ches", "shes", "sses", "xes")) else h[:-1] if h.endswith("s") and not h.endswith("ss") else h


def checkable(p):
    return isinstance(p, str) and head(p) and head(p) not in ABSTRACT


class Verifier:
    def __init__(self, template=None, threshold=None):
        self.proc = Owlv2Processor.from_pretrained(OWL)
        self.model = Owlv2ForObjectDetection.from_pretrained(OWL, dtype=torch.float16).to("cuda").eval()
        cal = json.loads(CAL.read_text(encoding="utf-8")) if CAL.exists() else {}
        self.template = template or cal.get("template", "plain")
        self.threshold = threshold if threshold is not None else cal.get("threshold", 0.1)

    @torch.no_grad()
    def scores(self, im, phrases, template=None):
        if not phrases:
            return {}
        t = TEMPLATES[template or self.template]
        inp = self.proc(text=[[t.format(p) for p in phrases]], images=im, return_tensors="pt").to("cuda")
        inp["pixel_values"] = inp["pixel_values"].half()
        out = self.model(**inp)
        s = torch.sigmoid(out.logits[0].float()).max(0).values.cpu().numpy()
        return {p: float(v) for p, v in zip(phrases, s)}

    def check(self, im, objects):
        objs = [o for o in objects or [] if isinstance(o, str)]
        sc = self.scores(im, [o for o in objs if checkable(o)])
        return {o: {"score": sc.get(o), "kept": (o not in sc) or sc[o] >= self.threshold} for o in objs}


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


if __name__ == "__main__":
    from collections import Counter
    from vlm_common import load_image, records
    random.seed(0)
    tr = [r for r in records() if r["split"] == "train"]
    freq = Counter(head(o) for r in tr for o in r["objects"] if checkable(o))
    vocab = {}
    for r in tr:
        for o in r["objects"]:
            if checkable(o):
                vocab.setdefault(head(o), o)
    common = [h for h, _ in freq.most_common(80)]
    v = Verifier(template="plain", threshold=0.0)
    res = {t: {"pos": [], "neg": []} for t in TEMPLATES}
    examples = []
    for r in tr:
        im = load_image(ROOT / r["file"], r["crop"], side=960)
        pos = [o for o in r["objects"] if checkable(o)]
        gold_h = {head(o) for o in r["objects"]}
        text = " ".join(re.findall(r"[a-z]+", (r["description"] + " " + r["explanation"]).lower()))
        cand = [vocab[h] for h in common if h not in gold_h and h not in {head(w) for w in text.split()}]
        neg = random.sample(cand, min(6, len(cand)))
        for t in TEMPLATES:
            sc = v.scores(im, pos + neg, template=t)
            res[t]["pos"] += [sc[p] for p in pos]
            res[t]["neg"] += [sc[n] for n in neg]
            if t == "plain":
                examples.append({"id": r["id"], "pos": {p: round(sc[p], 3) for p in pos}, "neg": {n: round(sc[n], 3) for n in neg}})
        print(r["id"], len(pos), len(neg), flush=True)
    summ = {}
    for t, d in res.items():
        p, n = np.array(d["pos"]), np.array(d["neg"])
        thr = float(np.quantile(p, 0.10))
        summ[t] = {"n_pos": len(p), "n_neg": len(n), "auc": auc(p, n), "threshold_10pct_pos_rejected": thr,
                   "neg_rejected_at_thr": float((n < thr).mean()), "pos_median": float(np.median(p)), "neg_median": float(np.median(n))}
    best = max(summ, key=lambda t: summ[t]["auc"])
    out = {"template": best, "threshold": summ[best]["threshold_10pct_pos_rejected"], "by_template": summ}
    CAL.write_text(json.dumps(out, indent=1), encoding="utf-8")
    (ROOT / "outputs" / "verifier_calib_examples.json").write_text(json.dumps(examples, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1))

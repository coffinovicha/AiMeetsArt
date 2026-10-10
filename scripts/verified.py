import json, re, time
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from autocrop import crop_processor, find_crop
from palette import palette
from verify_objects import Verifier, checkable, head
from vlm_common import PROMPT, SIDE, load, parse

UNCLEAR = 0.55
MATCH = 0.7
CONTEXT = re.compile(r"\b(typical of|common in|popular in|often|were (?:used|popular|made|common)|"
                     r"\d+(?:st|nd|rd|th) centur|centuries|tradition|cultures?|known as|symbol of|symboli[sz]es|"
                     r"Byzantine|Renaissance|Baroque|Gothic|Ottoman|Egyptian|African|Asian|Jewish|Christian|Orthodox)\b", re.I)


def sub_boxes(b):
    x0, y0, x1, y1 = b
    w, h = x1 - x0, y1 - y0
    return [b, [x0, y0 + 0.04 * h, x1 - 0.04 * w, y1], [x0 + 0.04 * w, y0, x1, y1 - 0.04 * h]]


def sentences(t):
    return [s for s in re.split(r"(?<=[.!?])\s+", (t or "").strip()) if s]


def mentions(sentence, obj):
    h = head(obj)
    words = re.findall(r"[a-z]+", sentence.lower())
    return obj.lower() in sentence.lower() or any(head(w) == h or (len(h) >= 4 and w.startswith(h)) for w in words)


def mark(sentence, obj, tag):
    m = re.search(rf"\b{re.escape(obj)}\b", sentence, re.I)
    if not m:
        h = head(obj)
        for w in re.finditer(r"[A-Za-z]+(?:-[A-Za-z]+)*", sentence):
            x = w.group(0).lower()
            if head(x) == h or (len(h) >= 4 and x.startswith(h)):
                m = w
                break
    if not m:
        return sentence
    e = m.end()
    tail = re.match(r"(?:-[A-Za-z]+)+", sentence[e:])
    e += tail.end() if tail else 0
    return sentence[:e] + f" [{tag}]" + sentence[e:]


def render(P, report, lowest, drop=False):
    removed = [r["object"] for r in report if r["status"] == "removed"]
    unconf = [r["object"] for r in report if r["status"] == "unconfirmed"]
    out, dropped, context = {}, [], []
    for field in ("description", "explanation"):
        keep, marked = [], []
        for s in sentences(P.get(field)):
            hit = [o for o in removed if mentions(s, o)]
            if hit and drop:
                dropped.append({"field": field, "sentence": s})
                continue
            keep.append(s)
            ms = s
            for o in hit:
                ms = mark(ms, o, "likely not present")
            for o in unconf:
                if mentions(ms, o):
                    ms = mark(ms, o, "unconfirmed")
            if field == "explanation" and CONTEXT.search(s):
                context.append(s)
                ms = ms + " [context, not verified from the image]"
            marked.append(ms)
        out[field] = " ".join(keep)
        out[field + "_marked"] = " ".join(marked)
    out["subject_unclear"] = lowest < UNCLEAR
    if out["subject_unclear"]:
        out["description_marked"] = "[Subject unclear: the three readings of this photo disagree.] " + out["description_marked"]
    out["dropped"], out["context"] = dropped, context
    return out


class VerifiedAnalyzer:
    def __init__(self, adapter):
        self.model, self.proc = load(adapter)
        self.model.eval()
        self.proc.tokenizer.padding_side = "left"
        self.cproc = crop_processor()
        self.ver = Verifier()
        self.st = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")

    def generate(self, ims):
        m = [[{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": PROMPT}]}] for im in ims]
        texts = [self.proc.apply_chat_template(x, tokenize=False, add_generation_prompt=True) for x in m]
        inp = self.proc(text=texts, images=ims, padding=True, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            ids = self.model.generate(**inp, max_new_tokens=800, do_sample=False)
        raws = self.proc.batch_decode(ids[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        return raws, [parse(r) for r in raws]

    def enc(self, xs):
        return self.st.encode([str(x or "") for x in xs], normalize_embeddings=True)

    def analyze(self, photo, crop=None):
        t0 = time.time()
        box = crop if crop is not None else find_crop(self.model, self.cproc, photo)[0]
        W, H = photo.size
        boxes = sub_boxes(box)
        pcs = [photo.crop((int(b[0] * W), int(b[1] * H), int(b[2] * W), int(b[3] * H))) for b in boxes]
        ims = []
        for p in pcs:
            q = p.copy()
            q.thumbnail((SIDE, SIDE))
            ims.append(q)
        raws, preds = self.generate(ims)
        ok = [i for i, p in enumerate(preds) if p and p.get("description")]
        if not ok:
            return {"crop": box, "raw": raws[0], "pred": None, "seconds": round(time.time() - t0, 1)}
        D = self.enc([preds[i]["description"] for i in ok])
        S = D @ D.T
        n = len(ok)
        agreement = float((S.sum() - n) / (n * (n - 1))) if n > 1 else 1.0
        lowest = float(S[~np.eye(n, dtype=bool)].min()) if n > 1 else 1.0
        primary = ok[int(np.argmax(S.sum(1)))]
        P = dict(preds[primary])
        objs = [o for o in P.get("objects") or [] if isinstance(o, str)]
        others = [[o for o in preds[i].get("objects") or [] if isinstance(o, str)] for i in ok if i != primary]
        big = pcs[0].copy()
        big.thumbnail((960, 960))
        det = self.ver.check(big, objs)
        report = []
        if objs:
            E = self.enc(objs)
            for k, o in enumerate(objs):
                sup = 1
                for oth in others:
                    if not oth:
                        continue
                    sims = E[k] @ self.enc(oth).T
                    if float(sims.max()) >= MATCH or any(head(x) == head(o) for x in oth):
                        sup += 1
                agree = sup >= 2
                d = det[o]
                dv = None if d["score"] is None else d["kept"]
                if agree and dv is not False:
                    status = "confirmed"
                elif not agree and dv is False:
                    status = "removed"
                else:
                    status = "unconfirmed"
                report.append({"object": o, "support": sup, "detector": d["score"], "status": status})
        out = render(P, report, lowest)
        P["description"], P["explanation"] = out["description"], out["explanation"]
        P["objects"] = [r["object"] for r in report if r["status"] != "removed"]
        return {"crop": box, "boxes": boxes, "raw": raws[primary], "pred": P, "readings": preds, "primary": primary,
                "agreement": round(agreement, 3), "lowest_pair": round(lowest, 3), "subject_unclear": out["subject_unclear"], "objects": report,
                "description_marked": out["description_marked"], "explanation_marked": out["explanation_marked"],
                "dropped": out["dropped"], "context": out["context"], "palette": palette(pcs[0]), "seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":
    import argparse
    from autocrop import open_photo
    from vlm_common import ROOT, records
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(ROOT / "models" / "qwen_lora_expl"))
    ap.add_argument("--tag", default="verified")
    ap.add_argument("--split", default="val")
    ap.add_argument("--ids", default="")
    a = ap.parse_args()
    out = ROOT / "outputs" / f"preds_{a.tag}_{a.split}.jsonl"
    done = {json.loads(l)["id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()} if out.exists() else set()
    ac = ROOT / "outputs" / "autocrop.jsonl"
    auto = {json.loads(l)["id"]: json.loads(l)["auto"] for l in ac.read_text(encoding="utf-8").splitlines() if l.strip()} if ac.exists() else {}
    want = set(a.ids.split(",")) if a.ids else None
    recs = [r for r in records() if r["split"] == a.split and r["id"] not in done and (want is None or r["id"] in want)]
    an = VerifiedAnalyzer(a.adapter)
    for r in recs:
        res = an.analyze(open_photo(ROOT / r["file"]), crop=auto.get(r["id"]))
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"id": r["id"], **res}, ensure_ascii=False) + "\n")
        st = {}
        for o in res.get("objects", []):
            st[o["status"]] = st.get(o["status"], 0) + 1
        print(r["id"], f"{res['seconds']}s agree {res.get('agreement')} min {res.get('lowest_pair')} unclear {res.get('subject_unclear')} {st} dropped {len(res.get('dropped', []))}", flush=True)
    print("peak GB", round(torch.cuda.max_memory_allocated() / 1e9, 2))

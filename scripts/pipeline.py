import json, os, time
from pathlib import Path
import torch
from autocrop import crop_processor, find_crop
from palette import palette
from verify_objects import Verifier, head
from vlm_common import PROMPT, SIDE, load, parse


def variants(phrase):
    p = phrase.strip().lower()
    h = head(p)
    forms = {p, h, h + "s", h + "es"}
    out = set()
    for f in forms:
        if f:
            out |= {f, " " + f, f.capitalize(), " " + f.capitalize()}
    return out


class Analyzer:
    def __init__(self, adapter, rounds=2):
        meta = json.loads((Path(adapter) / "vlm_meta.json").read_text(encoding="utf-8"))
        assert meta["side"] == SIDE, f"adapter trained at {meta['side']} px, VLM_SIDE is {SIDE}"
        self.model, self.proc = load(adapter)
        self.model.eval()
        self.cproc = crop_processor()
        self.ver = Verifier()
        self.rounds = rounds

    def bad_ids(self, words):
        tok = self.proc.tokenizer
        ids = {tuple(tok(v, add_special_tokens=False).input_ids) for w in words for v in variants(w)}
        return [list(i) for i in ids if i]

    def generate(self, im, extra="", banned=()):
        m = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": PROMPT + extra}]}]
        text = self.proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        inp = self.proc(text=[text], images=[im], return_tensors="pt").to(self.model.device)
        kw = {"bad_words_ids": self.bad_ids(banned)} if banned else {}
        with torch.no_grad():
            ids = self.model.generate(**inp, max_new_tokens=800, do_sample=False, **kw)
        raw = self.proc.batch_decode(ids[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return raw, parse(raw)

    def analyze(self, photo, crop=None):
        t0 = time.time()
        box = crop if crop is not None else find_crop(self.model, self.cproc, photo)[0]
        W, H = photo.size
        pc = photo.crop((int(box[0] * W), int(box[1] * H), int(box[2] * W), int(box[3] * H)))
        im, big = pc.copy(), pc.copy()
        im.thumbnail((SIDE, SIDE))
        big.thumbnail((960, 960))
        raw, pred = self.generate(im)
        raw0, pred0 = raw, pred
        rounds, banned = [], set()
        for r in range(self.rounds + 1):
            if not pred:
                break
            chk = self.ver.check(big, pred.get("objects"))
            rej = [o for o, d in chk.items() if not d["kept"]]
            rounds.append({"round": r, "objects": chk})
            if not rej or r == self.rounds:
                break
            banned |= set(rej)
            kept = [o for o, d in chk.items() if d["kept"]]
            extra = (f" Objects confirmed in the image: {', '.join(kept) or 'none'}. "
                     f"These are NOT in the image, do not mention them: {', '.join(sorted(banned))}.")
            raw2, pred2 = self.generate(im, extra, banned)
            if pred2:
                raw, pred = raw2, pred2
        removed = []
        if pred and rounds:
            last = rounds[-1]["objects"]
            removed = [o for o, d in last.items() if not d["kept"]]
            pred["objects"] = [o for o in pred.get("objects") or [] if last.get(o, {"kept": True})["kept"]]
        return {"crop": box, "raw": raw, "pred": pred, "raw0": raw0, "pred0": pred0, "banned": sorted(banned),
                "removed": removed, "verification": rounds, "palette": palette(pc), "seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":
    import argparse
    from autocrop import open_photo
    from vlm_common import ROOT, records
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(ROOT / "models" / "qwen_lora_expl"))
    ap.add_argument("--tag", default="pipe")
    ap.add_argument("--split", default="val")
    ap.add_argument("--ids", default="")
    a = ap.parse_args()
    out = ROOT / "outputs" / f"preds_{a.tag}_{a.split}.jsonl"
    done = {json.loads(l)["id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()} if out.exists() else set()
    ac = ROOT / "outputs" / "autocrop.jsonl"
    auto = {json.loads(l)["id"]: json.loads(l)["auto"] for l in ac.read_text(encoding="utf-8").splitlines() if l.strip()} if ac.exists() else {}
    want = set(a.ids.split(",")) if a.ids else None
    recs = [r for r in records() if r["split"] == a.split and r["id"] not in done and (want is None or r["id"] in want)]
    an = Analyzer(a.adapter)
    for r in recs:
        res = an.analyze(open_photo(ROOT / r["file"]), crop=auto.get(r["id"]))
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"id": r["id"], **res}, ensure_ascii=False) + "\n")
        print(r["id"], f"{res['seconds']}s", "banned:", res["banned"], "removed:", res["removed"], flush=True)
    print("peak GB", round(torch.cuda.max_memory_allocated() / 1e9, 2))

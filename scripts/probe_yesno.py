import argparse, json, re
import torch
from vlm_common import ROOT, load, load_image, records

PROBES = {
    "F1-040": {"pomegranates": "yes", "apples": "no", "jug": "yes", "bottle": "yes"},
    "F1-046": {"pomegranates": "yes", "apples": "no", "jug": "yes", "bottle": "yes"},
    "F1-084": {"men": "yes", "women": "no", "backgammon board": "yes", "stringed instrument": "no"},
    "F3-073": {"lance": "yes", "dragon": "yes", "horse": "yes", "cat": "no"},
    "F3-041": {"chairs": "yes", "tables": "no", "balls": "yes", "window": "no"},
    "F2-059": {"children": "yes", "geese": "yes", "cow": "yes", "man": "no"},
    "F3-060": {"winged goddess": "yes", "throne": "yes", "man": "no", "hieroglyphs": "yes"},
    "F1-015": {"chairs": "yes", "tables": "yes", "buildings": "no", "train": "no"},
    "F1-036": {"women": "yes", "child": "no", "hair": "yes", "cap": "no"},
    "F1-020": {"masks": "yes", "flowers": "yes", "elephant": "no", "figures": "yes"},
    "F3-074": {"angels": "yes", "swords": "yes", "lute": "no", "child": "yes"},
    "F3-024": {"birds": "yes", "sun": "yes", "rabbit": "no", "cat": "no"},
}
Q = "For each item, answer yes if it is visible in this artwork and no if it is not: {}."

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True)
ap.add_argument("--tag", required=True)
a = ap.parse_args()
R = {r["id"]: r for r in records()}
model, proc = load(a.adapter)
model.eval()
res, ok, tot = {}, 0, 0
look = {"apples", "women", "tables", "buildings", "man", "child", "stringed instrument", "lute", "elephant", "rabbit", "cat", "train", "cap", "window"}
lok = ltot = 0
for i, items in PROBES.items():
    img = load_image(ROOT / R[i]["file"], R[i]["crop"])
    m = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": Q.format(", ".join(items))}]}]
    text = proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=[img], return_tensors="pt").to(model.device)
    with torch.no_grad():
        ids = model.generate(**inp, max_new_tokens=60, do_sample=False)
    raw = proc.batch_decode(ids[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    ans = {k.strip().lower(): v.lower() for k, v in re.findall(r"([^;:]+):\s*(yes|no)", raw, re.I)}
    row = {}
    for obj, truth in items.items():
        got = ans.get(obj.lower())
        row[obj] = {"truth": truth, "got": got, "ok": got == truth}
        tot += 1
        ok += got == truth
        if truth == "no" and obj in look:
            ltot += 1
            lok += got == truth
    res[i] = {"raw": raw, "answers": row}
    print(i, raw, flush=True)
summ = {"accuracy": ok / tot, "n": tot, "lookalike_no_correct": lok, "lookalike_no_total": ltot}
(ROOT / "outputs" / f"probe_{a.tag}.json").write_text(json.dumps({"summary": summ, "items": res}, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summ))

import json, re, time
import torch
from PIL import Image, ImageOps
from transformers import AutoProcessor
from vlm_common import BASE, ROOT

CROP_SIDE = 768
CROP_PROMPT = ("Locate the painted artwork itself in this photo: only the painted surface, excluding the frame, mat, "
               "wall, glass reflections outside it and any label. Output its bbox coordinates in JSON format.")


def crop_processor():
    return AutoProcessor.from_pretrained(BASE, min_pixels=256 * 28 * 28, max_pixels=CROP_SIDE * CROP_SIDE)


def find_crop(model, proc, im):
    m = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": CROP_PROMPT}]}]
    text = proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=[im], return_tensors="pt").to(model.device)
    ctx = model.disable_adapter() if hasattr(model, "disable_adapter") else torch.no_grad()
    with ctx, torch.no_grad():
        ids = model.generate(**inp, max_new_tokens=80, do_sample=False)
    raw = proc.batch_decode(ids[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    _, gh, gw = inp["image_grid_thw"][0].tolist()
    H, W = gh * 14, gw * 14
    n = re.search(r"\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]", raw)
    if not n:
        return [0.0, 0.0, 1.0, 1.0], raw
    x0, y0, x1, y1 = map(float, n.groups())
    b = [max(0.0, min(x0, x1) / W), max(0.0, min(y0, y1) / H), min(1.0, max(x0, x1) / W), min(1.0, max(y0, y1) / H)]
    if (b[2] - b[0]) * (b[3] - b[1]) < 0.05:
        return [0.0, 0.0, 1.0, 1.0], raw
    return b, raw


def open_photo(path):
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u else 0.0


if __name__ == "__main__":
    import numpy as np
    from vlm_common import load, records
    out = ROOT / "outputs" / "autocrop.jsonl"
    done = {json.loads(l)["id"]: json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()} if out.exists() else {}
    recs = records()
    model, _ = load(None)
    model.eval()
    proc = crop_processor()
    for r in recs:
        if r["id"] in done:
            continue
        t = time.time()
        im = open_photo(ROOT / r["file"])
        box, raw = find_crop(model, proc, im)
        hand = r["crop"] or [0.0, 0.0, 1.0, 1.0]
        row = {"id": r["id"], "split": r["split"], "auto": box, "hand": hand, "iou": iou(box, hand), "iou_nocrop": iou([0, 0, 1, 1], hand), "raw": raw}
        done[r["id"]] = row
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(r["id"], f"{time.time() - t:.1f}s iou {row['iou']:.3f}", flush=True)
    v = list(done.values())
    s = {"n": len(v), "iou_mean": float(np.mean([x["iou"] for x in v])), "iou_median": float(np.median([x["iou"] for x in v])),
         "iou_ge_0.9": sum(x["iou"] >= 0.9 for x in v), "iou_ge_0.8": sum(x["iou"] >= 0.8 for x in v),
         "nocrop_iou_mean": float(np.mean([x["iou_nocrop"] for x in v])), "hand_crop_needed": sum(x["iou_nocrop"] < 0.95 for x in v)}
    (ROOT / "outputs" / "autocrop_eval.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    print(json.dumps(s, indent=1))

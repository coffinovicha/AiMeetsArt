import argparse
import torch
from vlm_common import ROOT, load, load_image, messages, records, target

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True)
ap.add_argument("--split", default="val")
a = ap.parse_args()
model, proc = load(a.adapter)
model.eval()
tot, recs = 0.0, [r for r in records() if r["split"] == a.split]
with torch.no_grad():
    for r in recs:
        img = load_image(ROOT / r["file"], r["crop"])
        full = proc.apply_chat_template(messages(img, target(r)), tokenize=False)
        prompt = proc.apply_chat_template(messages(img), tokenize=False, add_generation_prompt=True)
        enc = proc(text=[full], images=[img], return_tensors="pt").to(model.device)
        plen = proc(text=[prompt], images=[img], return_tensors="pt")["input_ids"].shape[1]
        labels = enc["input_ids"].clone()
        labels[:, :plen] = -100
        tot += model(**enc, labels=labels).loss.item()
print(a.adapter, "val loss", round(tot / len(recs), 4))

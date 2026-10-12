import argparse, json, time
import torch
from vlm_common import ROOT, load, load_image, messages, parse, records

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--split", default="train")
ap.add_argument("--batch", type=int, default=4)
a = ap.parse_args()

out = ROOT / "outputs" / f"preds_{a.tag}_{a.split}.jsonl"
done = {json.loads(l)["id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()} if out.exists() else set()
recs = [r for r in records() if r["split"] == a.split and r["id"] not in done]
model, proc = load(a.adapter)
model.eval()
proc.tokenizer.padding_side = "left"
for s in range(0, len(recs), a.batch):
    t = time.time()
    chunk = recs[s:s + a.batch]
    ims = [load_image(ROOT / r["file"], r["crop"]) for r in chunk]
    texts = [proc.apply_chat_template(messages(im), tokenize=False, add_generation_prompt=True) for im in ims]
    inp = proc(text=texts, images=ims, padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        ids = model.generate(**inp, max_new_tokens=800, do_sample=False)
    raws = proc.batch_decode(ids[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
    with out.open("a", encoding="utf-8") as f:
        for r, raw in zip(chunk, raws):
            f.write(json.dumps({"id": r["id"], "raw": raw, "pred": parse(raw)}, ensure_ascii=False) + "\n")
    print(",".join(r["id"] for r in chunk), f"{time.time() - t:.1f}s", sum(parse(x) is not None for x in raws), "ok", flush=True)
print("peak GB", round(torch.cuda.max_memory_allocated() / 1e9, 2))

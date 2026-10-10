import argparse, json, time
import torch
from vlm_common import ROOT, load, load_image, messages, parse, records

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", default=None)
ap.add_argument("--tag", required=True)
ap.add_argument("--split", default="val")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

out = ROOT / "outputs" / f"preds_{a.tag}_{a.split}.jsonl"
done = {json.loads(l)["id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()} if out.exists() else set()
recs = [r for r in records() if r["split"] == a.split and r["id"] not in done]
if a.limit:
    recs = recs[:a.limit]
model, proc = load(a.adapter)
model.eval()
for r in recs:
    t = time.time()
    img = load_image(ROOT / r["file"], r["crop"])
    m = messages(img)
    text = proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=[img], return_tensors="pt").to(model.device)
    with torch.no_grad():
        ids = model.generate(**inp, max_new_tokens=800, do_sample=False)
    raw = proc.batch_decode(ids[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"id": r["id"], "raw": raw, "pred": parse(raw)}, ensure_ascii=False) + "\n")
    print(r["id"], f"{time.time() - t:.1f}s", "ok" if parse(raw) else "PARSE FAIL", flush=True)
print("peak GB", round(torch.cuda.max_memory_allocated() / 1e9, 2))

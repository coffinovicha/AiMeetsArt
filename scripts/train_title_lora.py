import argparse, io, json, math, random, time
import pyarrow.parquet as pq
import torch
from PIL import Image
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from vlm_common import ROOT, SIDE, load, load_image, records

ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=["wikiart", "gallery"], required=True)
ap.add_argument("--init", default="")
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=12000)
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--rank", type=int, default=16)
a = ap.parse_args()
random.seed(0)
torch.manual_seed(0)
IMG = ROOT / "artificio_data" / "full" / "title_images"
IMG.mkdir(parents=True, exist_ok=True)


def prompt(artist):
    return f"Give this artwork a title in the style of {artist}. Answer with the title only." if artist else \
        "Give this artwork a title. Answer with the title only."


if a.stage == "wikiart":
    rows = [json.loads(l) for l in (ROOT / "labels" / "wikiart_title_train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()][:a.n]
    need = {}
    for r in rows:
        if not (IMG / f"{r['row']}.jpg").exists():
            need.setdefault(r["shard"], []).append(r)
    import glob, os
    shards = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--Artificio--WikiArt/snapshots/*/data/*.parquet")))
    for s, rs in need.items():
        col = pq.read_table(shards[s], columns=["image"]).column("image")
        for r in rs:
            im = Image.open(io.BytesIO(col[r["index"]].as_py()["bytes"])).convert("RGB")
            im.thumbnail((SIDE, SIDE))
            im.save(IMG / f"{r['row']}.jpg", quality=90)
        print("cached images from shard", s, len(rs), flush=True)
    items = [(IMG / f"{r['row']}.jpg", None, r["artist"], r["title"]) for r in rows]
    val, train = items[:300], items[300:]
else:
    recs = [r for r in records() if r["known_title"] and r["known_artist"]]
    train = [(ROOT / r["file"], r["crop"], r["known_artist"].strip(), r["known_title"].strip()) for r in recs if r["split"] == "train"]
    val = [(ROOT / r["file"], r["crop"], r["known_artist"].strip(), r["known_title"].strip()) for r in recs if r["split"] == "val"]

model, proc = load()
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
if a.init:
    model = PeftModel.from_pretrained(model, a.init, is_trainable=True)
else:
    cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                     target_modules=r".*model\.language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
    model = get_peft_model(model, cfg)
model.print_trainable_parameters()


def encode(item, with_artist):
    path, crop, artist, title = item
    img = load_image(path, crop)
    m = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt(artist if with_artist else None)}]}]
    full = proc.apply_chat_template(m + [{"role": "assistant", "content": [{"type": "text", "text": title}]}], tokenize=False)
    pre = proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
    enc = proc(text=[full], images=[img], return_tensors="pt")
    plen = proc(text=[pre], images=[img], return_tensors="pt")["input_ids"].shape[1]
    labels = enc["input_ids"].clone()
    labels[:, :plen] = -100
    enc["labels"] = labels
    return enc


vcache = [encode(x, True) for x in val]


def val_loss():
    model.eval()
    tot = 0.0
    with torch.no_grad():
        for e in vcache:
            tot += model(**{k: v.to(model.device) for k, v in e.items()}).loss.item()
    model.train()
    return tot / len(vcache)


opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
steps = math.ceil(len(train) * a.epochs / a.accum)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / steps))
log, best, step = [], float("inf"), 0
v0 = val_loss()
print("val loss before", round(v0, 4), flush=True)
model.train()
for ep in range(a.epochs):
    order = train[:]
    random.shuffle(order)
    t0, run = time.time(), 0.0
    for i, item in enumerate(order):
        e = encode(item, random.random() < 0.7)
        loss = model(**{k: v.to(model.device) for k, v in e.items()}).loss / a.accum
        loss.backward()
        run += loss.item()
        if (i + 1) % a.accum == 0 or i == len(order) - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 25 == 0 or i == len(order) - 1:
                print(f"epoch {ep + 1}/{a.epochs} step {step}/{steps} item {i + 1}/{len(order)} loss {run * a.accum / (i + 1):.3f} "
                      f"{time.time() - t0:.0f}s gpu {torch.cuda.max_memory_allocated() / 1e9:.1f}GB", flush=True)
    vl = val_loss()
    log.append({"epoch": ep + 1, "train_loss": run * a.accum / len(order), "val_loss": vl, "sec": time.time() - t0})
    print(log[-1], flush=True)
    if vl < best:
        best = vl
        model.save_pretrained(ROOT / a.out)
(ROOT / a.out).mkdir(parents=True, exist_ok=True)
(ROOT / a.out / "train_log.json").write_text(json.dumps({"val_loss_before": v0, "epochs": log}, indent=1), encoding="utf-8")
(ROOT / a.out / "vlm_meta.json").write_text(json.dumps({"side": SIDE, "task": "title", "stage": a.stage, "init": a.init,
                                                        "n_train": len(train), "status": "test model, not used by the app"}), encoding="utf-8")
print("best val", round(best, 4))

import argparse, json, math, random, time
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from vlm_common import ROOT, load, load_image, messages, records, target

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--out", default="models/qwen_lora")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
torch.manual_seed(a.seed)
random.seed(a.seed)

model, proc = load()
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                 target_modules=r".*model\.language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
model = get_peft_model(model, cfg)
model.print_trainable_parameters()

recs = records()
train = [r for r in recs if r["split"] == "train"]
val = [r for r in recs if r["split"] == "val"]


def encode(r):
    img = load_image(ROOT / r["file"], r["crop"])
    full = proc.apply_chat_template(messages(img, target(r)), tokenize=False)
    prompt = proc.apply_chat_template(messages(img), tokenize=False, add_generation_prompt=True)
    enc = proc(text=[full], images=[img], return_tensors="pt")
    plen = proc(text=[prompt], images=[img], return_tensors="pt")["input_ids"].shape[1]
    labels = enc["input_ids"].clone()
    labels[:, :plen] = -100
    enc["labels"] = labels
    return enc


cache = {r["id"]: encode(r) for r in train + val}
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
steps = math.ceil(len(train) * a.epochs / a.accum)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 3) * max(0.0, 1 - s / steps))


def val_loss():
    model.eval()
    tot = 0.0
    with torch.no_grad():
        for r in val:
            b = {k: v.to(model.device) for k, v in cache[r["id"]].items()}
            tot += model(**b).loss.item()
    model.train()
    return tot / len(val)


log = []
print("val loss before", round(val_loss(), 4), flush=True)
model.train()
step, best = 0, float("inf")
for ep in range(a.epochs):
    order = train[:]
    random.shuffle(order)
    t, run = time.time(), 0.0
    for i, r in enumerate(order):
        b = {k: v.to(model.device) for k, v in cache[r["id"]].items()}
        loss = model(**b).loss / a.accum
        loss.backward()
        run += loss.item()
        if (i + 1) % a.accum == 0 or i == len(order) - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
    vl = val_loss()
    log.append({"epoch": ep + 1, "train_loss": run * a.accum / len(order), "val_loss": vl, "sec": time.time() - t})
    print(log[-1], "peak GB", round(torch.cuda.max_memory_allocated() / 1e9, 2), flush=True)
    if vl < best:
        best = vl
        model.save_pretrained(ROOT / a.out)
(ROOT / a.out).mkdir(parents=True, exist_ok=True)
(ROOT / a.out / "train_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
print("best val", round(best, 4))

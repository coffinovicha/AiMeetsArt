import argparse, io, json, math, random, re, time
from collections import Counter
import torch
from PIL import Image, ImageEnhance, ImageOps
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from verify_objects import checkable, head
from vlm_common import ROOT, SCHEMA, SIDE, load, load_image, messages, records, target

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--out", default="models/qwen_lora")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--qa", action="store_true")
ap.add_argument("--aug", action="store_true")
ap.add_argument("--hard", default="")
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


QA_PROMPT = "For each item, answer yes if it is visible in this artwork and no if it is not: {}."
freq = Counter(head(o) for r in train for o in r["objects"] if checkable(o))
vocab = {}
for r in train:
    for o in r["objects"]:
        if checkable(o):
            vocab.setdefault(head(o), o)
PERSON = {"man", "men", "woman", "women", "girl", "boy", "child", "children", "lady", "dancer", "saint", "queen", "king", "pharaoh",
          "goddess", "god", "angel", "musician", "couple", "family", "mother", "soldier", "rider", "fisherman", "farmer", "monk",
          "worker", "vendor", "herder", "figure", "people", "person", "face", "portrait", "nude", "baby", "jester", "warrior"}
GENERIC = {"figure", "people", "person", "human", "character", "animal", "creature", "plant", "object", "shape", "form", "scene", "thing"}
common = [h for h, _ in freq.most_common(90) if h not in GENERIC][:80]
NEG = {}
if a.qa:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
    V = dict(zip(common, st.encode([vocab[h] for h in common], normalize_embeddings=True)))
    for r in train:
        gold = [o for o in r["objects"] if isinstance(o, str)]
        G = st.encode(gold, normalize_embeddings=True) if gold else np.zeros((0, 768))
        gh = {head(o) for o in gold}
        text = {head(w) for w in re.findall(r"[a-z]+", (r["description"] + " " + (r["explanation"] or "")).lower())}
        human = bool((gh | text) & PERSON)
        NEG[r["id"]] = [vocab[h] for h in common if h not in gh and h not in text and not (human and h in PERSON)
                        and (len(G) == 0 or float((G @ V[h]).max()) < 0.5)]
    del st


HARD = json.loads(open(a.hard, encoding="utf-8").read())["hard"] if a.hard else {}


def qa_sample(r):
    pos = [o for o in r["objects"] if checkable(o)]
    cand = NEG[r["id"]]
    hard = random.sample(HARD.get(r["id"], []), min(2, len(HARD.get(r["id"], []))))
    p = list(dict.fromkeys(h["true"] for h in hard if h["true"]))
    p += random.sample([o for o in pos if o not in p], min(max(0, 2 - len(p)), len([o for o in pos if o not in p])))
    hn = list(dict.fromkeys(h["wrong"] for h in hard))
    n = hn + random.sample([c for c in cand if c not in hn], min(max(0, 4 - len(p) - len(hn)), len([c for c in cand if c not in hn])))
    items = [(o, "yes") for o in p] + [(o, "no") for o in n]
    random.shuffle(items)
    return QA_PROMPT.format(", ".join(o for o, _ in items)), "; ".join(f"{o}: {y}" for o, y in items)


def photo(r, aug):
    if not aug:
        return load_image(ROOT / r["file"], r["crop"])
    im = ImageOps.exif_transpose(Image.open(ROOT / r["file"])).convert("RGB")
    x0, y0, x1, y1 = r["crop"] or [0.0, 0.0, 1.0, 1.0]
    w, h = x1 - x0, y1 - y0
    j = lambda: random.uniform(-0.02, 0.05)
    x0, y0, x1, y1 = max(0.0, x0 + j() * w), max(0.0, y0 + j() * h), min(1.0, x1 - j() * w), min(1.0, y1 - j() * h)
    W, H = im.size
    im = im.crop((int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)))
    im.thumbnail((SIDE, SIDE))
    im = ImageEnhance.Brightness(im).enhance(random.uniform(0.9, 1.1))
    im = ImageEnhance.Contrast(im).enhance(random.uniform(0.9, 1.1))
    if random.random() < 0.3:
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=random.randint(60, 90))
        im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return im


def encode(r, kind="caption", aug=False):
    img = photo(r, aug)
    if kind == "qa":
        q, ans = qa_sample(r)
        msg = lambda answer=None: [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": q}]}] + (
            [{"role": "assistant", "content": [{"type": "text", "text": answer}]}] if answer is not None else [])
        full = proc.apply_chat_template(msg(ans), tokenize=False)
        prompt = proc.apply_chat_template(msg(), tokenize=False, add_generation_prompt=True)
    else:
        full = proc.apply_chat_template(messages(img, target(r)), tokenize=False)
        prompt = proc.apply_chat_template(messages(img), tokenize=False, add_generation_prompt=True)
    enc = proc(text=[full], images=[img], return_tensors="pt")
    plen = proc(text=[prompt], images=[img], return_tensors="pt")["input_ids"].shape[1]
    labels = enc["input_ids"].clone()
    labels[:, :plen] = -100
    enc["labels"] = labels
    return enc


cache = {r["id"]: encode(r) for r in val}
if not a.aug:
    cache.update({r["id"]: encode(r) for r in train})
kinds = ["caption", "qa"] if a.qa else ["caption"]
per_epoch = len(train) * len(kinds)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
steps = math.ceil(per_epoch * a.epochs / a.accum)
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
    order = [(r, k) for r in train for k in kinds]
    random.shuffle(order)
    t, run = time.time(), 0.0
    for i, (r, kind) in enumerate(order):
        enc = cache[r["id"]] if (kind == "caption" and not a.aug) else encode(r, kind, a.aug)
        b = {k: v.to(model.device) for k, v in enc.items()}
        loss = model(**b).loss / a.accum
        loss.backward()
        run += loss.item()
        if (i + 1) % a.accum == 0 or i == len(order) - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            print(f"epoch {ep + 1}/{a.epochs} step {step}/{steps} image {i + 1}/{len(order)} "
                  f"loss {run * a.accum / (i + 1):.3f} {time.time() - t:.0f}s "
                  f"gpu {torch.cuda.max_memory_allocated() / 1e9:.1f}GB", flush=True)
    vl = val_loss()
    log.append({"epoch": ep + 1, "train_loss": run * a.accum / len(order), "val_loss": vl, "sec": time.time() - t})
    print(log[-1], "peak GB", round(torch.cuda.max_memory_allocated() / 1e9, 2), flush=True)
    if vl < best:
        best = vl
        model.save_pretrained(ROOT / a.out)
(ROOT / a.out).mkdir(parents=True, exist_ok=True)
(ROOT / a.out / "train_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
(ROOT / a.out / "vlm_meta.json").write_text(json.dumps({"side": SIDE, "epochs": a.epochs, "rank": a.rank, "fields": "with explanation",
                                                        "qa": a.qa, "aug": a.aug, "hard": bool(a.hard), "schema": SCHEMA,
                                                        "labels": "v2 cleaned" + (" + photo_artifacts" if SCHEMA == "v3" else "")}), encoding="utf-8")
print("best val", round(best, 4))

import collections, csv, json, re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vlm_common import FIELDS, records

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "charts"
OUT.mkdir(parents=True, exist_ok=True)
BASE, LORA = "#2a78d6", "#eb6834"
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"font.family": "Segoe UI", "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.facecolor": BG, "figure.facecolor": BG,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.titlesize": 14, "axes.titlecolor": INK, "axes.titlelocation": "left"})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)


log = json.loads((ROOT / "models" / "qwen_lora" / "train_log.json").read_text(encoding="utf-8"))
first = float(re.search(r"val loss before ([\d.]+)", (ROOT / "outputs" / "train_lora.log").read_text(encoding="utf-8")).group(1))
fig, ax = plt.subplots(figsize=(7, 4.2))
ep = [0] + [l["epoch"] for l in log]
val = [first] + [l["val_loss"] for l in log]
tr = [l["train_loss"] for l in log]
ax.plot(ep, val, color=LORA, lw=2, marker="o", ms=8, label="Validation (29 images)")
ax.plot(ep[1:], tr, color=BASE, lw=2, marker="o", ms=8, label="Train (136 images)")
for x, y in [(0, val[0]), (ep[-1], val[-1])]:
    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", color=INK)
ax.set_xticks(ep)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_ylim(0, max(val) * 1.15)
ax.grid(axis="y", color=GRID)
ax.legend(frameon=False, labelcolor=INK)
ax.set_title("LoRA fine-tune: validation loss 1.83 → 1.11")
save(fig, "1_training_loss.png")

ev = json.loads((ROOT / "outputs" / "eval_val.json").read_text(encoding="utf-8"))
names = [("schema_ok", "All fields present"), ("json_ok", "Valid JSON"), ("theme_clip_sim", "Theme similarity (CLIP)"),
         ("desc_clip_sim", "Description similarity (CLIP)"), ("objects_f1", "Objects F1"), ("desc_rougeL", "Description ROUGE-L")]
fig, ax = plt.subplots(figsize=(8, 4.8))
h = 0.36
for i, (k, lab) in enumerate(names):
    b, l = ev["base"][k], ev["lora"][k]
    ax.barh(i + h / 2 + 0.02, b, h, color=BASE, label="Base Qwen2.5-VL-3B" if i == 0 else None)
    ax.barh(i - h / 2 - 0.02, l, h, color=LORA, label="Fine-tuned (LoRA)" if i == 0 else None)
    ax.text(b + 0.01, i + h / 2 + 0.02, f"{b:.2f}", va="center", color=INK2, fontsize=10)
    ax.text(l + 0.01, i - h / 2 - 0.02, f"{l:.2f}", va="center", color=INK, fontsize=10, fontweight="bold")
ax.set_yticks(range(len(names)), [n for _, n in names])
ax.set_xlim(0, 1.1)
ax.grid(axis="x", color=GRID)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.1), ncol=2, labelcolor=INK)
ax.set_title("Base vs fine-tuned on validation (higher is better)")
save(fig, "2_base_vs_lora.png")


def words(s):
    return re.findall(r"[a-z]+", (s or "").lower())


def rouge_l(p, g):
    p, g = words(p), words(g)
    if not p or not g:
        return 0.0
    dp = [[0] * (len(g) + 1) for _ in range(len(p) + 1)]
    for i in range(len(p)):
        for j in range(len(g)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if p[i] == g[j] else max(dp[i][j + 1], dp[i + 1][j])
    l = dp[-1][-1]
    return 0.0 if l == 0 else 2 * (l / len(p)) * (l / len(g)) / (l / len(p) + l / len(g))


recs = records()
gold = {r["id"]: r for r in recs if r["split"] == "val"}
P = {}
for t in ("base", "lora"):
    f = ROOT / "outputs" / f"preds_{t}_val.jsonl"
    P[t] = {json.loads(l)["id"]: json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()}
sc = []
for i in gold:
    d = {t: str((P[t][i]["pred"] or {}).get("description", P[t][i]["raw"])) for t in P}
    sc.append((i, rouge_l(d["base"], gold[i]["description"]), rouge_l(d["lora"], gold[i]["description"])))
sc.sort(key=lambda x: x[2] - x[1])
fig, ax = plt.subplots(figsize=(7.5, 8))
for y, (i, b, l) in enumerate(sc):
    ax.plot([b, l], [y, y], color=GRID, lw=2, zorder=1)
    ax.scatter(b, y, color=BASE, s=55, zorder=2, edgecolors=BG, linewidths=2, label="Base" if y == 0 else None)
    ax.scatter(l, y, color=LORA, s=55, zorder=3, edgecolors=BG, linewidths=2, label="Fine-tuned" if y == 0 else None)
ax.set_yticks(range(len(sc)), [s[0] for s in sc], fontsize=9)
ax.set_xlabel("ROUGE-L vs hand label")
ax.grid(axis="x", color=GRID)
ax.set_axisbelow(True)
better = sum(1 for _, b, l in sc if l > b)
ax.legend(frameon=False, loc="lower right", labelcolor=INK)
ax.set_title(f"Per image: fine-tuned better on {better} of {len(sc)}")
save(fig, "3_per_image_rouge.png")

cnt = collections.Counter((r["split"], bool(r["known_artist"])) for r in recs)
fig, ax = plt.subplots(figsize=(7, 3.6))
splits = ["train", "val", "test"]
for y, s in enumerate(splits):
    k, u = cnt[(s, True)], cnt[(s, False)]
    ax.barh(y, k, color=LORA, label="Artist known (wall label)" if y == 0 else None)
    ax.barh(y, u, left=k + 1.2, color=BASE, label="Artist unknown" if y == 0 else None)
    ax.text(k + u + 3, y, f"{k + u}", va="center", color=INK, fontweight="bold")
ax.set_yticks(range(3), splits)
ax.invert_yaxis()
ax.set_xlabel("Paintings")
ax.grid(axis="x", color=GRID)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower right", labelcolor=INK)
ax.set_title("Dataset: 195 labeled paintings, 82 with known artist")
save(fig, "4_dataset_splits.png")

GROUPS = [("Egyptian papyrus", "egyptian"), ("Ethiopian icon", "ethiopian"), ("African contemporary", "african"),
          ("Abstract", "abstract|drip|biomorphic|textural"), ("Landscape", "landscape|seascape"),
          ("Cityscape", "cityscape|townscape|urban|architectural"), ("Still life / floral", "still life|floral"),
          ("Portrait", "portrait"), ("Figurative", "figurative|caricature|surreal|illustration|cubist|naive")]
g = collections.Counter()
for r in recs:
    s = (r["style_guess"] or "").lower()
    g[next((n for n, p in GROUPS if re.search(p, s)), "Other")] += 1
items = g.most_common()
fig, ax = plt.subplots(figsize=(7, 4.4))
ax.barh(range(len(items)), [v for _, v in items], color=BASE, height=0.62)
for y, (_, v) in enumerate(items):
    ax.text(v + 0.8, y, str(v), va="center", color=INK)
ax.set_yticks(range(len(items)), [n for n, _ in items])
ax.invert_yaxis()
ax.set_xlabel("Paintings")
ax.grid(axis="x", color=GRID)
ax.set_axisbelow(True)
ax.set_title("What the collection contains (style groups)")
save(fig, "5_style_groups.png")
print(sorted(p.name for p in OUT.glob("*.png")))

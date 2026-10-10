import json, re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
from charts import BASE, BG, GRID, INK, INK2, LORA, OUT, ROOT, rouge_l, save
from vlm_common import records

MISS = "#c9c8c1"
recs = records()
n = len(recs)

fields = [("known_artist", "Artist"), ("known_title", "Title"), ("known_year", "Year")]
fig, ax = plt.subplots(figsize=(8, 3.6))
for y, (k, lab) in enumerate(fields):
    have = sum(1 for r in recs if r[k])
    miss = n - have
    ax.barh(y, have, color=BASE, height=0.6, label="Known (from wall label)" if y == 0 else None)
    ax.barh(y, miss, left=have + 1.5, color=MISS, height=0.6, label="Missing" if y == 0 else None)
    ax.text(have / 2, y, f"{have}", ha="center", va="center", color="white", fontweight="bold")
    ax.text(have + 1.5 + miss / 2, y, f"{miss} missing ({100 * miss / n:.0f}%)", ha="center", va="center", color=INK)
ax.set_yticks(range(len(fields)), [f for _, f in fields])
ax.invert_yaxis()
ax.set_xlim(0, n + 4)
ax.set_xlabel(f"Paintings (of {n})")
ax.grid(axis="x", color=GRID)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2, labelcolor=INK)
ax.set_title(f"Metadata coverage: {n - sum(1 for r in recs if r['known_artist'])} of {n} paintings have no known artist")
save(fig, "6_missing_metadata.png")

gold = {r["id"]: r for r in recs if r["split"] == "val"}
P = {}
for t in ("base", "lora"):
    f = ROOT / "outputs" / f"preds_{t}_val.jsonl"
    P[t] = {json.loads(l)["id"]: json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()}
model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", device="cuda")
tok = open_clip.get_tokenizer("ViT-L-14")


@torch.no_grad()
def sim(a, b):
    e = model.encode_text(tok(a + b).cuda()).float()
    e = e / e.norm(dim=-1, keepdim=True)
    return (e[:len(a)] * e[len(a):]).sum(-1).cpu().numpy()


ids = list(gold)
groups = {"Artist known": [i for i in ids if gold[i]["known_artist"]], "Artist unknown": [i for i in ids if not gold[i]["known_artist"]]}
score = {}
for t in P:
    d = [str((P[t][i]["pred"] or {}).get("description", P[t][i]["raw"]))[:600] for i in ids]
    g = [gold[i]["description"] for i in ids]
    cs = sim(d, g)
    for i, dd, gg, c in zip(ids, d, g, cs):
        score[(t, i)] = (float(c), rouge_l(dd, gg))

fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=False)
for ax, (m, lab) in zip(axes, [(0, "Meaning match (CLIP similarity)"), (1, "Wording match (ROUGE-L)")]):
    x = np.arange(len(groups))
    w = 0.36
    for j, (t, col, name) in enumerate([("base", BASE, "Base model"), ("lora", LORA, "Fine-tuned")]):
        vals = [np.mean([score[(t, i)][m] for i in groups[gname]]) for gname in groups]
        bars = ax.bar(x + (j - 0.5) * (w + 0.03), vals, w, color=col, label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", color=INK if t == "lora" else INK2,
                    fontweight="bold" if t == "lora" else "normal", fontsize=10)
    ax.set_xticks(x, [f"{g}\n(n={len(v)})" for g, v in groups.items()])
    ax.set_ylim(0, 0.8 if m == 0 else 0.5)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    ax.set_title(lab, fontsize=12)
axes[0].legend(frameon=False, loc="upper left", labelcolor=INK)
fig.suptitle("Description accuracy vs hand labels (validation, 29 paintings)", x=0.02, ha="left", fontweight="bold", fontsize=14, color=INK)
save(fig, "7_description_accuracy_by_author.png")

lo = [score[("lora", i)][0] for i in ids]
bins = [("Close match (≥ 0.70)", lambda v: v >= 0.70), ("Partly right (0.55–0.70)", lambda v: 0.55 <= v < 0.70), ("Off (< 0.55)", lambda v: v < 0.55)]
fig, ax = plt.subplots(figsize=(8, 3.4))
for y, (t, col, name) in enumerate([("base", BASE, "Base model"), ("lora", LORA, "Fine-tuned")]):
    vs = [score[(t, i)][0] for i in ids]
    left = 0
    for k, (bn, f) in enumerate(bins):
        c = sum(1 for v in vs if f(v))
        shade = [col, col + "aa", col + "55"][k]
        ax.barh(y, c, left=left, color=shade, height=0.6, edgecolor=BG, linewidth=2)
        if c:
            ax.text(left + c / 2, y, str(c), ha="center", va="center", color=INK, fontweight="bold")
        left += c
ax.set_yticks([0, 1], ["Base model", "Fine-tuned"])
ax.invert_yaxis()
ax.set_xlim(0, len(ids))
ax.set_xlabel("Paintings")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#888888" + a, label=b[0]) for a, b in zip(["", "aa", "55"], bins)], frameon=False,
          loc="upper center", bbox_to_anchor=(0.45, -0.28), ncol=3, labelcolor=INK)
ax.set_title("How many descriptions match the hand label (CLIP meaning similarity)")
save(fig, "8_description_match_buckets.png")
for t in P:
    print(t, [sum(1 for i in ids if f(score[(t, i)][0])) for _, f in bins])

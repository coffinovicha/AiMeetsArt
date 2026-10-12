import argparse, csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from vlm_common import ROOT, records

ap = argparse.ArgumentParser()
ap.add_argument("--tags", default="base,lora,lora560,expl560,verified2")
ap.add_argument("--final", default="verified2")
ap.add_argument("--review", default="manual_review_v2_val.csv:rating_verified2")
ap.add_argument("--split", default="val")
a = ap.parse_args()

NAMES = {"base": "Base Qwen, no training", "lora": "Fine-tuned, 448 px", "lora560": "Fine-tuned, 560 px",
         "expl560": "560 px + explanations", "verified2": "Final: v2 + verified mode"}
ACC, ANCH, MUTED = "#2a78d6", "#8a8984", "#c9c7c0"
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
OUT = ROOT / "outputs" / "charts"
plt.rcParams.update({"font.family": "Segoe UI", "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.facecolor": BG, "figure.facecolor": BG,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.titlesize": 14, "axes.titlecolor": INK, "axes.titlelocation": "left"})

gold = {r["id"]: r for r in records() if r["split"] == a.split}
ids = sorted(gold)
m = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")
enc = lambda xs: m.encode([str(x or "") for x in xs], normalize_embeddings=True)
G = enc([gold[i]["description"] for i in ids])
GE = enc([gold[i]["explanation"] for i in ids])
off = ~np.eye(len(ids), dtype=bool)


def stats(P, R):
    S = P @ R.T
    d = np.diag(S)
    rank = (S > d[:, None]).sum(1) + 1
    return {"same": float(d.mean()), "other": float(S[off].mean()), "top1": int((rank == 1).sum()),
            "top3": int((rank <= 3).sum()), "per_image": dict(zip(ids, map(float, d)))}


res = {"n": len(ids), "anchors": {"human_other_painting": float((G @ G.T)[off].mean()),
                                  "human_same_painting": float(np.diag(G @ GE.T).mean())}, "models": {}, "explanations": {}}
for t in a.tags.split(","):
    f = ROOT / "outputs" / f"preds_{t}_{a.split}.jsonl"
    if not f.exists():
        continue
    P = {json.loads(l)["id"]: json.loads(l)["pred"] or {} for l in f.read_text(encoding="utf-8").splitlines() if l.strip()}
    if not all(i in P for i in ids):
        print(f"skip {t}: {sum(i in P for i in ids)}/{len(ids)}")
        continue
    res["models"][t] = stats(enc([P[i].get("description") for i in ids]), G)
    if all(P[i].get("explanation") for i in ids):
        res["explanations"][t] = stats(enc([P[i]["explanation"] for i in ids]), GE)
(ROOT / "outputs" / f"meaning_{a.split}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def save(fig, name, rect=(0, 0, 1, 1)):
    fig.tight_layout(rect=rect)
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)


def scale_chart(rows, title, sub, name):
    fig, ax = plt.subplots(figsize=(8.5, 0.62 * len(rows) + 1.9))
    y = np.arange(len(rows))[::-1]
    for yy, (lab, v, anchor) in zip(y, rows):
        c = ANCH if anchor else ACC
        ax.hlines(yy, 0, v, color=GRID, lw=2)
        ax.plot(v, yy, "o", ms=11, color=c, mec=BG, mew=2)
        ax.text(v + 0.015, yy, f"{v:.2f}", va="center", color=INK, fontweight="bold" if not anchor else "normal")
    ax.set_yticks(y, [r[0] for r in rows])
    for t, r in zip(ax.get_yticklabels(), rows):
        t.set_color(INK2 if r[2] else INK)
        t.set_fontstyle("italic" if r[2] else "normal")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Meaning similarity to the human-written text (0 = unrelated, 1 = same meaning)")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    for s in ("left",):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title)
    fig.text(0.01, 0.015, sub, color=INK2, fontsize=9)
    save(fig, name, rect=(0, 0.05, 1, 1))


A, M = res["anchors"], res["models"]
mean_other = float(np.mean([v["other"] for v in M.values()]))
rows = [("Reference: a different painting", mean_other, True)]
rows += [(NAMES[t], v["same"], False) for t, v in M.items()]
rows += [("Reference: same painting, 2nd human text", A["human_same_painting"], True)]
scale_chart(rows, "How close in meaning are the model's descriptions?",
            "Grey = reference points. 'Different painting' = model text vs the wrong painting's description. Validation, 29 images, cleaned labels.",
            "9_meaning_scale.png")

fig, ax = plt.subplots(figsize=(8.5, 0.62 * len(M) + 1.9))
y = np.arange(len(M))[::-1]
for yy, (t, v) in zip(y, M.items()):
    ax.barh(yy, v["top1"], 0.55, color=ACC)
    ax.text(v["top1"] + 0.3, yy, f"{v['top1']} of {res['n']}  ({v['top1'] / res['n']:.0%})", va="center", color=INK)
ax.axvline(1, color=ANCH, lw=1.5, ls="--")
ax.text(1.3, y[-1] - 0.5, "pure chance = 1", color=INK2, fontsize=9, va="center")
ax.set_ylim(y[-1] - 0.75, y[0] + 0.5)
ax.set_yticks(y, [NAMES[t] for t in M])
ax.set_xlim(0, res["n"])
ax.set_xlabel("Descriptions whose closest human description is the right painting")
ax.grid(axis="x", color=GRID)
ax.set_axisbelow(True)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.set_title("Can the description pick out its own painting among 29?")
save(fig, "10_picks_own_painting.png")

rf_name, rkey = a.review.split(":")
rev = {r["id"]: r for r in csv.DictReader(open(ROOT / "outputs" / rf_name, encoding="utf-8"))}
order = [("correct", "Correct"), ("minor", "Minor errors"), ("invented", "Invented details"), ("wrong_subject", "Wrong subject")]
fig, ax = plt.subplots(figsize=(8.5, 4.4))
rng = np.random.default_rng(0)
summary = {}
for k, (key, tag) in enumerate([(rkey, a.final)]):
    per = M[tag]["per_image"]
    for x, (r, lab) in enumerate(order):
        vals = [per[i] for i in ids if rev.get(i, {}).get(key) == r]
        summary[r] = (len(vals), float(np.mean(vals)) if vals else None)
        if not vals:
            continue
        ax.plot(x + rng.uniform(-0.12, 0.12, len(vals)), vals, "o", ms=8, color=ACC, alpha=0.75, mec=BG, mew=1.5)
        mu = np.mean(vals)
        ax.hlines(mu, x - 0.28, x + 0.28, color=INK, lw=2)
        ax.text(x + 0.31, mu, f"avg {mu:.2f}", va="center", color=INK, fontsize=10)
ax.set_xticks(range(len(order)), [f"{lab}\n({summary[r][0]} image{'' if summary[r][0] == 1 else 's'})" for r, lab in order])
ax.set_xlim(-0.5, len(order) - 0.2)
ax.set_ylim(0.3, 1)
ax.set_ylabel("Meaning similarity")
ax.grid(axis="y", color=GRID)
ax.set_axisbelow(True)
ax.set_title(f"Does the score agree with a human reviewer? ({NAMES[a.final]})")
save(fig, "11_score_vs_human_rating.png")
res["by_rating_final"] = summary

E = res["explanations"]
reviews = [("manual_review_expl_val.csv", "context_expl560", "expl560", "Run 3 (explanation model)", MUTED),
           ("manual_review_expl_v2_val.csv", "context_verified2", "verified2", "Final: v2 + verified mode", ACC)]
reviews = [x for x in reviews if (ROOT / "outputs" / x[0]).exists() and x[2] in E]
if reviews:
    cats = [("sound", "Sound: matches the painting, or clearly hedged"), ("propagated", "Built on a misread object"),
            ("wrong_fact", "Wrong background fact"), ("unsupported", "Unsupported leap")]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    h = 0.36 if len(reviews) > 1 else 0.55
    res["explanation_claims"] = {}
    for j, (fname, key, tag, lab, col) in enumerate(reviews):
        rv = list(csv.DictReader(open(ROOT / "outputs" / fname, encoding="utf-8")))
        cnt = {k: sum(r[key] == k for r in rv) for k, _ in cats}
        res["explanation_claims"][tag] = cnt
        off = (len(reviews) - 1) / 2 - j
        for yy, (k, _) in zip(np.arange(len(cats))[::-1], cats):
            yb = yy + off * (h + 0.04)
            ax.barh(yb, cnt[k], h, color=col, label=lab if k == "sound" else None)
            ax.text(cnt[k] + 0.3, yb, str(cnt[k]), va="center", color=INK, fontsize=10, fontweight="bold" if col == ACC else "normal")
    ax.set_yticks(np.arange(len(cats))[::-1], [lab for _, lab in cats])
    ax.set_xlim(0, res["n"])
    ax.set_xlabel("Explanations (of 29, manual review)")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="lower right", labelcolor=INK)
    ax.set_title("What are the explanations' interpretations based on?")
    foot = "; ".join(f"{lab.split(' (')[0].split(':')[0]}: meaning similarity {E[tag]['same']:.2f} (different painting {E[tag]['other']:.2f})" for _, _, tag, lab, _ in reviews)
    fig.text(0.01, 0.015, foot + ". Validation set, cleaned labels.", color=INK2, fontsize=9)
    save(fig, "12_explanation_claims.png", rect=(0, 0.06, 1, 1))

(ROOT / "outputs" / f"meaning_{a.split}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k not in ("models", "explanations")}, indent=1))
for grp in ("models", "explanations"):
    for t, v in res[grp].items():
        print(grp, t, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items() if k != "per_image"})

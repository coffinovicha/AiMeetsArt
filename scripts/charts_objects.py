import csv
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vlm_common import ROOT

ACC, MUTED = "#2a78d6", "#c9c7c0"
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"font.family": "Segoe UI", "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.facecolor": BG, "figure.facecolor": BG,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.titlesize": 14, "axes.titlecolor": INK, "axes.titlelocation": "left"})
rows = list(csv.DictReader(open(ROOT / "outputs" / "object_review_val.csv", encoding="utf-8")))
variants = [("expl560", "Run 3, plain output"), ("verified", "Run 3 + verified mode"), ("verified2", "Final: v2 + verified mode")]
fig, ax = plt.subplots(figsize=(8.5, 3.8))
for y, (v, lab) in zip([2, 1, 0], variants):
    r = [x for x in rows if x["variant"] == v]
    wrong = [x for x in r if x["present"] == "0"]
    un = sum(x["status"] in ("shown", "confirmed") for x in wrong)
    mk = len(wrong) - un
    ax.barh(y, un, 0.55, color=ACC, label="shown without a warning" if y == 2 else None)
    ax.barh(y, mk, 0.55, left=un + 0.3 if mk else un, color=MUTED, label="marked as doubtful" if y == 1 else None)
    ax.text(un / 2, y, str(un), ha="center", va="center", color="white", fontweight="bold")
    if mk:
        ax.text(un + 0.3 + mk / 2, y, str(mk), ha="center", va="center", color=INK)
    ax.text(len(wrong) + 0.8, y, f"{len(wrong)} wrong of {len(r)} objects", va="center", color=INK2, fontsize=10)
ax.set_yticks([2, 1, 0], [l for _, l in variants])
ax.set_xlim(0, 34)
ax.set_xlabel("Wrong objects (checked by hand against each painting)")
ax.grid(axis="x", color=GRID)
ax.set_axisbelow(True)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2, labelcolor=INK)
ax.set_title("Verified mode turns most wrong objects into marked ones")
fig.tight_layout()
fig.savefig(ROOT / "outputs" / "charts" / "13_wrong_objects.png", dpi=160)

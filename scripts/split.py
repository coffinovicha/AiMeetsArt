import csv, json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
L = ROOT / "labels"
SEED = 42
DUPES = [["F1-016", "F1-017"], ["F1-040", "F1-046"], ["F1-056", "F1-062"]]

ids = [json.loads(l)["id"] for l in (L / "labels.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
grouped = {i for g in DUPES for i in g}
groups = [g for g in DUPES] + [[i] for i in ids if i not in grouped]
random.Random(SEED).shuffle(groups)
n = len(ids)
targets = {"train": round(0.70 * n), "val": round(0.15 * n)}
split, counts = {}, {"train": 0, "val": 0, "test": 0}
for g in groups:
    s = "train" if counts["train"] + len(g) <= targets["train"] else "val" if counts["val"] + len(g) <= targets["val"] else "test"
    for i in g:
        split[i] = s
    counts[s] += len(g)
with (L / "splits.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "split"])
    for i in ids:
        w.writerow([i, split[i]])
print(counts)

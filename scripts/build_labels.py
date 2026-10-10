import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
L = ROOT / "labels"
FIELDS = ["description", "theme", "mood", "objects", "style_guess", "medium_guess",
          "era_guess", "lighting", "composition", "explanation", "title_guess"]

with (L / "manifest.csv").open(encoding="utf-8") as f:
    manifest = {r["id"]: r for r in csv.DictReader(f)}
pal = json.loads((L / "palettes.json").read_text(encoding="utf-8"))
expl = {json.loads(l)["id"]: json.loads(l)["explanation"] for l in (L / "explanations.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
ann = {}
for line in (L / "annotations.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        a = json.loads(line)
        ann[a["id"]] = a

n = 0
with (L / "labels.jsonl").open("w", encoding="utf-8") as f:
    for i, m in manifest.items():
        if i not in ann:
            continue
        rec = {"id": i, "file": m["file"],
               "known_artist": m["known_artist"] or None,
               "known_title": m["known_title"] or None,
               "known_year": m["known_year"] or None}
        rec.update({k: ann[i].get(k) for k in FIELDS})
        rec["explanation"] = expl.get(i)
        rec["palette"] = pal[i]["colors"] if i in pal else None
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1
print(n, "records,", len(manifest) - n, "remaining")

import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "labels" / "annotations.jsonl"
crops = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
recs = [json.loads(l) for l in A.read_text(encoding="utf-8").splitlines() if l.strip()]
for r in recs:
    if r["id"] in crops:
        r["crop"] = crops[r["id"]]
A.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
print(sum(1 for r in recs if "crop" in r), "records with crop")

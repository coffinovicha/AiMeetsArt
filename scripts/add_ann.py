import json, sys
from pathlib import Path

A = Path(__file__).resolve().parents[1] / "labels" / "annotations.jsonl"
have = {json.loads(l)["id"] for l in A.read_text(encoding="utf-8").splitlines() if l.strip()}
new = [json.loads(l) for l in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if l.strip()]
add = [r for r in new if r["id"] not in have]
with A.open("a", encoding="utf-8") as f:
    for r in add:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(add), "added,", len(new) - len(add), "skipped")

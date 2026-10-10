import json, sys
from pathlib import Path

E = Path(__file__).resolve().parents[1] / "labels" / "explanations.jsonl"
have = {json.loads(l)["id"] for l in E.read_text(encoding="utf-8").splitlines() if l.strip()} if E.exists() else set()
new = [json.loads(l) for l in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if l.strip()]
add = [r for r in new if r["id"] not in have]
with E.open("a", encoding="utf-8") as f:
    for r in add:
        f.write(json.dumps({"id": r["id"], "explanation": r["explanation"]}, ensure_ascii=False) + "\n")
print(len(add), "added,", len(have) + len(add), "total")

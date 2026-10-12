import argparse, json, random
import numpy as np
from titles import analysis_text, enc, meaning_text, norm
from vlm_common import ROOT, records

ap = argparse.ArgumentParser()
ap.add_argument("--preds", default="preds_verified2_val.jsonl")
ap.add_argument("--draws", type=int, default=10)
a = ap.parse_args()
R = {r["id"]: r for r in records()}
P = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "outputs" / a.preds).read_text(encoding="utf-8").splitlines() if l.strip()}
pool = [(r["id"], r["known_artist"].strip(), r["known_title"].strip()) for r in R.values() if r["known_title"]]
rng = random.Random(0)
rows, hits, tot = [], 0, 0
for i, d in P.items():
    r = R[i]
    if not r["known_title"]:
        continue
    conf = [o["object"] for o in d["objects"] if o["status"] == "confirmed"]
    doubt = [o["object"] for o in d["objects"] if o["status"] != "confirmed"]
    ref_a = enc([analysis_text(d["pred"], conf, doubt, d.get("palette"))])[0]
    ref_m = enc([meaning_text(d["pred"])])[0]
    others = list({norm(t): t for _, art, t in pool if art != r["known_artist"].strip() and norm(t) != norm(r["known_title"])}.values())
    correct = 0
    picks = []
    for _ in range(a.draws):
        opts = [r["known_title"]] + rng.sample(others, 4)
        V = enc(opts)
        score = 0.5 * (V @ ref_a) + 0.5 * (V @ ref_m)
        pick = int(np.argmax(score))
        correct += pick == 0
        picks.append(opts[pick])
    rows.append({"id": i, "real": r["known_title"], "correct": correct, "draws": a.draws, "example_options": opts, "example_pick": picks[-1]})
    hits += correct
    tot += a.draws
    print(f"{i} {r['known_title'][:40]:40s} picked the real title {correct}/{a.draws}", flush=True)
summ = {"paintings": len(rows), "accuracy": round(hits / tot, 3), "chance": 0.2,
        "paintings_mostly_right": sum(x["correct"] > a.draws / 2 for x in rows)}
out = ROOT / "outputs" / f"title_choice_{a.preds.replace('preds_', '').replace('.jsonl', '')}.json"
out.write_text(json.dumps({"summary": summ, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summ))

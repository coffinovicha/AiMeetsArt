import json
import numpy as np
from sentence_transformers import SentenceTransformer
from vlm_common import ROOT, records

TAGS = ["base560", "verified2"]
gold = {r["id"]: r for r in records() if r["split"] == "test"}
m = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")
enc = lambda xs: m.encode([str(x or "") for x in xs], normalize_embeddings=True)
P = {t: {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "outputs" / f"preds_{t}_test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
     for t in TAGS}
ids = [i for i in gold if all(i in P[t] for t in TAGS)]
desc = lambda t, i, k="description": str((P[t][i]["pred"] or {}).get(k) or P[t][i].get("raw", ""))[:600]
G = enc([gold[i]["description"] for i in ids])
GE = enc([gold[i]["explanation"] for i in ids])
off = ~np.eye(len(ids), dtype=bool)


def stats(X, R):
    S = X @ R.T
    d = np.diag(S)
    rank = (S > d[:, None]).sum(1) + 1
    return {"same": round(float(d.mean()), 3), "other": round(float(S[off].mean()), 3), "own_painting_top1": int((rank == 1).sum()),
            "own_painting_top3": int((rank <= 3).sum()), "n": len(ids)}


res = {"n": len(ids), "anchors": {"human_other_painting": round(float((G @ G.T)[off].mean()), 3),
                                  "human_same_painting": round(float(np.diag(G @ GE.T).mean()), 3)}, "descriptions": {}, "explanations": {}}
for t in TAGS:
    res["descriptions"][t] = stats(enc([desc(t, i) for i in ids]), G)
    res["descriptions"][t]["json_ok"] = sum(bool(P[t][i]["pred"]) for i in ids)
    res["explanations"][t] = stats(enc([desc(t, i, "explanation") for i in ids]), GE)
st = {}
for i in ids:
    for o in P["verified2"][i].get("objects", []):
        st[o["status"]] = st.get(o["status"], 0) + 1
res["object_check"] = st
res["subject_unclear"] = sum(bool(P["verified2"][i].get("subject_unclear")) for i in ids)
res["per_image"] = {i: round(float(x), 3) for i, x in zip(ids, (enc([desc("verified2", i) for i in ids]) * G).sum(1))}
(ROOT / "outputs" / "meaning_test.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k != "per_image"}, indent=1))

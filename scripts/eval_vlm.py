import argparse, json, re
from pathlib import Path
import numpy as np
from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer
from vlm_common import FIELDS, ROOT, records

ap = argparse.ArgumentParser()
ap.add_argument("--tags", default="base,lora")
ap.add_argument("--split", default="val")
ap.add_argument("--examples", type=int, default=5)
a = ap.parse_args()

gold = {r["id"]: r for r in records() if r["split"] == a.split}
bs = BERTScorer(model_type="roberta-large", lang="en", rescale_with_baseline=True, device="cuda")
sbert = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")


def words(s):
    return re.findall(r"[a-z]+", (s or "").lower())


def rouge_l(p, g):
    p, g = words(p), words(g)
    if not p or not g:
        return 0.0
    dp = [[0] * (len(g) + 1) for _ in range(len(p) + 1)]
    for i in range(len(p)):
        for j in range(len(g)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if p[i] == g[j] else max(dp[i][j + 1], dp[i + 1][j])
    l = dp[-1][-1]
    pr, rc = l / len(p), l / len(g)
    return 0.0 if l == 0 else 2 * pr * rc / (pr + rc)


def bert_f1(ps, gs):
    return bs.score(ps, gs)[2].numpy()


def sbert_sim(ps, gs):
    a = sbert.encode(ps, normalize_embeddings=True)
    b = sbert.encode(gs, normalize_embeddings=True)
    return (a * b).sum(-1)


def obj_f1(p, g):
    p = {x.lower().strip() for x in (p or []) if isinstance(x, str)}
    g = {x.lower().strip() for x in (g or [])}
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    tp = len(p & g)
    return 0.0 if tp == 0 else 2 * tp / (len(p) + len(g))


rows, preds = {}, {}
for tag in a.tags.split(","):
    f = ROOT / "outputs" / f"preds_{tag}_{a.split}.jsonl"
    P = {json.loads(l)["id"]: json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()}
    preds[tag] = P
    ids = [i for i in gold if i in P]
    parsed = [i for i in ids if isinstance(P[i]["pred"], dict)]
    schema = [i for i in parsed if all(k in P[i]["pred"] for k in FIELDS)]
    pd = [str((P[i]["pred"] or {}).get("description", P[i]["raw"]))[:600] for i in ids]
    gd = [gold[i]["description"] for i in ids]
    rows[tag] = {
        "n": len(ids),
        "json_ok": len(parsed) / len(ids),
        "schema_ok": len(schema) / len(ids),
        "desc_rougeL": float(np.mean([rouge_l(p, g) for p, g in zip(pd, gd)])),
        "desc_bertscore": float(np.mean(bert_f1(pd, gd))),
        "desc_sbert_sim": float(np.mean(sbert_sim(pd, gd))),
        "theme_sbert_sim": float(np.mean(sbert_sim([str((P[i]["pred"] or {}).get("theme", "")) for i in ids], [gold[i]["theme"] for i in ids]))),
        "objects_f1": float(np.mean([obj_f1((P[i]["pred"] or {}).get("objects"), gold[i]["objects"]) for i in ids])),
        "avg_desc_words": float(np.mean([len(words(p)) for p in pd])),
    }

print(f"{'metric':16}" + "".join(f"{t:>12}" for t in rows))
for k in next(iter(rows.values())):
    print(f"{k:16}" + "".join(f"{rows[t][k]:>12.3f}" for t in rows))
out = ROOT / "outputs" / f"eval_{a.split}.json"
out.write_text(json.dumps(rows, indent=1), encoding="utf-8")

common = [i for i in gold if all(i in preds[t] for t in rows)][: a.examples]
lines = []
for i in common:
    lines.append(f"## {i}\nGOLD: {gold[i]['description']}")
    for t in rows:
        p = preds[t][i]["pred"] or {}
        lines.append(f"{t.upper()}: {p.get('description', preds[t][i]['raw'][:400])}")
    lines.append("")
(ROOT / "outputs" / f"examples_{a.split}.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))

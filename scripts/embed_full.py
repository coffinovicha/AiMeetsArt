import csv, glob, io, os, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from db_index import clip_model
from vlm_common import ROOT

SRC = Path(os.path.expanduser("~/.cache/huggingface/hub/datasets--Artificio--WikiArt"))
OUT = ROOT / "artificio_data" / "full"
OUT.mkdir(parents=True, exist_ok=True)
CHUNK, BATCH = 4096, 48
COLS = ["title", "artist", "date", "genre", "style", "filename"]

shards = sorted(glob.glob(str(SRC / "snapshots" / "*" / "data" / "*.parquet")))
meta_path = OUT / "metadata.csv"
if not meta_path.exists():
    with meta_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row", "shard", "index"] + COLS)
        n = 0
        for s, path in enumerate(shards):
            t = pq.read_table(path, columns=COLS).to_pylist()
            for i, r in enumerate(t):
                w.writerow([n, s, i] + [r[c] for c in COLS])
                n += 1
    print("metadata rows", n, flush=True)

model, prep = clip_model()
for s, path in enumerate(shards):
    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    col = None
    for c0 in range(0, total, CHUNK):
        out = OUT / f"emb_{s}_{c0:06d}.npy"
        if out.exists():
            continue
        t0 = time.time()
        if col is None:
            col = pq.read_table(path, columns=["image"]).column("image")
        tab = col.slice(c0, CHUNK).to_pylist()
        vecs = []
        for b in range(0, len(tab), BATCH):
            ims = []
            for x in tab[b:b + BATCH]:
                try:
                    ims.append(prep(Image.open(io.BytesIO(x["bytes"])).convert("RGB")))
                except Exception:
                    ims.append(torch.zeros(3, 224, 224))
            with torch.no_grad():
                v = model.encode_image(torch.stack(ims).cuda().half()).float()
            vecs.append((v / v.norm(dim=-1, keepdim=True)).cpu().numpy().astype("float16"))
        np.save(out, np.concatenate(vecs))
        print(f"shard {s + 1}/{len(shards)} rows {c0}-{c0 + len(tab)} of {total} {time.time() - t0:.0f}s", flush=True)
print("ALL EMBEDDED", flush=True)

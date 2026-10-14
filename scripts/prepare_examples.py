import hashlib, json, pickle, sys
from autocrop import open_photo
from interpret import Interpreter
from vlm_common import ROOT, records

OUT = ROOT / "outputs" / "app_examples"
OUT.mkdir(parents=True, exist_ok=True)
R = {r["id"]: r for r in records()}
ids = sys.argv[1].split(",")
it = Interpreter()
for i in ids:
    path = ROOT / R[i]["file"]
    f = OUT / f"{i}.pkl"
    if f.exists():
        continue
    out = it.run(open_photo(path), "")
    out["id"], out["file"], out["artist"] = i, str(path), ""
    out["sha1"] = hashlib.sha1(path.read_bytes()).hexdigest()
    pickle.dump(out, open(f, "wb"))
    t = out["title"]
    a = out["analysis"]
    print(i, out["seconds"], "s |", [x["title"] for x in t["titles"]], "| path", t["path"], "| objects",
          [(o["object"], o["status"][:4], o.get("box") is not None) for o in a["objects"]], "| dropped", len(a.get("dropped", [])),
          "| unclear", a.get("subject_unclear"), flush=True)

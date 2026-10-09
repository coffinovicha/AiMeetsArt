import csv, json, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(Path(__file__).parent))
from palette import load

ROOT = Path(__file__).resolve().parents[1]
ids, out = sys.argv[1].split(","), Path(sys.argv[2])
with (ROOT / "labels" / "manifest.csv").open(encoding="utf-8") as f:
    files = {r["id"]: r["file"] for r in csv.DictReader(f)}
ann = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "labels" / "annotations.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
pal = json.loads((ROOT / "labels" / "palettes.json").read_text(encoding="utf-8"))
font = ImageFont.truetype("arial.ttf", 22)
T, cols = 300, 5
rows = (len(ids) + cols - 1) // cols
sheet = Image.new("RGB", (cols * T, rows * (T + 40)), "white")
d = ImageDraw.Draw(sheet)
for k, i in enumerate(ids):
    im = load(ROOT / files[i], ann[i].get("crop"))
    im.thumbnail((T - 4, T - 4))
    x, y = (k % cols) * T, (k // cols) * (T + 40)
    sheet.paste(im, (x + 2, y + 2))
    d.text((x + 4, y + 4), i, fill="red", font=font)
    cs = pal[i]["colors"]
    cx = x + 2
    for c in cs:
        wd = int((T - 4) * c["pct"] / 100)
        d.rectangle([cx, y + T + 4, cx + wd, y + T + 34], fill=c["hex"])
        cx += wd
sheet.save(out, quality=88)

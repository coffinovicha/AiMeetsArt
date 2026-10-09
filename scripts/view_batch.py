import csv, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
start, n, out = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
with (ROOT / "labels" / "manifest.csv").open(encoding="utf-8") as f:
    items = list(csv.DictReader(f))[start:start + n]
font = ImageFont.truetype("arial.ttf", 30)
T = 780
for s in range(0, len(items), 4):
    sheet = Image.new("RGB", (2 * T, 2 * T), "white")
    d = ImageDraw.Draw(sheet)
    for k, it in enumerate(items[s:s + 4]):
        im = ImageOps.exif_transpose(Image.open(ROOT / it["file"])).convert("RGB")
        im.thumbnail((T - 6, T - 6))
        x, y = (k % 2) * T, (k // 2) * T
        sheet.paste(im, (x + 3, y + 3))
        d.rectangle([x + 3, y + 3, x + 130, y + 42], fill="yellow")
        d.text((x + 8, y + 6), it["id"], fill="black", font=font)
    sheet.save(out / f"b_{start + s:03d}.jpg", quality=88)
print([it["id"] for it in items])

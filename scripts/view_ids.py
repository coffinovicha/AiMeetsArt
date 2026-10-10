import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from palette import load
from vlm_common import ROOT, records

split, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
recs = [r for r in records() if r["split"] == split]
font = ImageFont.truetype("arial.ttf", 30)
T = 780
for s in range(0, len(recs), 4):
    sheet = Image.new("RGB", (2 * T, 2 * T), "white")
    d = ImageDraw.Draw(sheet)
    for k, r in enumerate(recs[s:s + 4]):
        im = load(ROOT / r["file"], r["crop"])
        im.thumbnail((T - 6, T - 6))
        x, y = (k % 2) * T, (k // 2) * T
        sheet.paste(im, (x + 3, y + 3))
        d.rectangle([x + 3, y + 3, x + 130, y + 42], fill="yellow")
        d.text((x + 8, y + 6), r["id"], fill="black", font=font)
    sheet.save(out / f"{split}_{s:02d}.jpg", quality=88)
print([r["id"] for r in recs])

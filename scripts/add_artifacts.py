import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
L = ROOT / "labels"
GLARE = "glare on the glass"
ART = {
    "F1-009": "window reflections on the glass, including a bright rectangle in the upper middle",
    "F1-010": "matted and framed behind glass",
    "F1-011": "strong reflections of a window and the room on the glass",
    "F1-016": "reflections on the glass",
    "F1-017": "reflections on the glass in the lower center",
    "F1-020": "gold frame, behind glass",
    "F1-029": "shadow of the photographer on the glass",
    "F1-036": "reflection of the photographer in the glass, between the two faces",
    "F1-040": "reflections on the glass",
    "F1-046": "reflections on the glass",
    "F1-051": "reflections of ceiling lights on the glass",
    "F1-052": "light reflections on the glass",
    "F1-054": "a window reflection on the glass",
    "F1-057": "matted, behind glass",
    "F1-058": "matted inside a dark frame",
    "F1-079": "strong reflections of the photographer and the room cover most of the lower half",
    "F2-004": "the photo is slightly blurred",
    "F2-005": "part of a gold frame on the right",
    "F2-006": "the photo is slightly blurred, with a light reflection on the right",
    "F2-012": "part of a gold frame at the bottom",
    "F2-017": "glare in the upper center",
    "F2-019": "glare on the surface",
    "F2-021": "glare on the surface",
    "F2-025": "glare in the center",
    "F2-027": "glare on the surface",
    "F3-003": GLARE,
    "F3-005": "reflections of windows on the glass",
    "F3-009": "a strong reflection at the bottom",
    "F3-022": "reflections on the glass",
    "F3-027": "a reflected window on the glass",
    "F3-058": GLARE,
    "F3-060": GLARE,
    "F3-061": "strong reflections of a window and sky on the glass",
    "F3-062": "strong reflections of windows on the glass",
    "F3-063": GLARE,
    "F3-064": "a window reflection over part of the face",
    "F3-070": GLARE,
    "F3-071": GLARE,
    "F3-079": GLARE,
}

ann = [json.loads(l) for l in (L / "annotations.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
for a in ann:
    a["photo_artifacts"] = ART.get(a["id"], "none")
(L / "annotations.jsonl").write_text("".join(json.dumps(a, ensure_ascii=False) + "\n" for a in ann), encoding="utf-8")
print(sum(a["photo_artifacts"] != "none" for a in ann), "with artifacts of", len(ann))

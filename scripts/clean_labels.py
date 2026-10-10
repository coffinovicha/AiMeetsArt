import csv, json, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
L = ROOT / "labels"
D, X = "description", "explanation"

EDITS = [
    ("F1-009", D, " The work is behind glass with visible reflections.", ""),
    ("F1-009", "medium_guess", "mixed media on paper, under glass (guess)", "mixed media on paper (guess)"),
    ("F1-009", "lighting", "flat, with glare reflections on glass", "flat"),
    ("F1-010", D, " A signature reading 'Vanni' appears in the lower left; the work is matted and framed behind glass.", ""),
    ("F1-011", D, " Strong reflections of a window and room are visible on the glass.", ""),
    ("F1-011", X, "It is an abstract work about texture and energy, and the reflections on the glass add a second layer of the real room on top of it.", "It is an abstract work about texture and energy."),
    ("F1-011", "lighting", "flat, with strong glare reflections on glass", "flat"),
    ("F1-016", D, " The work is framed behind glass with visible reflections.", ""),
    ("F1-016", "lighting", "flat, with reflections on glass", "flat"),
    ("F1-017", D, "Same artwork as F1-016, photographed again: elegant", "Elegant"),
    ("F1-017", D, " Reflections on the glass are visible in the lower center.", ""),
    ("F1-017", "lighting", "flat, with reflections on glass", "flat"),
    ("F1-020", D, " Initials and '24' are written in the lower left; the work is in a gold frame behind glass.", ""),
    ("F1-024", D, " A signature is visible in the lower right.", ""),
    ("F1-029", D, "'Puerto Rico' is written in the upper left and a signature with '2023' in the lower right.", "'Puerto Rico' is written in the upper left."),
    ("F1-029", X, "Words written on the board, 'Puerto Rico' and '2023', suggest a place and date connected to the work.", "The words 'Puerto Rico' written on the board suggest a place connected to the work."),
    ("F1-029", "lighting", "flat, with shadow of photographer on glass", "flat"),
    ("F1-034", D, " A small signature appears in the lower right.", ""),
    ("F1-036", D, " A signature 'Tekla Mela...' is in the lower right; the photographer is reflected in the glass.", ""),
    ("F1-036", "lighting", "flat, with reflections on glass", "flat"),
    ("F1-040", D, " There are reflections on the glass.", ""),
    ("F1-046", D, "Same artwork as F1-040, photographed again: a dark", "A dark"),
    ("F1-046", D, " Reflections on the glass are visible.", ""),
    ("F1-050", D, " Initials 'A.T.' and '2016' are written in the lower right.", ""),
    ("F1-051", D, " Reflections of ceiling lights appear on the glass.", ""),
    ("F1-052", D, "The stitched texture is visible, with light reflections on the glass.", "The stitched texture is visible."),
    ("F1-053", D, " A signature 'B' with '2010' appears in the lower right.", ""),
    ("F1-054", D, " The work is behind glass with a window reflection.", ""),
    ("F1-054", "lighting", "bright, flat, with reflections on glass", "bright, flat"),
    ("F1-057", D, " A signature appears in the lower left; the work is matted behind glass.", ""),
    ("F1-058", D, "The ground is pale marbled cream, matted inside a dark frame.", "The ground is pale marbled cream."),
    ("F1-061", D, " A signature is in the lower right.", ""),
    ("F1-062", D, "Same artwork as F1-056, photographed again: a pale", "A pale"),
    ("F1-063", D, " A signature is in the lower right.", ""),
    ("F1-069", D, " A signature 'Bacho' appears in the lower right.", ""),
    ("F1-071", D, "White columns and dripping strokes are visible, with a signature in the lower right.", "White columns and dripping strokes are visible."),
    ("F1-074", D, "It is drawn with charcoal and colored crayon on cream paper and signed in the lower right.", "It is drawn with charcoal and colored crayon on cream paper."),
    ("F1-076", D, " A signature is in the lower right corner.", ""),
    ("F1-077", D, " A signature is in the lower right.", ""),
    ("F1-078", D, " A signature is in the lower right.", ""),
    ("F1-079", D, " Strong reflections of the photographer and the room on the glass obscure most of the lower half.", ""),
    ("F1-079", X, "The reflections on the glass make the lower half hard to read, but the tower remains the clear focus.", "The tower is the clear focus."),
    ("F1-079", "lighting", "obscured by glass reflections", "flat"),
    ("F2-003", D, " A signature beginning 'Gel...' appears on the right edge.", ""),
    ("F2-004", D, "The image is framed by a violet-blue painted border; the photo is slightly blurred.", "The image is framed by a violet-blue painted border."),
    ("F2-005", D, " Part of a gold frame is visible on the right.", ""),
    ("F2-006", D, " The photo is slightly blurred and has a light reflection on the right.", ""),
    ("F2-006", "lighting", "no depicted light source; glare on right", "no depicted light source"),
    ("F2-012", D, " A signature is in the lower right and part of a gold frame shows at the bottom.", ""),
    ("F2-017", D, " A glare of light reflects in the upper center.", ""),
    ("F2-017", "lighting", "glare on surface, no depicted source", "no depicted light source"),
    ("F2-019", "lighting", "no depicted light source; glare on surface", "no depicted light source"),
    ("F2-021", "lighting", "no depicted light source; glare on surface", "no depicted light source"),
    ("F2-025", "lighting", "no depicted light source; glare in the center", "no depicted light source"),
    ("F2-027", "lighting", "no depicted light source; glare on surface", "no depicted light source"),
    ("F2-048", D, " Initials 'O.T.' appear in the lower right.", ""),
    ("F2-049", D, " Initials 'O.T.' appear in the lower right.", ""),
    ("F2-057", D, " A signature appears in the lower right.", ""),
    ("F2-074", D, " A signature is at the bottom.", ""),
    ("F2-079", D, "'NOY 2015' and 'Vientiane, Laos' are written in the lower left.", "'Vientiane, Laos' is written in the lower left."),
    ("F2-079", X, "The writing on the painting names Vientiane, Laos, and the year 2015.", "The writing on the painting names Vientiane, Laos."),
    ("F2-084", D, "; a signature is in the lower left.", "."),
    ("F3-003", D, "; glare is visible on the glass.", "."),
    ("F3-003", "lighting", "flat, with glare on glass", "flat"),
    ("F3-005", D, " Reflections of windows show on the glass.", ""),
    ("F3-005", "lighting", "flat, with reflections on glass", "flat"),
    ("F3-009", D, "The surface is glossy with a strong reflection at the bottom.", "The surface is glossy."),
    ("F3-009", "lighting", "no depicted light source; glare on surface", "no depicted light source"),
    ("F3-019", D, " A date '5.6.4.69' and a signature are written at the top right.", ""),
    ("F3-020", D, "; a number and signature appear at the top and bottom.", "."),
    ("F3-022", D, "; reflections on the glass are visible.", "."),
    ("F3-027", D, "The background is white with a reflected window on the glass.", "The background is white."),
    ("F3-041", D, "Two green tables with small colored balls on them stand", "Two green chairs with small colored balls on their seats stand"),
    ("F3-041", X, "Two green tables with small colored balls stand", "Two green chairs with small colored balls on their seats stand"),
    ("F3-041", X, "the tables seem to wait for a game to start", "the chairs seem to wait for someone to arrive"),
    ("F3-041", "composition", "two tables on a diagonal in the lower half", "two chairs on a diagonal in the lower half"),
    ("F3-045", D, " A signature and '2021' appear at the bottom.", ""),
    ("F3-058", "lighting", "flat, with glare on glass", "flat"),
    ("F3-060", "lighting", "flat, with glare on glass", "flat"),
    ("F3-061", D, " Strong reflections of a window and sky cover the glass.", ""),
    ("F3-061", "lighting", "flat, with strong reflections on glass", "flat"),
    ("F3-062", D, " Strong reflections of windows cover the glass.", ""),
    ("F3-062", "lighting", "flat, with strong reflections on glass", "flat"),
    ("F3-063", D, " Glare is visible on the glass.", ""),
    ("F3-063", "lighting", "flat, with glare on glass", "flat"),
    ("F3-064", D, " Reflections of a window obscure part of the face.", ""),
    ("F3-064", "lighting", "flat, with reflections on glass", "flat"),
    ("F3-070", "lighting", "flat, with glare on glass", "flat"),
    ("F3-071", "lighting", "flat, with glare on glass", "flat"),
    ("F3-077", D, " A signature and '2021' are in the lower left.", ""),
    ("F3-079", D, " The work is under glass with glare.", ""),
    ("F3-079", "lighting", "flat, with glare on glass", "flat"),
    ("F3-080", D, " A signature and '2016' appear in the lower left.", ""),
    ("F3-085", D, " A signature appears in the lower left.", ""),
    ("F3-086", D, "; a Georgian signature and '2019' appear at the lower right.", "."),
]
OBJ_EDITS = [("F3-041", "tables", "chairs"), ("F3-086", "signature", None)]

if __name__ == "__main__":
    for f in ("annotations.jsonl", "explanations.jsonl"):
        b = L / f.replace(".jsonl", "_v1.jsonl")
        if not b.exists():
            shutil.copy(L / f, b)
    ann = [json.loads(l) for l in (L / "annotations_v1.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    exp = [json.loads(l) for l in (L / "explanations_v1.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    A, E = {a["id"]: a for a in ann}, {e["id"]: e for e in exp}
    log = []
    for i, field, old, new in EDITS:
        rec = E[i] if field == X else A[i]
        assert old in rec[field], (i, field, old)
        rec[field] = rec[field].replace(old, new, 1)
        log.append([i, field, old.strip(), new.strip()])
    for i, old, new in OBJ_EDITS:
        objs = A[i]["objects"]
        assert old in objs, (i, old)
        A[i]["objects"] = [new if o == old else o for o in objs if not (o == old and new is None)]
        log.append([i, "objects", old, new or ""])
    (L / "annotations.jsonl").write_text("".join(json.dumps(a, ensure_ascii=False) + "\n" for a in ann), encoding="utf-8")
    (L / "explanations.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in exp), encoding="utf-8")
    with (L / "cleanup_log.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "field", "removed_or_old", "new"])
        w.writerows(log)
    print(len(log), "edits on", len({r[0] for r in log}), "records")

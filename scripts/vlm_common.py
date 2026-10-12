import csv, json, os
from pathlib import Path
import torch
from PIL import Image, ImageOps
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
L = ROOT / "labels"
BASE = "Qwen/Qwen2.5-VL-3B-Instruct"
SIDE = int(os.environ.get("VLM_SIDE", "448"))
FIELDS = ["description", "theme", "mood", "objects", "style_guess", "medium_guess",
          "era_guess", "lighting", "composition", "explanation"]
PROMPT = (
    "You are cataloguing an artwork. Look only at what is visible. Return one JSON object with keys: "
    "description (2-4 sentences: subject, scene, composition, notable details), theme, mood, "
    "objects (list of visible things), style_guess, medium_guess, era_guess, lighting, composition, explanation (4-6 plain-language sentences for a student or museum visitor: what is happening, what to notice, how it is made, and possible meaning or symbolism worded as interpretation; never name the artist or title). "
    "Never state an artist, date or title as fact; mark uncertain guesses with '(guess)'."
)
SCHEMA = os.environ.get("VLM_SCHEMA", "v2")
if SCHEMA == "v3":
    FIELDS = ["photo_artifacts"] + FIELDS
    PROMPT = PROMPT.replace("Return one JSON object with keys: ",
        "Return one JSON object with keys: photo_artifacts (anything in the photo that is not part of the artwork, such as "
        "reflections, glare, frame, mat or blur, with where it is; 'none' if nothing), ") +         " Describe only the artwork itself in every field except photo_artifacts."


def records():
    recs = [json.loads(l) for l in (L / "labels.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ann = {json.loads(l)["id"]: json.loads(l) for l in (L / "annotations.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    with (L / "splits.csv").open(encoding="utf-8") as f:
        split = {r["id"]: r["split"] for r in csv.DictReader(f)}
    for r in recs:
        r["crop"] = ann[r["id"]].get("crop")
        r["split"] = split[r["id"]]
    return recs


def load_image(path, crop=None, side=SIDE):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if crop:
        w, h = im.size
        im = im.crop((int(crop[0] * w), int(crop[1] * h), int(crop[2] * w), int(crop[3] * h)))
    im.thumbnail((side, side))
    return im


def target(rec):
    return json.dumps({k: rec[k] for k in FIELDS}, ensure_ascii=False)


def messages(img, answer=None):
    m = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
    if answer is not None:
        m.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return m


def load(adapter=None):
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE, quantization_config=q, dtype=torch.bfloat16, device_map="cuda:0")
    proc = AutoProcessor.from_pretrained(BASE, min_pixels=128 * 28 * 28, max_pixels=SIDE * SIDE)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return model, proc


def parse(text):
    s, e = text.find("{"), text.rfind("}")
    try:
        return json.loads(text[s:e + 1])
    except Exception:
        return None

import os
os.environ.setdefault("VLM_SIDE", "560")
os.environ.setdefault("DB_SET", "full")
os.environ.setdefault("CLIP_DEVICE", "cpu")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import time
import torch
from db_index import embed, load_db
from titles import artist_centroids, load_gallery, text_model, title_cascade
from verified import VerifiedAnalyzer
from vlm_common import ROOT

DESC = ROOT / "models" / "qwen_lora_v2"
TITLE = ROOT / "models" / "qwen_title_test_ft"
TITLE_SIDE = 336


@torch.no_grad()
def adapter_titles(model, proc, img, artist, n=10):
    q = f"Give this artwork a title in the style of {artist}. Answer with the title only." if artist else \
        "Give this artwork a title. Answer with the title only."
    m = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": q}]}]
    text = proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=[img], return_tensors="pt").to(model.device)
    ids = model.generate(**inp, max_new_tokens=24, num_beams=n, num_return_sequences=n, do_sample=False)
    outs = [t.strip().strip('"“”').strip() for t in proc.batch_decode(ids[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)]
    return list(dict.fromkeys(t for t in outs if t))


class Interpreter:
    def __init__(self):
        self.an = VerifiedAnalyzer(str(DESC))
        self.model, self.proc = self.an.model, self.an.proc
        self.model.load_adapter(str(TITLE), adapter_name="title")
        self.model.set_adapter("default")
        load_db()
        load_gallery()
        artist_centroids()
        text_model()
        embed([__import__("PIL.Image", fromlist=["Image"]).new("RGB", (64, 64))])

    def run(self, photo, artist=None, step=lambda name: None):
        t0 = time.time()
        step("reading")
        self.model.set_adapter("default")
        res = self.an.analyze(photo)
        if not res.get("pred"):
            return {"error": "The model could not read this image.", "analysis": res}
        W, H = photo.size
        b = res["crop"]
        painting = photo.crop((int(b[0] * W), int(b[1] * H), int(b[2] * W), int(b[3] * H)))
        step("searching")
        q = embed([painting])[0]
        small = painting.copy()
        small.thumbnail((TITLE_SIDE, TITLE_SIDE))
        cache = {}

        def extra(a):
            if a not in cache:
                self.model.set_adapter("title")
                try:
                    cache[a] = adapter_titles(self.model, self.proc, small, a)
                finally:
                    self.model.set_adapter("default")
            return cache[a]

        step("titling")
        t = title_cascade(self.model, self.proc, q, res["pred"], res["objects"], res["palette"], artist=(artist or "").strip() or None,
                          use_era=False, evocative=True, top=5, extra=extra, writer=True, artistic=True)
        self.model.set_adapter("default")
        return {"analysis": res, "painting": painting, "title": t, "seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":
    import json, sys
    from autocrop import open_photo
    it = Interpreter()
    print("loaded, gpu GB", round(torch.cuda.memory_allocated() / 1e9, 2), flush=True)
    for path in sys.argv[1:]:
        torch.cuda.reset_peak_memory_stats()
        out = it.run(open_photo(path), step=lambda s: print("..", s, flush=True))
        t = out["title"]
        print(path, out["seconds"], "s | peak GB", round(torch.cuda.max_memory_allocated() / 1e9, 2))
        print(" title:", [x["title"] for x in t["titles"]], "| path", t["path"], "|", t["why"])
        print(" objects:", [(o["object"], o["status"], o["box"] is not None) for o in out["analysis"]["objects"]])
        print(" palette:", out["analysis"]["palette"][:3])

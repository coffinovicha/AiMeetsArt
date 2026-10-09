# AI MEETS ARTS — Artwork Interpreter

Hackathon: "AI MEETS ARTS: Collect. Label. Train. Interpret."

## Environment
- Windows, RTX 4050 6 GB, PowerShell.
- Always run Python with `py` (CUDA PyTorch). `python` is a different install without torch.
- Style: terse answers, no inline comments in code, copy-paste-ready commands.

## Goal
App: artwork image in →
1. Description (subject, scene, composition, mood/theme, notable details).
2. Palette: 5–6 dominant colors, hex + %, k-means on pixels.
3. Theme (from description + palette).
4. Extras: objects, style/medium guess, era guess, lighting.
5. Title via cascade, output states path + matched artist/image + similarity:
   - a. Same artist in DB → learn that artist's naming pattern → title in that pattern.
   - b. Else most similar DB image (CLIP + FAISS cosine) above threshold → title in that artist's pattern.
   - c. Else → title from description, theme, palette, objects.

## Data
- Labeling set: the 195 painting photos in the 3 `WhatsApp Unknown 2026-10-09 at …` folders, as listed in the `Paintings` sheet of `Paintings_Metadata.xlsx`. The `Skipped` sheet (labels, floors, signature close-ups) is excluded.
- `Paintings_Metadata.xlsx` is read-only. Its artist/title from wall labels (82 paintings) are ground truth: copy into records as `known_artist`, `known_title`, `known_year`; never overwrite or reinterpret them.
- Artwork DB: `artificio_data/` (copied from `AiMeetsArts(OLD)/artificio_data`). Ignore everything else in `AiMeetsArts(OLD)`.

## Step 1 — Labels
- `labels/labels.jsonl`, one record per image: `id, file, known_artist, known_title, known_year, description (2–4 sentences), theme, mood, objects, style_guess, medium_guess, era_guess, lighting, composition, palette, title_guess`.
- Describe only what is visible. Guesses are guesses; never state artist/date/title as fact unless from the xlsx.
- Palette by script (k-means k=6), merged into records.
- Batches of ~20; resumable (skip labeled ids). After batch 1, show 3 records and wait for approval.
- Split 70/15/15, fixed seed → `labels/splits.csv`.

## Step 2 — DB index
- Inspect `artificio_data` structure/metadata and report before coding.
- `db_index`: open_clip embeddings (fp16), FAISS index, row → file/artist/title table; titles grouped by artist.

## Step 3 — Models
- Description: fine-tuned small captioner (Florence-2-base / BLIP) vs vision-LLM API — recommend, wait for choice.
- Objects: Florence-2 OD or OWLv2.
- Titles: LLM with few-shot real titles of the artist — ask which LLM/API key.
- Tune similarity threshold on val; report a/b/c usage and near-threshold examples.

## Step 4 — Eval (test split)
- Descriptions vs labels (CLIP text sim or BLEU/ROUGE) + 10 side-by-side.
- 10 titles with path used, judged against artist pattern.

## Step 5 — Gradio app
Upload → description, palette swatches, theme, objects with boxes, title + path explanation, top-5 similar DB images with scores.

## Rules
- Ask before expensive-to-redo decisions (model choice, paths, API usage).
- Everything resumable: skip already labeled/embedded items.
- Check GPU memory before training; pick batch size that fits 6 GB.

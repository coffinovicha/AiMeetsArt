# Report update: featured examples, app polish, promo video

To be added to the technical report's "The app" section and the reproducibility table when the docs connector is back.

## Featured examples

The app opens with six featured paintings chosen from full runs of the app pipeline on 12 validation and test candidates (`scripts/prepare_examples.py`, results in `outputs/app_examples/`). Candidates were ranked by meaning similarity to the hand labels, confirmed objects with boxes, no unclear subject, and title quality. Each was then read in full; paintings whose analysis contained a visible error were dropped.

| Featured painting | Title the system wrote | Path | What it shows |
| --- | --- | --- | --- |
| Ethiopian-style icon, saint on horseback (F3-073) | Heroic Saint | c | 6 objects boxed; the checks removed "The Resurrection", "Horsemen of the Apocalypse", "The Archangel Michael"; closest work Vasnetsov's *Georgy Victorious* at 0.68, below the 0.70 threshold |
| Lake village with stilt houses (F2-086) | Landscape with Boats | c | 4 confirmed objects, accurate description |
| Two angels (F1-028) | Two Angels with Fruit | c | the checks removed "bread", "cake", "cherubs" |
| Egyptian papyrus queen (F3-064) | Portrait of a Queen in Profile | b | the checks removed "Tutankhamun" and "Pharaoh" |
| Girl in blue (F2-041) | The Girl with Dandelions | a | catalogue match: the photo is a titled work in our gallery (similarity 1.00) |
| Court doll (F2-050) | Untitled — Doll in Gold | b | the Untitled-plus-name rule |

Dropped after reading: the elephant mandala (counted eight roundels where there are twelve), the cattle herd ("running"), the papyrus goddess (one figure described where there are two), the street café ("Awning Variations"), the papyrus god and queen (most objects unconfirmed).

Featured results are stored and shown instantly, labelled "prepared example, computed in 89 s"; any other photo runs the full pipeline.

## App polish

- While it works, the uploaded photo is shown with a scanning light and three live stages with a running clock.
- The result animates in: the exhibit rises, pins pop in one by one, the palette strip grows.
- Object pins sit at the centre of each object's box and are nudged apart when they would overlap; unconfirmed objects use hollow pins.
- Inline flag chips were removed. In the description and explanation, unconfirmed objects are amber, objects likely not present are red, and sentences of background context (not seen in the image) are blue; hovering shows the note.
- Two gaps found while preparing the demo were closed: two-letter words are now checked like any other ("Eye of Ra" had passed), and the context detector now catches phrasings like "were once used" and "was traditionally worn".

## Promo video

`demo/artwork_interpreter_promo.mp4`: 64 seconds, 1920×1080, H.264, built only from real system output for the saint icon, plus a montage of five featured paintings, the test-split results (0 of 143 titles with an invented word; meaning match 0.69 against 0.62 for the base model; 18 of 30 descriptions closest to their own painting against 12) and the app itself. It was rendered frame by frame from an HTML timeline in headless Edge and encoded with ffmpeg; the source is in `demo/promo_source/` (`py render_promo.py full`, then ffmpeg).

## Reproducibility rows

| Step | Script | Output |
| --- | --- | --- |
| Featured examples | `prepare_examples.py` | `outputs/app_examples/*.pkl`, `featured.json` |
| Promo video | `demo/promo_source/render_promo.py` | `demo/artwork_interpreter_promo.mp4` |

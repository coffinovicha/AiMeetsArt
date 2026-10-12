### Title cascade: database search and similarity threshold

The search method works: it finds same-artist and same-artwork matches whenever the artist is in the collection. It finds almost none for our photos in WikiArt because WikiArt does not contain 43 of the 47 gallery artists. A database that includes local artists would let paths a and b fire on these paintings.

**Artwork database.** 25,480 WikiArt artworks by 1,206 artists, all titled (135 "Untitled"); 23,702 have image embeddings, stored for two CLIP models. The DFN5B ViT-H/14 vectors are reproduced most closely when re-embedded (cosine 0.97–0.99), so the search uses that model. Picasso, Chagall and Dalí are the only gallery artists in the database (4 paintings); none of the exact gallery artworks is in it.

**Threshold from the database itself.** For every database artwork, the nearest other artwork was checked for the same artist:

| Similarity of nearest neighbor | Same artist |
| --- | --- |
| 0.75–0.80 | 26% |
| 0.80–0.85 | 35% |
| 0.85–0.90 | 49% |
| 0.90–0.95 | 67% |
| ≥ 0.95 | 92% |

Path b (borrow the matched artist's naming pattern) is set at 0.85, where a match is by the same artist about as often as not; the near-duplicate rule for path a is set at 0.95.

**Our photos against WikiArt.** The best WikiArt match per gallery photo has a median similarity of 0.68 and a maximum of 0.83; no photo reaches 0.85, so on the gallery set titles come from path a (4 known database artists) or path c.

**The same search within our own 195 photos:**

| Measure | Within our photos | Against WikiArt |
| --- | --- | --- |
| Best match per photo, median | 0.79 | 0.68 |
| Best match per photo, maximum | 0.96 | 0.83 |
| Photos with a match ≥ 0.85 | 30% | 0% |
| Same artwork photographed twice (3 pairs) | 0.95, 0.97, 0.98 | n/a |
| Same-artist pairs, median similarity | 0.88 (108 pairs) | n/a |
| Different-artist pairs, median similarity | 0.60 (3,212 pairs) | n/a |
| Nearest photo is by the same artist (52 photos whose artist has another work) | 34 of 52 (65%; chance 5%) | n/a |
| Matches ≥ 0.85 that are by the same artist | 19 of 20 (95%) | n/a |

All three repeat photos of the same artwork score above 0.95, so the near-duplicate rule would recognize them. At the 0.85 path-b threshold, 19 of 20 within-set matches are by the same artist: the threshold chosen on WikiArt also holds on our photos. The one miss (F1-082, Vakho Bugadze) matched an Ushangi Khumarashvili abstract at 0.875.

**What limits matching is coverage, not the method.** WikiArt is a large database, but it holds 4 of our 47 gallery artists. When an artist's other works are present, as within our own set, the same search finds them two times out of three and is right 95% of the time above the threshold. Adding the gallery artists, or a local artist collection, to the database would make paths a and b usable for these paintings.

**Caveats.** Ushangi Khumarashvili accounts for 14 of the 52 photos in the same-artist test, and his works share a consistent abstract style. Photos taken in the same exhibition share lighting and camera, which may raise within-set similarity somewhat compared with a database of clean scans.

**Explored, not adopted.** A hand review of 86 photos' best WikiArt matches found that the highest raw scores often come from "hub" images that sit close to every expressive abstract (Luis Dourdil's *Painting* is the top match for 12 photos). Matches that stand far above a photo's similarity to the whole database (z-score ≥ 4.5) were all genuine, for example the Maiden's Tower photo matched *Kızkulesi (Maiden's Tower)*. The cascade keeps the plain similarity threshold; the review is kept in `outputs/gallery_match_review.csv`.

Files: `scripts/db_index.py`, `outputs/db_similarity.json`, `outputs/gallery_self_similarity.json`, `outputs/gallery_matches.json`, `outputs/gallery_match_review.csv`.

### Title generation: does it learn the artist's pattern?

The final title writer follows the artist's pattern a little more than plain titles do, never copies, and is the closest of all versions to the real titles. 24 WikiArt works by 8 artists with distinct naming habits (Mondrian, Kandinsky, Klee, Monet, Toulouse-Lautrec, Konchalovsky, Saryan, Klimt) were hidden; the analysis model described each image, and titles were written with the artist's pattern (path a) and without it (path c). A title counts as traced to its artist when its meaning is closest to that artist's other titles among the 8.

| Version | Traced to the right artist | Meaning similarity to the real title | Copies |
| --- | --- | --- | --- |
| 1: 6 candidates, pattern summary | 8 / 24 | 0.32 | 0 |
| 2: 10 candidates, stronger style ranking | 10 / 24 | 0.30 | 0 |
| 3: + title templates | 13 / 24 | 0.33 | 0 |
| 4: + template-fit and name checks (final) | 11 / 24 | 0.34 | 0 |
| Plain titles (no pattern) | 8 / 24 | 0.32 | 0 |
| The real titles (ceiling) | 12 / 24 | n/a | n/a |

**Templates.** Each artist's frequent title openings are extracted from their real titles (Klimt and Saryan "Portrait of ...", Konchalovsky "Still Life. ...", Toulouse-Lautrec "At the ...", Khumarashvili "Abstraction", Timchenko "Untitled") and offered to the writer, which fills them with words from the verified analysis only. Openings that start with a proper name (Saryan's "Ararat ...") are excluded.

**Why version 4 traces fewer titles than version 3.** The fit checks removed two wrong titles: "Still Life. Trees and Path" for Konchalovsky's park landscape (the template did not fit; the title now comes from path c) and "Ararat's night" for a Saryan (a place not in the analysis). Every title word must now come from the analysis or be a common English word, and a template is used only when its key word matches the painting.

**Examples (version 4).** Konchalovsky, *Still Life. Lilacs in a tub*: "Still Life. Lilacs on a table". Mondrian, *Nature died with Sunflower*: "Still Life with Sunflowers". Mondrian, *Church tower at Domburg*: "Church Tower at Twilight". Klimt, *Female organ player*: "Lady playing piano".

**Remaining weakness.** Formulas are not always used where they fit: Klimt's portraits get "Girl in White Bonnet and Dress" instead of his "Portrait of a girl" pattern. Even the real titles are traced to their artist only half the time, because many artists title works with similar words.

**Other rules.** Habitual titles (an artist who titles every work "Abstraction" or "Untitled") may be reused; titles an artist used only once count as copies and are rejected. Titles may not contain dates, the artist's name, objects that verified mode did not confirm, or words outside the analysis that look like names or places.

Files: `scripts/titles.py`, `scripts/title_pattern_test.py`, `outputs/title_pattern_test_run1.json` to `run4.json`.

### Searching the full WikiArt (103,250 works) and our gallery together

Quadrupling WikiArt did not add a single path-b match for our photos: what limits matching is that our artists are not in WikiArt, not the size of the database. Every image is searched against both collections: the full Artificio/WikiArt set (103,250 works, 2,313 artists, embedded with the same DFN5B ViT-H model) and the 82 titled works of our gallery, leaving out the photo itself and any repeat photo of the same artwork.

| | WikiArt subset | Full WikiArt |
| --- | --- | --- |
| Artworks / artists | 25,480 / 1,206 | 103,250 / 2,313 |
| Best WikiArt match for our photos, median / maximum | 0.68 / 0.83 | 0.69 / 0.82 |
| Our photos with a WikiArt match ≥ 0.85 | 0 | 0 |
| Path-b threshold (nearest neighbour is the same artist > 50%) | 0.85 | 0.85 |
| Near-duplicate threshold (same artist > 90%) | 0.95 | 0.99 |

On the full set the nearest neighbour of a database work is by the same artist 42% of the time at 0.80–0.85, 58% at 0.85–0.90, 77% at 0.90–0.95 and 91% above 0.95, so the 0.85 rule is unchanged.

**Routing for all 195 photos** (full WikiArt plus our gallery):

| Thresholds | Artist information | Path a | Path b | Path c |
| --- | --- | --- | --- | --- |
| 0.85 for both collections | Artist from the wall label, where there is one | 54 | 7 | 134 |
| 0.85 for both collections | No artist given (the usual app case) | 0 | 27 | 168 |
| 0.80 for our gallery, 0.85 for WikiArt | Artist from the wall label | 54 | 13 | 128 |
| 0.80 for our gallery, 0.85 for WikiArt | No artist given | 0 | 41 | 154 |

Within our gallery, matches at 0.80–0.85 are by the same artist two times out of three, which is the case for the lower gallery threshold.

The best match comes from our own gallery for 152 of the 195 photos. Path a needs the artist's other titled works: 82 photos have a wall-label artist, but only 54 of those artists have another titled work in either collection. Georgian painters present in the full WikiArt (Niko Pirosmani, 218 works; David Kakabadze, 61) were not the closest match for any of our photos. One related match did appear: our photo of Picasso's *Woman in a Fish Hat* matched his *Seated Woman with Fish* (0.814), from the same series.

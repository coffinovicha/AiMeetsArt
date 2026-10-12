### Retrain v3: the model's own mistakes as look-alike negatives, and a photo-artifacts field

The v3 retrain did not improve the model. Training on its own mistakes did not transfer to new look-alike pairs, and the new artifacts field learned to answer "none". v2 stays the final model.

**What changed from v2 (one run, all other settings equal):**

- **Look-alike negatives from the model's own mistakes.** v2 described the 136 training photos (batched, 41 s per 4 photos). Its objects were compared with the labels, giving 160 candidate mistakes on 87 photos. A hand check against each label kept 42 clear mistakes on 32 photos, for example "chalice" for a panpipe, "cat" for a coat, "ladder" for a dagger, "monkey" and "eagle" for a lion, "lamppost" for a minaret, and invented goddesses ("Isis", "Nephthys"). Synonyms, objects that are probably present but unlisted, and false negatives (such as "woman" for a ballerina) were removed. Each pair became a yes/no training item, such as "panpipe: yes; chalice: no". An automatic version that spread each mistake to every photo with the same true object produced 629 negatives and was rejected, because it would teach false "no" answers (for example "clouds: no" on landscapes whose labels simply omit the sky).
- **A photo_artifacts field instead of deleting artifacts.** The reflection and glare notes removed in the v2 cleanup were restored as a separate first field, for example "reflection of the photographer in the glass, between the two faces" (39 photos; the other 156 say "none"). The idea was to let the model set reflections aside before describing the painting.

| Validation, 29 photos, verified mode | v2 (final) | v3 |
| --- | --- | --- |
| Wrong objects named | 19 of 101 | 23 of 101 |
| Wrong objects shown without a mark | 11 | 14 |
| Descriptions: correct / minor / invented / wrong subject | 8 / 19 / 1 / 1 | 7 / 17 / 3 / 2 |
| Look-alike traps answered "no" (direct yes/no probe) | 9 of 17 | 7 of 17 |
| Photos with artifacts whose artifacts were reported | n/a | 1 of 9 |
| Signature, date or glass sentences | 0 | 0 |
| BERTScore F1 / meaning similarity / objects F1 | 0.369 / 0.694 / 0.364 | 0.353 / 0.691 / 0.290 |

**Direct look-alike probe.** Both models were asked yes/no questions about the exact validation confusions. On the pomegranate still lifes both answer "pomegranates: yes" and "apples: yes"; on the backgammon players both answer "women: yes". The models do not choose between look-alikes; they accept both. 42 specific word pairs taught those pairs, not a general ability to separate similar shapes, which is consistent with the frozen vision encoder producing nearly the same features for both readings.

**Why the artifacts field failed.** 156 of 195 labels say "none", so the field learned the majority answer, the same label-frequency prior seen with signatures. v3 reported artifacts for 1 of the 9 validation photos that have them, and still described the photographer's reflection in F1-036 as "a small figure in a white shirt and dark cap".

**What v3 got right.** It kept v2's gains: no signature, date or glass sentences. F1-009 is now described correctly as a row of five colored round shapes, and F3-073's explanation names the Saint George legend.

**Conclusion.** Changing the training text has reached its limit for the remaining errors. Both cleanups that worked (removing signature habits) changed what the model writes; the errors left are about what it sees. The next step that can change them is adapting the vision encoder (LoRA on its last blocks) or a stronger base model, both of which need a memory check first.

**Operational note.** v3 trained in about 3 hours, against 1.5 hours for v2, with the same peak memory (5.7 GB). Other applications held more GPU memory during this run, so training spilled into shared system memory; closing GPU-heavy applications before training avoids this.

Files: `scripts/predict_batch.py`, `scripts/mine_negatives.py`, `scripts/add_artifacts.py`, `scripts/probe_yesno.py`, `labels/hard_negatives_curated.json`, `outputs/preds_v2_train.jsonl`, `outputs/preds_verified3_val.jsonl`, `outputs/probe_v2.json`, `outputs/probe_v3.json`, `outputs/manual_review_v3_val.csv`, `outputs/object_review_val.csv` (rows tagged verified3).

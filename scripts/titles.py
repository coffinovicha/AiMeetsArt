import difflib, re
from collections import Counter
from functools import lru_cache
import numpy as np
import torch
from db_index import load_db
from verify_objects import head

B_THRESHOLD = 0.70
HABIT_B_MIN = 0.85
DUP_THRESHOLD = 0.99
N_CAND = 10
STOP = {"the", "a", "an", "of", "in", "on", "with", "and", "at", "to", "for", "by", "from", "no", "de", "la", "le", "les", "du",
        "des", "der", "die", "das", "el", "los", "del", "di", "il", "et", "en", "und"}
FOREIGN = {"de", "la", "le", "les", "du", "des", "der", "die", "das", "el", "los", "del", "di", "il", "et", "und", "avec", "dans", "sur"}
COLORS = {"black": (20, 20, 20), "white": (240, 240, 240), "grey": (128, 128, 128), "red": (200, 30, 30), "dark red": (120, 20, 20),
          "orange": (240, 140, 30), "yellow": (240, 220, 50), "ochre": (200, 150, 60), "brown": (120, 80, 40), "beige": (225, 210, 180),
          "pink": (240, 160, 180), "purple": (120, 50, 140), "violet": (160, 120, 200), "blue": (40, 80, 200), "navy": (20, 30, 90),
          "light blue": (150, 190, 230), "teal": (30, 130, 130), "green": (50, 150, 60), "olive": (120, 120, 50), "dark green": (20, 70, 40)}


def color_name(hexv):
    rgb = tuple(int(hexv[i:i + 2], 16) for i in (1, 3, 5))
    return min(COLORS, key=lambda c: sum((a - b) ** 2 for a, b in zip(COLORS[c], rgb)))


def norm(t):
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def clean_titles(rows):
    out = []
    for r in rows:
        t = r["title"].strip()
        if t and "�" not in t and norm(t) != norm(r["artist"]):
            out.append(t)
    return out


def pattern_stats(titles):
    words = [len(t.split()) for t in titles]
    low = [t.lower() for t in titles]
    first = Counter(t.split()[0] for t in low if t.split())
    vocab = Counter(w for t in low for w in re.findall(r"[a-z\-]+", t) if w not in STOP and len(w) > 2)
    share = lambda f: sum(map(f, low)) / len(low)
    return {
        "n": len(titles),
        "words_median": int(np.median(words)), "words_p10": int(np.percentile(words, 10)), "words_p90": int(np.percentile(words, 90)),
        "untitled": share(lambda t: "untitled" in t),
        "numbered": share(lambda t: bool(re.search(r"\d|\bno\.|\b[ivxlc]+\b$", t))),
        "portrait": share(lambda t: "portrait" in t),
        "colon_or_paren": share(lambda t: bool(re.search(r"[:(]", t))),
        "starts_the": share(lambda t: t.startswith("the ")),
        "foreign": share(lambda t: bool(set(t.split()) & FOREIGN)),
        "top_first_words": [w for w, _ in first.most_common(4)],
        "top_words": [w for w, _ in vocab.most_common(6)],
    }


PERSON_GENERIC = {"a", "an", "the", "young", "old", "little", "girl", "woman", "man", "boy", "child", "lady", "gentleman", "children",
                  "women", "men", "girls", "boys", "mother", "father", "family", "couple", "friend", "friends", "artist", "musician",
                  "dancer", "self", "seated", "standing", "sitting", "with", "in", "and"}


def _tokens(title):
    return re.findall(r"[\w'\-]+|[.,:;]", title)


def extract_templates(titles, min_share=0.05, k=5):
    toks = [_tokens(x) for x in titles]
    cnt, first = Counter(), {}
    for orig in toks:
        low = tuple(w.lower() for w in orig)
        for n in (1, 2, 3):
            if len(low) > n or (n == 1 and len(low) == 1):
                pre = low[:n]
                cnt[pre] += 1
                first.setdefault(pre, orig[:n])
    N = max(1, len(titles))
    cand = [(pre, c / N) for pre, c in cnt.items() if c >= 3 and c / N >= min_share
            and not (len(pre) == 1 and pre[0] in STOP) and any(w.isalpha() for w in pre)]
    vocab = text_model().tokenizer.vocab
    cand = [(pre, s) for pre, s in cand if all(w in STOP or not w.isalpha() or w in vocab for w in pre)]
    keep = [(pre, s) for pre, s in cand if not any(len(q) > len(pre) and q[:len(pre)] == pre and qs >= 0.8 * s for q, qs in cand)]
    keep.sort(key=lambda x: -x[1])
    out = []
    for pre, s in keep[:k]:
        words = first[pre]
        text = ""
        for w in words:
            text += w if w in ".,:;" else (" " + w if text else w)
        out.append({"prefix": text, "low": " ".join(pre), "share": round(s, 2)})
    return out


def template_of(title, templates):
    low = " ".join(w.lower() for w in _tokens(title))
    hits = [tp for tp in templates if low == tp["low"] or low.startswith(tp["low"] + " ")]
    return max(hits, key=lambda tp: len(tp["low"])) if hits else None


def describe_pattern(s):
    parts = [f"length {s['words_p10']}-{s['words_p90']} words (usually {s['words_median']})"]
    for key, text in [("untitled", "are 'Untitled'"), ("numbered", "contain a number"), ("portrait", "are portraits ('Portrait of ...')"),
                      ("colon_or_paren", "use a colon or brackets"), ("starts_the", "start with 'The'"), ("foreign", "are not in English")]:
        if s[key] >= 0.15:
            parts.append(f"{round(100 * s[key])}% {text}")
    parts.append("frequent first words: " + ", ".join(s["top_first_words"]))
    parts.append("frequent words: " + ", ".join(s["top_words"]))
    return "; ".join(parts)


@lru_cache(maxsize=1)
def text_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")


def enc(xs):
    return text_model().encode(list(xs), normalize_embeddings=True)


@lru_cache(maxsize=1)
def artist_centroids():
    from db_index import DB, DB_SET
    _, _, meta, by_artist = load_db()
    cache = DB / ("full" if DB_SET == "full" else "") / f"title_mpnet_{DB_SET}.npy"
    if cache.exists():
        T = np.load(cache)
    else:
        T = enc([m["title"] or m["artist"] for m in meta]).astype("float16")
        np.save(cache, T)
    names = sorted(by_artist)
    C = []
    for a in names:
        ks = [k for k in by_artist[a] if clean_titles([meta[k]])] or by_artist[a]
        v = T[ks].astype("float32").mean(0)
        C.append(v / np.linalg.norm(v))
    return names, np.array(C)


@lru_cache(maxsize=1)
def load_gallery():
    from vlm_common import ROOT, records
    ids = (ROOT / "outputs" / "gallery_vith_ids.txt").read_text(encoding="utf-8").split()
    V = np.load(ROOT / "outputs" / "gallery_vith.npy").astype("float32")
    R = {r["id"]: r for r in records()}
    wiki = list(load_db()[3])
    items = []
    for k, i in enumerate(ids):
        r = R[i]
        if r["known_artist"] and r["known_title"]:
            a = r["known_artist"].strip()
            items.append({"id": i, "artist": _closest(a, wiki) or a, "title": r["known_title"].strip(), "vec": V[k],
                          "year": year_of(r["known_year"])})
    return items


ERA_WINDOW = 10
ERA_BONUS = 0.01


def year_of(v):
    m = re.search(r"(1[4-9]|20)\d\d", str(v or ""))
    return int(m.group()) if m else None


def era_range(year=None, era_guess=None):
    if year:
        return (year - ERA_WINDOW, year + ERA_WINDOW), "known year"
    g = (era_guess or "").lower()
    c = [int(x) for x in re.findall(r"(\d{2})(?:st|nd|rd|th)", g)]
    if not c:
        return None, None
    lo, hi = (min(c) - 1) * 100, max(c) * 100
    if "late" in g:
        lo += 50
    if "early" in g and len(c) == 1:
        hi -= 50
    return (lo, hi), "model's era guess"


def _closest(name, names):
    hit = difflib.get_close_matches(name, names, n=1, cutoff=0.8)
    if hit:
        return hit[0]
    last = name.split()[-1].lower()
    cand = [a for a in names if a.split()[-1].lower() == last]
    return cand[0] if len(cand) == 1 else None


def match_artist(name):
    if not name:
        return None
    names = sorted(set(load_db()[3]) | {g["artist"] for g in load_gallery()})
    return _closest(name.strip(), names)


def search_all(qvec, exclude_rows=(), exclude_ids=(), k=5, era=None):
    from db_index import search
    _, _, meta, _ = load_db()
    hits = [dict(h, source="WikiArt", year=year_of(h["year"])) for h in search(qvec, k + 10 + len(exclude_rows)) if h["row"] not in set(exclude_rows)]
    for g in load_gallery():
        if g["id"] not in set(exclude_ids):
            hits.append({"source": "our gallery", "id": g["id"], "artist": g["artist"], "title": g["title"], "year": g["year"],
                         "score": float(g["vec"] @ qvec)})
    for h in hits:
        bonus = 0.0
        if era and h.get("year"):
            lo, hi = era
            bonus = ERA_BONUS if lo <= h["year"] <= hi else (-ERA_BONUS if h["year"] < lo - 50 or h["year"] > hi + 50 else 0.0)
        h["era_bonus"] = bonus
    return sorted(hits, key=lambda h: -(h["score"] + h["era_bonus"]))[:k]


def artist_examples(artist, qvec, exclude=(), exclude_ids=(), k=12, y0=None):
    _, E, meta, by_artist = load_db()
    pool = [(meta[j]["title"], float(E[j] @ qvec) if qvec is not None else 0.0, year_of(meta[j]["year"]), meta[j])
            for j in by_artist.get(artist, []) if j not in set(exclude)]
    pool = [(t, s, y) for t, s, y, m in pool if clean_titles([m])]
    pool += [(g["title"], float(g["vec"] @ qvec) if qvec is not None else 0.0, g["year"]) for g in load_gallery()
             if g["artist"] == artist and g["id"] not in set(exclude_ids)]
    pool.sort(key=lambda x: -x[1])
    if y0 is None:
        ys = [y for _, _, y in pool[:5] if y]
        y0 = int(np.median(ys)) if ys else None
    period, used = pool, None
    if y0:
        for w in (ERA_WINDOW, 2 * ERA_WINDOW):
            near = [x for x in pool if x[2] and abs(x[2] - y0) <= w]
            if len(near) >= 6:
                period, used = near, w
                break
    seen, ex = set(), []
    for t, _, _ in period:
        if norm(t) not in seen:
            seen.add(norm(t))
            ex.append(t)
    titles = [t for t, _, _ in (period if len(period) >= 12 else pool)]
    return ex[:k], titles, {"period_year": y0, "window": used, "n_period": len(period) if used else None}


def analysis_text(pred, confirmed, doubtful, palette):
    cols = list(dict.fromkeys(color_name(c["hex"]) for c in (palette or [])[:4]))
    lines = [f"Description: {pred.get('description', '')}", f"Theme: {pred.get('theme', '')}. Mood: {pred.get('mood', '')}."]
    if confirmed:
        lines.append("Confirmed objects: " + ", ".join(confirmed) + ".")
    if cols:
        lines.append("Main colors: " + ", ".join(cols) + ".")
    if doubtful:
        lines.append("Uncertain, do not use: " + ", ".join(doubtful) + ".")
    return "\n".join(lines)


def prompt_pattern(artist, examples, stats, analysis, templates=(), meaning=None):
    tpl = ""
    if templates:
        tpl = ("\nTitle templates this artist uses: " + "; ".join(f"'{tp['prefix']} ...' ({round(100 * tp['share'])}% of titles)" for tp in templates) +
               ". If the painting fits one of these templates, fill it in, using only words from the analysis below "
               "(for example a portrait becomes 'Portrait of a girl', never an invented name).")
    return (f"You write artwork titles. Below are real titles by {artist}, taken from the works that look most like a new painting, "
            f"and a summary of how this artist titles works.\nTitles:\n" + "\n".join(f"- {t}" for t in examples) +
            f"\nPattern: {describe_pattern(stats)}.{tpl}\n\nNew painting, verified analysis:\n{analysis}\n"
            + (f"What the new painting means:\n{meaning}\n{EVOKE} Keep the artist's pattern when it applies.\n" if meaning else "") + "\n"
            f"Write {N_CAND} different titles {artist} could plausibly give the new painting. Follow the pattern: the same length, "
            "wording, language and numbering habits, and reuse the artist's typical openings and phrases (see frequent first words) "
            "whenever they fit. Base the content only on the description, theme, confirmed objects and colors. "
            "Do not copy any title from the list. Do not use names of people or places unless they appear in the description, "
            "and no dates. One title per line, no numbering, no quotation marks.")


EVOKE = ("Name exactly one confirmed object as the subject and add a mood or quality taken from the theme, mood or lighting. "
         "Do not add actions, events, places, seasons or ideas that are not in the analysis.")
GROUND_SIM = 0.7


def meaning_text(pred):
    return f"Theme: {pred.get('theme', '')}. Mood: {pred.get('mood', '')}. Lighting: {pred.get('lighting', '')}."


@lru_cache(maxsize=4096)
def _word_vec(w):
    return enc([w])[0]


def ungrounded_words(title, evidence, allowed=()):
    ev = [w for w in evidence if w not in STOP and len(w) > 2]
    E = np.array([_word_vec(w) for w in ev]) if ev else None
    out = []
    for w in re.findall(r"[a-z]+", title.lower()):
        if w in STOP or w in evidence or head(w) in evidence or w in allowed or len(w) <= 2:
            continue
        if any(len(a) >= 4 and (a.startswith(w[:max(4, min(len(w), 5))]) or w.startswith(a[:max(4, min(len(a), 5))])) and abs(len(a) - len(w)) <= 3
               for a in ev):
            continue
        if E is not None and float((E @ _word_vec(w)).max()) >= GROUND_SIM:
            continue
        out.append(w)
    return out


def prompt_free(analysis, meaning=None):
    if meaning:
        return (f"Write {N_CAND} different short titles (2-6 words) a gallery could give this painting.\n"
                f"What the painting means:\n{meaning}\n\nWhat is verified in the painting:\n{analysis}\n\n{EVOKE} "
                "Do not invent people, places, names, dates or objects that are not in the analysis. "
                "One title per line, no numbering, no quotation marks.")
    return (f"Write {N_CAND} different short titles (2-6 words) a gallery could give this painting, based only on this verified "
            f"analysis:\n{analysis}\n\nDo not invent people, places, names, dates or objects that are not in the analysis. "
            "Prefer what the painting shows or evokes. One title per line, no numbering, no quotation marks.")


@torch.no_grad()
def write(model, proc, prompt, sample=False):
    m = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], return_tensors="pt").to(model.device)
    ctx = model.disable_adapter() if hasattr(model, "disable_adapter") else torch.no_grad()
    with ctx:
        ids = model.generate(**inp, max_new_tokens=160, do_sample=False, repetition_penalty=1.1)
        raws = [proc.batch_decode(ids[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]]
        if sample:
            torch.manual_seed(0)
            ids = model.generate(**inp, max_new_tokens=160, do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.1)
            raws.append(proc.batch_decode(ids[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0])
    raw = "\n".join(raws)
    out = []
    for line in raw.splitlines():
        t = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line).strip().strip('"“”\'').strip()
        t = re.sub(r"[*_#`]+", "", t).strip()
        if t and len(t) < 90:
            out.append(t)
    return list(dict.fromkeys(out)), raw


def check(t, ctx):
    from verified import mentions
    words = t.split()
    if norm(t) in ctx.get("habitual", {}):
        return None
    if norm(t) in ctx.get("forbid_copy", set()):
        return "copies an existing title"
    if re.search(r"\b(1[5-9]|20)\d\d\b", t):
        return "contains a date"
    if ctx.get("artist") and any(p.lower() in t.lower() for p in ctx["artist"].split() if len(p) > 3):
        return "names the artist"
    for o in ctx.get("doubtful", []):
        if mentions(t, o):
            return f"uses uncertain object '{o}'"
    lo, hi = ctx.get("words", (1, 8))
    if not lo <= len(words) <= hi:
        return f"length {len(words)} outside {lo}-{hi} words"
    ev = ctx.get("evidence_words", set())
    vocab = text_model().tokenizer.vocab
    for w in re.findall(r"[A-Za-z][A-Za-z']*", t):
        w = re.sub(r"'s$", "", w.lower())
        if w and w not in STOP and w not in ev and w not in vocab and head(w) not in vocab:
            return f"uses '{w}', which is neither in the analysis nor a common word (a name or place?)"
    tp = template_of(t, ctx.get("templates", []))
    if ctx.get("grounded"):
        tpl_words = {w for x in ctx.get("templates", []) for w in x["low"].split()} | {"untitled", "abstraction", "still", "life", "portrait"}
        un = ungrounded_words(t, ctx.get("ground_words", set()), tpl_words)
        if un:
            return f"not supported by the analysis: {', '.join(un)}"
        if ctx.get("confirmed") and not tp and not any(mentions(t, o) for o in ctx["confirmed"]):
            return "names no confirmed object (loses the subject)"
    if tp:
        key = [w for w in tp["low"].split() if w.isalpha() and w not in STOP]
        if "portrait" in key:
            fits = bool(ev & (PERSON_GENERIC - STOP - {"seated", "standing", "sitting"} | {"portrait", "face", "figure", "person"}))
        else:
            fits = all(w in ev or head(w) in ev or any(len(a) >= 5 and (a.startswith(w[:5]) or w.startswith(a[:5])) for a in ev) for w in key)
        if not fits:
            return f"template '{tp['prefix']}' does not fit this painting"
        slot = [w.lower().strip(",.:;()'") for w in t.split()][len(tp["low"].split()):]
        slot = [w for w in slot if w and w not in STOP]
        allowed_words = ctx.get("allowed_caps", set())
        if "portrait" in tp["low"] or tp["low"].split()[-1] in {"madame", "mademoiselle", "monsieur", "mrs", "mr", "miss"}:
            bad = [w for w in slot if w not in allowed_words and w not in PERSON_GENERIC]
            if bad:
                return f"invents a person ('{' '.join(bad)}')"
        else:
            vocab = text_model().tokenizer.vocab
            bad = [w for w in slot if w not in allowed_words and w not in vocab and head(w) not in vocab]
            if bad:
                return f"uses an unknown name ('{' '.join(bad)}')"
    if "untitled" in t.lower() and ctx.get("untitled_share", 0) < 0.3:
        return "'Untitled' is not this artist's habit"
    content = [w for w in words if w.lower().strip(",.:;()'") not in STOP]
    title_case = content and sum(w[:1].isupper() for w in content) / len(content) >= 0.8
    caps = [w.strip(",.:;()'") for w in words[1:] if w[:1].isupper()]
    allowed = ctx.get("allowed_caps", set())
    names = [w for w in caps if w.lower() not in allowed and w.lower() not in STOP]
    if not title_case and len(names) >= 2:
        return "looks like it names people or places"
    return None


def rank(cands, ctx):
    names, C = artist_centroids()
    V = enc(cands)
    fit = V @ enc([ctx["analysis"]])[0]
    if ctx.get("meaning"):
        fit = 0.5 * fit + 0.5 * (V @ enc([ctx["meaning"]])[0])
    own = None
    if ctx.get("artist_titles"):
        own = enc(ctx["artist_titles"]).mean(0)
        own /= np.linalg.norm(own)
        others = C[[k for k, a in enumerate(names) if a != ctx.get("artist")]]
    out = []
    for t, v, f in zip(cands, V, fit):
        reason = check(t, ctx)
        row = {"title": t, "fit": round(float(f), 3)}
        if own is not None:
            mine, rest = float(own @ v), others @ v
            row["pattern_rank"] = int((rest > mine).sum()) + 1
            row["pattern_margin"] = round(mine - float(np.median(rest)), 3)
            if reason is None and row["pattern_margin"] <= 0:
                reason = "does not follow the artist's pattern"
        row["rejected"] = reason
        out.append(row)
    ok = [r for r in out if not r["rejected"]]
    for r in ok:
        tp = template_of(r["title"], ctx.get("templates", []))
        r["template"] = tp["prefix"] if tp else None
        r["template_bonus"] = tp["share"] if tp else 0.0
        r["habit_bonus"] = ctx.get("habitual", {}).get(norm(r["title"]), 0.0)
        n_obj = sum(1 for o in ctx.get("objects_all", []) if __import__("verified").mentions(r["title"], o))
        r["objects_named"] = n_obj
        r["literal_penalty"] = 0.03 * max(0, n_obj - 1) if (ctx.get("meaning") and not tp) else 0.0
    key = (lambda r: r["fit"] + 2 * r.get("pattern_margin", 0) + r.get("template_bonus", 0) + r.get("habit_bonus", 0)
           - r.get("literal_penalty", 0))
    return sorted(ok, key=key, reverse=True), [r for r in out if r["rejected"]]


def title_cascade(model, proc, qvec, pred, objects, palette, artist=None, exclude_rows=(), exclude_ids=(), top=3, force=None,
                  year=None, use_era=True, evocative=True, extra=None, writer=True):
    era, era_source = era_range(year, pred.get("era_guess")) if use_era else (None, None)
    hits = search_all(qvec, exclude_rows, exclude_ids, era=era)
    top1 = hits[0] if hits else None
    confirmed = [o["object"] for o in objects if o["status"] == "confirmed"]
    doubtful = [o["object"] for o in objects if o["status"] != "confirmed"]
    analysis = analysis_text(pred, confirmed, doubtful, palette)
    path, why, known, examples, all_titles, period = "c", "artist unknown or without titles, and no image above the similarity threshold", None, [], [], None
    if force != "c":
        given = match_artist(artist)
        if given:
            examples, all_titles, period = artist_examples(given, qvec, exclude_rows, exclude_ids, y0=year if use_era else None)
            if not use_era:
                examples, all_titles, period = artist_examples(given, qvec, exclude_rows, exclude_ids, y0=0)
        if given and examples:
            known, path = given, "a"
            why = f"artist '{artist}' has {len(all_titles)} titled works in our gallery and WikiArt (as '{given}')"
        elif top1 and top1["score"] >= B_THRESHOLD:
            known = top1["artist"]
            examples, all_titles, period = artist_examples(known, qvec, exclude_rows, exclude_ids, y0=(top1.get("year") if use_era else 0))
            if examples:
                path = "a" if top1["score"] >= DUP_THRESHOLD else "b"
                kind = "near-duplicate of" if path == "a" else "most similar image:"
                why = f"{kind} '{top1['title']}' by {top1['artist']} ({top1['source']}, similarity {top1['score']:.2f})"
                if artist and not given:
                    why = f"artist '{artist}' has no titled works in either collection; " + why
    else:
        why = "forced path c"
    ctx = {"analysis": analysis, "doubtful": doubtful, "artist": known}
    allowed = {w.lower() for w in re.findall(r"[A-Za-z]+", analysis)}
    evidence = allowed | {w.lower() for f in ("style_guess", "composition", "medium_guess") for w in re.findall(r"[A-Za-z]+", str(pred.get(f) or ""))}
    ctx["evidence_words"] = evidence
    meaning = meaning_text(pred) if evocative else None
    ground = {w.lower() for f in ("description", "theme", "mood", "lighting", "composition", "style_guess")
              for w in re.findall(r"[A-Za-z]+", str(pred.get(f) or ""))}
    ground |= {w.lower() for o in confirmed for w in re.findall(r"[A-Za-z]+", o)}
    ground |= {w for c in (palette or [])[:6] for w in color_name(c["hex"]).split()}
    ctx.update(meaning=meaning, objects_all=[o["object"] for o in objects], grounded=evocative, ground_words=ground, confirmed=confirmed)
    if path in ("a", "b"):
        stats = pattern_stats(all_titles)
        templates = extract_templates(all_titles)
        uses = Counter(norm(x) for x in all_titles)
        first_form = {}
        for x in all_titles:
            first_form.setdefault(norm(x), x)
        habitual = {k: c / len(all_titles) for k, c in uses.items() if c >= 2 and c / len(all_titles) >= 0.5}
        if path == "b" and top1 and top1["score"] < HABIT_B_MIN:
            habitual = {}
        ctx["habitual"] = habitual
        ctx["habitual_forms"] = [first_form[k] for k in habitual]
        ctx.update(forbid_copy={x for x, c in uses.items() if c == 1}, words=(max(1, stats["words_p10"]), stats["words_p90"] + 2),
                   untitled_share=stats["untitled"], artist_titles=all_titles, templates=templates,
                   allowed_caps=allowed | {w.lower() for t in all_titles for w in t.split()})
        prompt = prompt_pattern(known, examples, stats, analysis, templates, meaning)
    else:
        stats = None
        ctx.update(words=(1, 7), allowed_caps=allowed)
        prompt = prompt_free(analysis, meaning)
    cands, raw = write(model, proc, prompt, sample=evocative) if writer else ([], "")
    added = extra(known) if extra else []
    cands = list(dict.fromkeys(ctx.get("habitual_forms", []) + added + cands))
    ranked, rejected = rank(cands, ctx)
    for r in ranked:
        r["source"] = "habit" if r["title"] in ctx.get("habitual_forms", []) else "adapter" if r["title"] in added else "writer"
    if not ranked and path in ("a", "b"):
        added2 = extra(None) if extra else []
        cands2, raw = write(model, proc, prompt_free(analysis, meaning), sample=evocative) if writer else ([], "")
        cands2 = list(dict.fromkeys(added2 + cands2))
        ranked, rej2 = rank(cands2, {"analysis": analysis, "doubtful": doubtful, "words": (1, 7), "allowed_caps": allowed,
                                     "evidence_words": evidence, "meaning": meaning, "objects_all": [o["object"] for o in objects],
                                     "grounded": evocative, "ground_words": ground, "confirmed": confirmed})
        rejected += rej2
        for r in ranked:
            r["source"] = "adapter" if r["title"] in added2 else "writer"
        path, why = "c", why + "; no candidate passed the pattern checks, fell back to path c"
    if not ranked:
        theme = str(pred.get("theme") or "").strip()
        ranked = [{"title": theme[:1].upper() + theme[1:] if theme else "Untitled", "fit": 0.0, "fallback": True}]
        why += "; no generated title passed the checks, so the verified theme is used as the title"
    return {"path": path, "why": why, "artist": known, "match": top1, "neighbors": hits, "examples": examples, "pattern": stats,
            "templates": ctx.get("templates", []), "era": {"range": era, "source": era_source, "examples_period": period},
            "titles": ranked[:top], "rejected": rejected, "raw": raw}

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import base64, html, io, queue, re, threading, time
import gradio as gr
from autocrop import open_photo
from interpret import Interpreter
from titles import B_THRESHOLD, DUP_THRESHOLD
from vlm_common import ROOT, records

N_WORKS = "103,332"
STAGES = [("reading", "Reading the painting", "finds the painted surface, reads it three times and checks every object"),
          ("searching", "Searching the collections", f"compares it with {N_WORKS} works from WikiArt and our gallery"),
          ("titling", "Writing the title", "drafts titles, removes anything the analysis does not support, picks the best")]
FIELDS = [("theme", "Theme"), ("mood", "Mood"), ("style_guess", "Style"), ("medium_guess", "Medium"), ("era_guess", "Era"),
          ("lighting", "Lighting"), ("composition", "Composition")]
SRC = {"adapter": "title model", "writer": "writer", "habit": "artist's habit"}
PATHS = {"a": ("Artist's own titles", "the artist is known and has titled works"),
         "b": ("Most similar artwork", f"closest work scores at least {B_THRESHOLD:.2f}"),
         "c": ("The analysis alone", "no artist and no close match")}

HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;1,9..144,400;1,9..144,500&family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
"""

CSS = """
/* Layout: a gallery wall. The artwork hangs under a spotlight with numbered pins; a white wall label sits beside it;
   a catalogue sheet below holds the palette sample strip, notes, data and the title's provenance. */
:root, .dark {
  --wall: #1b2531;
  --wall-deep: #131b24;
  --wall-line: #2c3a4a;
  --wall-text: #d9e1e8;
  --wall-muted: #8fa1b3;
  --paper: #f7f8f5;
  --paper-line: #dfe2dc;
  --ink: #161a1d;
  --ink-muted: #5b6268;
  --brass: #c49a5c;
  --brass-deep: #9a7339;
  --ok: #7cc49a;
  --doubt: #e0b35a;
  --f-display: "Fraunces", "Iowan Old Style", Georgia, serif;
  --f-body: "Instrument Sans", "Segoe UI", system-ui, sans-serif;
  --f-mono: "JetBrains Mono", Consolas, ui-monospace, monospace;
  --body-background-fill: var(--wall);
  --background-fill-primary: var(--wall-deep);
  --background-fill-secondary: var(--wall);
  --block-background-fill: var(--wall-deep);
  --block-border-color: var(--wall-line);
  --block-label-background-fill: var(--wall-deep);
  --block-label-text-color: var(--wall-muted);
  --block-title-text-color: var(--wall-muted);
  --body-text-color: var(--wall-text);
  --body-text-color-subdued: var(--wall-muted);
  --input-background-fill: var(--wall);
  --input-border-color: var(--wall-line);
  --input-placeholder-color: #6f8295;
  --border-color-primary: var(--wall-line);
  --color-accent: var(--brass);
  --color-accent-soft: rgba(196, 154, 92, 0.18);
  --button-primary-background-fill: var(--brass);
  --button-primary-background-fill-hover: #d4ab6c;
  --button-primary-text-color: #1a1408;
  --button-primary-border-color: var(--brass);
  --button-secondary-background-fill: transparent;
  --button-secondary-text-color: var(--wall-text);
  --button-secondary-border-color: var(--wall-line);
  --link-text-color: var(--brass);
  --font: var(--f-body);
  --font-mono: var(--f-mono);
  color-scheme: dark;
}
body, gradio-app, .gradio-container { background: var(--wall) !important; color: var(--wall-text); font-family: var(--f-body); }
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; padding-inline: 20px !important; }
footer { display: none !important; }

#masthead { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 8px 32px; align-items: end; padding-block: 28px 18px; border-bottom: 1px solid var(--wall-line); }
#masthead h1 { font: 500 clamp(30px, 4.4vw, 52px)/1.0 var(--f-display); letter-spacing: -0.015em; margin: 0; color: var(--wall-text); }
#masthead h1 em { color: var(--brass); font-style: italic; }
#masthead p { margin: 8px 0 0; color: var(--wall-muted); max-width: 62ch; font-size: 15px; line-height: 1.5; }
#masthead .plaque { font: 500 11px/1.6 var(--f-mono); letter-spacing: 0.08em; text-transform: uppercase; color: var(--wall-muted); text-align: right; }
#masthead .plaque b { color: var(--brass); font-weight: 500; }

#bench { gap: 22px !important; align-items: flex-start !important; }
#intake { position: sticky; top: 12px; }
#intake .label-note { font: 500 11px var(--f-mono); letter-spacing: 0.08em; text-transform: uppercase; color: var(--wall-muted); margin: 2px 0 6px; }
#go { font: 600 15px var(--f-body) !important; letter-spacing: 0.02em; min-height: 46px; border-radius: 3px !important; }
#examples { background: transparent !important; border: 0 !important; padding: 0 !important; }
#examples .label, #examples > .label-wrap, #examples span[data-testid="block-label"] { display: none !important; }
#examples button { background: transparent !important; border: 1px solid var(--wall-line) !important; border-radius: 2px !important; padding: 3px !important; }
#examples button:hover { border-color: var(--brass) !important; }
#examples img { border-radius: 1px !important; object-fit: cover; }

.stage-wall { min-height: 520px; display: grid; place-items: center; padding: 40px 16px;
  background: radial-gradient(60% 55% at 50% 30%, rgba(255, 236, 200, 0.07), transparent 70%), var(--wall-deep); border: 1px solid var(--wall-line); border-radius: 3px; }
.empty { text-align: center; max-width: 46ch; color: var(--wall-muted); display: grid; gap: 14px; justify-items: center; }
.empty .frame { width: 180px; aspect-ratio: 4 / 5; border: 1px dashed var(--wall-line); display: grid; place-items: center; font: italic 400 16px var(--f-display); color: var(--wall-muted); }
.empty h2 { font: italic 400 26px/1.2 var(--f-display); color: var(--wall-text); margin: 0; }

.stages { display: grid; gap: 14px; width: min(520px, 100%); }
.stages .clock { font: 500 12px var(--f-mono); color: var(--wall-muted); letter-spacing: 0.06em; text-transform: uppercase; }
.st { display: grid; grid-template-columns: 28px minmax(0, 1fr); gap: 12px; align-items: start; opacity: 0.45; }
.st .dot { width: 14px; height: 14px; margin-top: 4px; border-radius: 50%; border: 2px solid var(--wall-muted); }
.st.done { opacity: 0.8; } .st.done .dot { background: var(--ok); border-color: var(--ok); }
.st.now { opacity: 1; } .st.now .dot { border-color: var(--brass); box-shadow: 0 0 0 0 rgba(196, 154, 92, 0.6); animation: pulse 1.6s infinite; }
.st b { display: block; font: 500 17px/1.3 var(--f-display); color: var(--wall-text); }
.st span { font-size: 13px; color: var(--wall-muted); }
@keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(196, 154, 92, 0.55); } 100% { box-shadow: 0 0 0 12px rgba(196, 154, 92, 0); } }
@media (prefers-reduced-motion: reduce) { .st.now .dot { animation: none; background: var(--brass); } }

.exhibit { display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr); gap: 0; border: 1px solid var(--wall-line); border-radius: 3px; overflow: hidden; }
.hang { position: relative; padding: 48px 36px 22px; display: grid; gap: 18px; justify-items: center; align-content: center;
  background: radial-gradient(70% 60% at 50% 18%, rgba(255, 238, 205, 0.13), transparent 72%), var(--wall-deep); }
.hang .piece { position: relative; display: inline-block; max-width: 100%; line-height: 0; box-shadow: 0 2px 0 rgba(0,0,0,0.35), 0 22px 40px rgba(0,0,0,0.5); }
.hang img { display: block; max-width: 100%; max-height: 640px; width: auto; height: auto; }
.hang svg { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
.hang svg rect { fill: rgba(196, 154, 92, 0.08); stroke: var(--brass); stroke-width: 2.2; vector-effect: non-scaling-stroke; opacity: 0; transition: opacity 0.15s ease; }
.hang svg rect.doubt { stroke: var(--doubt); stroke-dasharray: 6 5; }
.pin { position: absolute; transform: translate(-30%, -30%); width: 22px; height: 22px; border-radius: 50%; background: var(--brass); color: #1a1408;
  font: 600 11px/22px var(--f-mono); text-align: center; box-shadow: 0 1px 4px rgba(0,0,0,0.5); cursor: default; }
.pin.doubt { background: var(--doubt); }
.hang .caption { justify-self: stretch; font: 500 11px var(--f-mono); color: var(--wall-muted); letter-spacing: 0.04em; line-height: 1.4; }

.label-card { background: var(--paper); color: var(--ink); padding: 34px 34px 28px; display: grid; gap: 18px; align-content: start; min-width: 0; }
.label-card .kick { font: 500 11px var(--f-mono); letter-spacing: 0.1em; text-transform: uppercase; color: var(--ink-muted); }
.label-card h2 { margin: 0; font: italic 400 clamp(30px, 3.2vw, 44px)/1.08 var(--f-display); color: var(--ink); text-wrap: balance; letter-spacing: -0.01em; }
.label-card h2 .pre { display: block; font: 500 13px var(--f-mono); font-style: normal; letter-spacing: 0.12em; text-transform: uppercase; color: var(--brass-deep); margin-bottom: 6px; }
.label-card .by { margin-top: -8px; font-size: 12px; color: var(--ink-muted); }
.label-card .lbl-text { margin: 0; font-size: 15px; line-height: 1.6; color: #2b3136; max-width: 46ch; }
.label-card .facts { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 4px 14px; font-size: 14px; border-top: 1px solid var(--paper-line); padding-top: 14px; }
.label-card .facts dt { font: 500 11px/1.9 var(--f-mono); letter-spacing: 0.06em; text-transform: uppercase; color: var(--ink-muted); }
.label-card .facts dd { margin: 0; line-height: 1.45; }
.guess { font: 500 10px var(--f-mono); letter-spacing: 0.05em; text-transform: uppercase; color: var(--ink-muted); border: 1px solid var(--paper-line); border-radius: 2px; padding: 0 4px; margin-left: 4px; white-space: nowrap; }
.alts { display: grid; gap: 6px; border-top: 1px solid var(--paper-line); padding-top: 14px; }
.alts .row { display: flex; flex-wrap: wrap; gap: 4px 10px; align-items: baseline; }
.alts .t { font: italic 400 18px/1.3 var(--f-display); }
.chip { font: 500 10px var(--f-mono); letter-spacing: 0.05em; text-transform: uppercase; padding: 2px 6px; border-radius: 2px; background: #e9ece6; color: var(--ink-muted); white-space: nowrap; }
.match-note { font-size: 13px; line-height: 1.45; color: var(--ink-muted); background: #eef0ea; border-left: 3px solid var(--brass); padding: 8px 10px; }

.sheet { margin-top: 22px; display: grid; grid-template-columns: repeat(12, minmax(0, 1fr)); gap: 22px; }
.panel { grid-column: span 12; background: var(--wall-deep); border: 1px solid var(--wall-line); border-radius: 3px; padding: 22px 24px; min-width: 0; }
.panel h3 { margin: 0 0 14px; font: 500 11px var(--f-mono); letter-spacing: 0.12em; text-transform: uppercase; color: var(--brass); display: flex; justify-content: space-between; gap: 12px; }
.panel h3 small { color: var(--wall-muted); font-size: 11px; letter-spacing: 0.04em; text-transform: none; }
.half { grid-column: span 6; }
.third { grid-column: span 4; }
.twothirds { grid-column: span 8; }

.strip { display: flex; height: 64px; border-radius: 2px; overflow: hidden; gap: 2px; background: var(--wall-deep); }
.strip div { min-width: 6px; }
.chips { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 10px; margin-top: 12px; }
.sw { display: grid; gap: 3px; font: 500 12px var(--f-mono); color: var(--wall-text); font-variant-numeric: tabular-nums; }
.sw i { display: block; height: 10px; border-radius: 1px; }
.sw span { color: var(--wall-muted); font-family: var(--f-body); font-size: 12px; }

.note { font-size: 16px; line-height: 1.65; color: var(--wall-text); max-width: 72ch; margin: 0; }
.note + .note { margin-top: 10px; }
.flag { font: 500 10px var(--f-mono); letter-spacing: 0.05em; text-transform: uppercase; color: var(--doubt); border: 1px solid rgba(224, 179, 90, 0.5); border-radius: 2px; padding: 0 4px; white-space: nowrap; }
.flag.ctx { color: var(--wall-muted); border-color: var(--wall-line); }

.objs { list-style: none; margin: 0; padding: 0; display: grid; gap: 2px; }
.obj { display: grid; grid-template-columns: 26px minmax(0, 1fr); gap: 0 10px; align-items: center; padding: 7px 8px; border-radius: 2px; cursor: default; }
.obj:hover { background: rgba(196, 154, 92, 0.1); }
.obj .n { width: 22px; height: 22px; border-radius: 50%; background: var(--brass); color: #1a1408; font: 600 11px/22px var(--f-mono); text-align: center; }
.obj .n.doubt { background: var(--doubt); }
.obj .n.none { background: transparent; border: 1px solid var(--wall-line); color: var(--wall-muted); line-height: 20px; }
.obj .nm { font-size: 15px; }
.obj .st2 { grid-column: 2; font: 500 11px var(--f-mono); color: var(--wall-muted); }

.route { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
.step { border: 1px solid var(--wall-line); border-radius: 3px; padding: 12px 14px; display: grid; gap: 4px; opacity: 0.5; }
.step .k { font: 600 22px/1 var(--f-display); color: var(--wall-muted); }
.step b { font-weight: 600; font-size: 14px; }
.step span { font-size: 12px; color: var(--wall-muted); line-height: 1.4; }
.step.on { opacity: 1; border-color: var(--brass); background: rgba(196, 154, 92, 0.08); }
.step.on .k { color: var(--brass); }
.why { margin: 14px 0 0; font-size: 15px; line-height: 1.55; max-width: 80ch; }
.meter { margin-top: 18px; display: grid; gap: 6px; }
.meter .bar { position: relative; height: 8px; border-radius: 4px; background: linear-gradient(90deg, #26313d 0%, #26313d 70%, #3a3222 70%, #3a3222 99%, #4a3a20 99%); }
.meter .mk { position: absolute; top: -5px; width: 3px; height: 18px; background: var(--brass); border-radius: 1px; }
.meter .tick { position: absolute; top: 12px; font: 500 10px var(--f-mono); color: var(--wall-muted); transform: translateX(-50%); white-space: nowrap; }
.meter .lbl { font: 500 12px var(--f-mono); color: var(--wall-text); }
.meter .scale { position: relative; height: 18px; }

.data { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 6px 14px; margin: 0; font-size: 14px; }
.data dt { font: 500 11px/1.9 var(--f-mono); letter-spacing: 0.06em; text-transform: uppercase; color: var(--wall-muted); }
.data dd { margin: 0; line-height: 1.45; }
.data .guess { color: var(--wall-muted); border-color: var(--wall-line); }
#result .label-card, #result .label-card dd, #result .label-card .alts .t, #result .label-card h2 { color: var(--ink); }
#result .label-card dt, #result .label-card .kick { color: var(--ink-muted); }
#result h2, #result h3, #result p, #result dl { margin-block-start: 0; }
.err { background: var(--paper); color: var(--ink); padding: 26px; border-radius: 3px; max-width: 60ch; }

@media (max-width: 900px) {
  .exhibit { grid-template-columns: minmax(0, 1fr); }
  .half, .third, .twothirds { grid-column: span 12; }
  .chips { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .route { grid-template-columns: minmax(0, 1fr); }
  #masthead { grid-template-columns: minmax(0, 1fr); }
  #masthead .plaque { text-align: left; }
}
"""

E = html.escape
IT = None
NAMES = {"white": (245, 244, 240), "ivory": (236, 231, 214), "cream": (232, 220, 190), "beige": (214, 200, 172), "sand": (196, 176, 140),
         "tan": (184, 152, 112), "ochre": (198, 148, 58), "mustard": (196, 168, 50), "yellow": (236, 214, 72), "gold": (200, 160, 70),
         "orange": (228, 128, 44), "terracotta": (190, 100, 70), "rust": (160, 72, 40), "brown": (120, 76, 48), "dark brown": (70, 44, 30),
         "red": (200, 40, 40), "crimson": (160, 24, 48), "burgundy": (110, 28, 42), "pink": (232, 160, 176), "rose": (200, 110, 130),
         "mauve": (170, 120, 150), "purple": (110, 60, 130), "violet": (150, 120, 200), "navy": (24, 34, 80), "blue": (50, 90, 190),
         "sky blue": (140, 190, 230), "slate blue": (90, 110, 140), "teal": (30, 120, 120), "turquoise": (60, 180, 180),
         "dark green": (30, 70, 44), "green": (60, 140, 70), "olive": (120, 120, 56), "sage": (150, 168, 130), "grey": (128, 128, 128),
         "light grey": (196, 196, 194), "charcoal": (54, 56, 60), "black": (18, 18, 20)}


def _lab(rgb):
    def f(c):
        c /= 255
        return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92
    r, g, b = (f(float(c)) for c in rgb)
    x, y, z = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047, r * 0.2126 + g * 0.7152 + b * 0.0722, (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883
    t = lambda v: v ** (1 / 3) if v > 0.008856 else 7.787 * v + 16 / 116
    return 116 * t(y) - 16, 500 * (t(x) - t(y)), 200 * (t(y) - t(z))


LABS = {k: _lab(v) for k, v in NAMES.items()}


def paint_name(hexv):
    q = _lab(tuple(int(hexv[i:i + 2], 16) for i in (1, 3, 5)))
    return min(LABS, key=lambda k: sum((a - b) ** 2 for a, b in zip(LABS[k], q)))


def plain_why(tc, artist):
    m, path, why = tc.get("match") or {}, tc["path"], tc["why"]
    first = (tc["titles"] or [{}])[0]
    s = []
    if m and m.get("score", 0) >= DUP_THRESHOLD:
        s.append(f"This photo matches a catalogued work, '{m['title']}' by {m['artist']} ({m['source']}, similarity {m['score']:.2f}), "
                 f"so the title follows {m['artist']}'s titles without copying this one.")
    elif path == "a":
        n = re.search(r"has (\d+) titled works", why)
        s.append(f"You named {artist}. {n.group(1) + ' of their' if n else 'Their'} titled works are in WikiArt and our gallery, "
                 "so the title follows how they title their work.")
    elif path == "b":
        s.append(f"{'The artist you named has no titled works in either collection. ' if artist else 'No artist was given. '}"
                 f"The closest work is '{m['title']}' by {m['artist']} ({m['source']}, similarity {m['score']:.2f}, at or above {B_THRESHOLD:.2f}), "
                 f"so the title follows {m['artist']}'s naming habits.")
    elif "fell back" in why and m:
        s.append(f"The closest work is '{m['title']}' by {m['artist']} (similarity {m['score']:.2f}), but no title written in that artist's manner "
                 "passed the checks, so the title comes from the analysis alone.")
    elif m:
        s.append(f"{'The artist you named has no titled works in either collection, and the' if artist else 'No artist was given, and the'} "
                 f"closest work ('{m['title']}' by {m['artist']}, {m['score']:.2f}) is below {B_THRESHOLD:.2f}, so the title comes from the analysis alone.")
    else:
        s.append("The title comes from the analysis alone.")
    if first.get("prefix"):
        s.append(f"That artist usually calls works '{first['prefix']}', so the title keeps the word and adds a name drawn from the painting.")
    if first.get("fallback"):
        s.append("No drafted title passed the checks, so the verified theme is used as the title.")
    return " ".join(s)


def label_text(text, n=2):
    t = re.sub(r"\s*\[[^\]]*\]", "", text or "")
    parts = re.split(r"(?<=[.!?])\s+", t.strip())
    return " ".join(parts[:n])


def masthead():
    return f"""<div id="masthead"><div><h1>The Artwork <em>Interpreter</em></h1>
<p>Hang a photo of a painting on the wall. It reads what is there, checks every object it names, mixes the palette, and writes the wall label, title included, using only what it can see.</p></div>
<div class="plaque">Qwen2.5-VL · fine-tuned<br>OWLv2 object check<br><b>{N_WORKS}</b> works searched</div></div>"""


def empty():
    return """<div class="stage-wall"><div class="empty"><div class="frame">your painting</div><h2>Nothing on the wall yet</h2>
<div>Upload a photo of a painting on the left. A phone photo with some wall around it works; the frame and wall are cropped away. If you know the artist, add the name and the title will follow their naming habits.</div></div></div>"""


def stage_html(now, secs):
    order = [s[0] for s in STAGES]
    k = order.index(now) if now in order else 0
    rows = "".join(f'<div class="st {"done" if i < k else "now" if i == k else ""}"><div class="dot"></div><div><b>{E(n)}</b><span>{E(d)}</span></div></div>'
                   for i, (_, n, d) in enumerate(STAGES))
    return f'<div class="stage-wall"><div class="stages"><div class="clock">Working · {int(secs)} s · usually about two minutes</div>{rows}</div></div>'


def img_b64(im, side=1100):
    im = im.copy()
    im.thumbnail((side, side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=86)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def marked(text):
    t = E(text or "")
    t = t.replace("[Subject unclear: the three readings of this photo disagree.]", '<span class="flag">subject unclear: three readings disagree</span>')
    t = t.replace("[likely not present]", '<span class="flag">likely not present</span>')
    t = t.replace("[unconfirmed]", '<span class="flag">unconfirmed</span>')
    t = t.replace("[context, not verified from the image]", '<span class="flag ctx">context, not from the image</span>')
    return t


def guess(v):
    v = str(v or "").strip()
    g = "(guess)" in v.lower()
    v = re.sub(r"\s*\(guess\)", "", v, flags=re.I).strip()
    return E(v) + (' <span class="guess">guess</span>' if g else "")


def title_parts(t):
    if " — " in t and t.split(" — ")[0] in ("Untitled", "Abstraction"):
        pre, name = t.split(" — ", 1)
        return f'<span class="pre">{E(pre)} —</span>{E(name)}'
    return E(t)


def source_chip(src):
    parts = [SRC.get(s, s) for s in str(src or "").split("+") if s]
    return f'<span class="chip">{E(" + ".join(parts))}</span>' if parts else ""


def render(out, artist):
    a, tc = out["analysis"], out["title"]
    pred = a["pred"]
    objs = [o for o in a["objects"] if o["status"] != "removed"]
    pins, rects, items, n = [], [], [], 0
    for o in objs:
        doubt = o["status"] != "confirmed"
        if o.get("box"):
            n += 1
            x0, y0, x1, y1 = o["box"]
            cls = "doubt" if doubt else ""
            rects.append(f'<rect id="r{n}" class="{cls}" x="{x0 * 1000:.0f}" y="{y0 * 1000:.0f}" width="{(x1 - x0) * 1000:.0f}" height="{(y1 - y0) * 1000:.0f}"/>')
            pins.append(f'<div class="pin {cls}" id="p{n}" title="{E(o["object"])}" style="left:{x0 * 100:.1f}%;top:{y0 * 100:.1f}%">{n}</div>')
            num = f'<span class="n {cls}">{n}</span>'
            hover_css = f".entry:has(#o{n}:hover) #r{n}, .entry:has(#p{n}:hover) #r{n} {{ opacity: 1; }}"
        else:
            num, hover_css = '<span class="n none">–</span>', ""
        status = "confirmed" if not doubt else "unconfirmed"
        det = f" · detector {o['detector']:.2f}" if o.get("detector") is not None else " · not boxable"
        opener = f'<li class="obj" id="o{n}">' if o.get("box") else '<li class="obj">'
        items.append((opener + f'{num}<span class="nm">{E(o["object"])}</span><span class="st2">{status}{det}</span></li>', hover_css))
    hover = "".join(h for _, h in items)
    obj_list = "".join(i for i, _ in items) or '<li class="obj"><span class="n none">–</span><span class="nm">No separate objects named</span><span></span></li>'

    titles = tc["titles"]
    main = titles[0]["title"] if titles else "Untitled"
    alts = "".join(f'<div class="row"><span class="t">{E(t["title"])}</span>{source_chip(t.get("source"))}</div>' for t in titles[1:4])
    m = tc.get("match") or {}
    near = m and m.get("score", 0) >= DUP_THRESHOLD
    match_note = (f'<div class="match-note">This photo matches a catalogued work: <i>{E(m["title"])}</i> by {E(m["artist"])} '
                  f'({E(m["source"])}, similarity {m["score"]:.2f}). The title above is written fresh, not copied.</div>') if near else ""
    facts = "".join(f"<dt>{lab}</dt><dd>{guess(pred.get(k))}</dd>" for k, lab in FIELDS[2:5])

    pal = a["palette"]
    strip = "".join(f'<div style="flex:{float(c["pct"]):.1f};background:{c["hex"]}" title="{c["hex"]} {float(c["pct"]):.1f}%"></div>' for c in pal)
    chips = "".join(f'<div class="sw"><i style="background:{c["hex"]}"></i>{c["hex"].upper()} · {float(c["pct"]):.1f}%<span>{E(paint_name(c["hex"]))}</span></div>' for c in pal)

    grid = "".join(f"<dt>{lab}</dt><dd>{guess(pred.get(k))}</dd>" for k, lab in FIELDS[:2] + FIELDS[5:])
    path = tc["path"]
    steps = "".join(f'<div class="step {"on" if k == path else ""}"><span class="k">{k}</span><b>{E(n_)}</b><span>{E(d)}</span></div>' for k, (n_, d) in PATHS.items())
    score = float(m.get("score", 0)) if m else 0.0
    meter = ""
    if m:
        meter = (f'<div class="meter"><div class="lbl">Closest work: <i>{E(m["title"])}</i> by {E(m["artist"])} · {E(m["source"])} · similarity {score:.2f}</div>'
                 f'<div class="bar"><div class="mk" style="left:calc({min(score, 1) * 100:.1f}% - 1px)"></div></div>'
                 f'<div class="scale"><span class="tick" style="left:{B_THRESHOLD * 100:.0f}%">path b from {B_THRESHOLD:.2f}</span>'
                 f'<span class="tick" style="left:96%">same work {DUP_THRESHOLD:.2f}</span></div></div>')
    n_rej = len(tc.get("rejected", []))
    why = E(plain_why(tc, artist))
    return f"""<style>{hover}</style>
<div class="entry"><div class="exhibit">
  <div class="hang"><div class="piece"><img src="{img_b64(out['painting'])}" alt="The uploaded painting, cropped to the painted surface">
    <svg viewBox="0 0 1000 1000" preserveAspectRatio="none">{''.join(rects)}</svg>{''.join(pins)}</div>
    <div class="caption">Cropped to the painted surface · pins mark objects the detector located · hover a pin or an object to outline it</div></div>
  <div class="label-card">
    <span class="kick">{('Artist given: ' + E(artist)) if artist else 'Artist not given'} · title path {path}</span>
    <h2>{title_parts(main)}</h2>
    <div class="by">written by {source_chip(titles[0].get("source") if titles else "")}</div>
    <p class="lbl-text">{E(label_text(a.get("explanation_marked")))}</p>
    {match_note}
    <dl class="facts">{facts}</dl>
    <div class="alts"><span class="kick">Other titles it would give</span>{alts or '<span class="chip">no other title passed the checks</span>'}</div>
  </div>
</div>
<div class="sheet">
  <div class="panel"><h3>Palette <small>six dominant colors, share of the painted surface</small></h3><div class="strip">{strip}</div><div class="chips">{chips}</div></div>
  <div class="panel twothirds"><h3>What is in the painting <small>verified description</small></h3><p class="note">{marked(a.get('description_marked'))}</p></div>
  <div class="panel third"><h3>Objects <small>{sum(1 for o in objs if o['status'] == 'confirmed')} confirmed</small></h3><ul class="objs">{obj_list}</ul></div>
  <div class="panel twothirds"><h3>For visitors <small>plain-language explanation</small></h3><p class="note">{marked(a.get('explanation_marked'))}</p></div>
  <div class="panel third"><h3>Catalogue data</h3><dl class="data">{grid}</dl></div>
  <div class="panel"><h3>How the title was found <small>{n_rej} drafts removed by the checks · {out['seconds']:.0f} s</small></h3>
    <div class="route">{steps}</div><p class="why">{why}</p>{meter}</div>
</div></div>"""


def run(path, artist):
    if not path:
        yield empty()
        return
    photo = open_photo(path)
    q, box = queue.Queue(), {}

    def work():
        try:
            box["out"] = IT.run(photo, artist, step=q.put)
        except Exception as e:
            box["err"] = repr(e)
        q.put(None)

    th = threading.Thread(target=work, daemon=True)
    th.start()
    now, t0 = "reading", time.time()
    yield stage_html(now, 0)
    while True:
        try:
            s = q.get(timeout=1.0)
            if s is None:
                break
            now = s
        except queue.Empty:
            pass
        yield stage_html(now, time.time() - t0)
    th.join()
    if "err" in box:
        yield f'<div class="err"><b>Something went wrong.</b> {E(box["err"])}</div>'
        return
    out = box["out"]
    if out.get("error"):
        yield f'<div class="err"><b>{E(out["error"])}</b> Try a sharper photo where the painting fills most of the frame.</div>'
        return
    yield render(out, (artist or "").strip())


def examples():
    R = {r["id"]: r for r in records()}
    ids = ["F1-053", "F1-015", "F1-054", "F2-086"]
    return [[str(ROOT / R[i]["file"])] for i in ids if i in R]


def build():
    with gr.Blocks(title="Artwork Interpreter", fill_width=True) as demo:
        gr.HTML(masthead(), apply_default_css=False, padding=False, container=False)
        with gr.Row(elem_id="bench"):
            with gr.Column(scale=4, min_width=280, elem_id="intake"):
                photo = gr.Image(type="filepath", label="Photo of a painting", sources=["upload", "clipboard"], height=340, elem_id="photo")
                artist = gr.Textbox(label="Artist, if known", placeholder="e.g. Wassily Kandinsky", elem_id="artist")
                go = gr.Button("Interpret", variant="primary", elem_id="go")
                gr.HTML('<div class="label-note">Or try one of the gallery photos</div>', apply_default_css=False, padding=False, container=False)
                gr.Examples(examples(), inputs=[photo], label="", elem_id="examples")
            with gr.Column(scale=9, min_width=320):
                result = gr.HTML(empty(), elem_id="result", apply_default_css=False, padding=False, container=False)
        go.click(run, inputs=[photo, artist], outputs=result)
    return demo


if __name__ == "__main__":
    print("loading models...", flush=True)
    IT = Interpreter()
    print("ready", flush=True)
    build().queue(default_concurrency_limit=1).launch(css=CSS, head=HEAD, theme=gr.themes.Base(), server_port=int(os.environ.get("PORT", "7860")))

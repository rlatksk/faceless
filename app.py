import glob
import json
import os
import shutil
import time
from datetime import datetime

from dotenv import load_dotenv
import streamlit as st

load_dotenv()

from pipeline.audio import synthesize_edge, list_edge_voices  # noqa: E402
from pipeline.script import generate_script  # noqa: E402
from pipeline.reddit import fetch_post  # noqa: E402
from pipeline.transcribe import transcribe  # noqa: E402
from pipeline.images import generate_images  # noqa: E402
from pipeline.assemble import assemble, MUSIC_DIR  # noqa: E402
from pipeline.cost import PRICES, KENARI_IMAGE_IDR, KENARI_IMAGE_IDR_DEFAULT, IDR_PER_USD  # noqa: E402

PROJECTS_DIR = "output"
STEP_NAMES = ["audio", "transcribe", "images", "assemble"]
STAGE_LABELS = [
    ("script", "Script"),
    ("audio", "Voice"),
    ("transcribe", "Timing"),
    ("images", "Images"),
    ("assemble", "Render"),
]

# Share of total render time each stage takes, measured on a 93s video. Used to
# weight the progress bar so it advances at roughly a constant rate — an
# unweighted bar would hit 40% after the two fast stages and then look stalled
# through images and render, which are the slow ones.
_STAGE_WEIGHTS = {
    "script": 0.0,      # already done before the render starts
    "audio": 0.06,
    "transcribe": 0.14,
    "images": 0.30,
    "assemble": 0.50,
}
STYLE_PRESETS = {
    "Indie Dark Comic — horror, true crime": (
        "A dark graphic novel illustration of [INSERT YOUR SCENE / CHARACTER HERE]. "
        "Gritty indie comic book art, thick clean black ink outlines, digital cel-shading, "
        "heavy chiaroscuro shadow and deep crushed blacks. Muted desaturated palette of cold "
        "blues, sickly greens and dried-blood reds. Character design signature: large eyes with "
        "tiny pinpoint pupils, drawn tight with whatever the scene demands — dread, suspicion, "
        "grief or sudden shock. Cinematic vertical framing, high contrast, subtle film grain, "
        "high quality."
    ),
    "AITA Storytime — drama, advice": (
        "A bright flat-vector cartoon illustration of [INSERT YOUR SCENE / CHARACTER HERE]. "
        "Modern YouTube storytime animation style, bold clean outlines, flat saturated colour "
        "fills, simple geometric shapes, no gradients. Warm everyday palette of coral, teal, "
        "mustard and cream. Characters are stylised with exaggerated expressive faces and clear "
        "body language — smug, furious, sheepish or tearful as the moment calls for. Simple "
        "uncluttered backgrounds, soft even lighting, vertical framing, high quality."
    ),
}

DEFAULT_MODELS = {
    "ollama": "llama3",
    "deepseek": "deepseek-v4-flash",
    "kenari": "deepseek-v4-1-flash",
    "openrouter": "openai/gpt-4o-mini",
}

# Keys the app can actually use, and where to get them.
API_KEYS = {
    "kenari": ("KENARI_API_KEY", "Kenari", "https://kenari.id"),
    "deepseek": ("DEEPSEEK_API_KEY", "DeepSeek", "https://platform.deepseek.com/api_keys"),
    "openrouter": ("OPENROUTER_API_KEY", "OpenRouter", "https://openrouter.ai/keys"),
}


def _setting(provider):
    """The configured model for a provider.

    The Settings widgets write these keys, but they are rendered after the
    Create tab, so on the first pass they do not exist yet. Falling back to the
    module default keeps the value stable across reruns — reading an absent key
    as "" would make the estimate look stale the moment Settings rendered.
    """
    return st.session_state.get(f"_settings_{provider}") or DEFAULT_MODELS[provider]


_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Share+Tech+Mono&display=swap');

/* Phosphor terminal theme. Every colour is a variable so the palette lives in
   one place — the previous sheet hardcoded its accent in eleven rules, which
   made any restyle a find-and-replace. */
:root {
    --bg: #050705;
    --surface: #0a0d0a;
    --surface-2: #0d120d;
    --line: #1c241c;
    --line-bright: #2b3a2b;
    --amber: #ffb000;
    --amber-dim: #b37b00;
    --amber-glow: rgba(255, 176, 0, 0.35);
    --green: #4ee06a;
    --red: #ff5b4a;
    --text: #cfe3cf;
    /* Contrast measured against --bg: --text-dim 8.4, --text-faint 4.9.
       The first pass had faint at 2.65, which failed WCAG AA for small text —
       inactive tabs and card metadata were genuinely hard to read. */
    --text-dim: #9db09d;
    --text-faint: #7f917f;
}

.stApp {
    background: var(--bg);
    color: var(--text);
}

/* Phosphor screen texture: a faint scanline grid plus a slow vertical drift,
   and a subtle vignette so the edges fall off like a CRT. Static grid, animated
   sweep — the sweep is slow enough (8s) to read as ambience, not motion. */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: repeating-linear-gradient(
        0deg,
        rgba(0, 0, 0, 0.22) 0px,
        rgba(0, 0, 0, 0.22) 1px,
        transparent 1px,
        transparent 3px
    );
    pointer-events: none;
    z-index: 9998;
}

.stApp::after {
    content: '';
    position: fixed;
    inset: 0;
    background: linear-gradient(
        180deg,
        transparent 0%,
        rgba(255, 176, 0, 0.035) 48%,
        rgba(255, 176, 0, 0.05) 50%,
        rgba(255, 176, 0, 0.035) 52%,
        transparent 100%
    );
    background-size: 100% 220px;
    animation: sweep 8s linear infinite;
    pointer-events: none;
    z-index: 9999;
}

@keyframes sweep {
    from { background-position: 0 -220px; }
    to   { background-position: 0 100vh; }
}

@media (prefers-reduced-motion: reduce) {
    .stApp::after { animation: none; }
}

/* Streamlit ships its own heading rules (.st-bp etc.) with equal specificity that
   load after this sheet, so bare `h1` loses. Scoping under .stApp outranks them
   without !important. */
.stApp h1 {
    font-family: 'Share Tech Mono', monospace !important;
    font-weight: 400;
    font-size: 2.6rem !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    line-height: 1;
    margin-bottom: 0;
    color: var(--amber) !important;
    text-shadow: 0 0 12px var(--amber-glow);
}

.stApp h2, .stApp h3 {
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* numbered section heading inside the create form */
.step-head {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--amber);
    margin: 0.75rem 0 0.2rem;
}
.step-head span {
    color: var(--text-faint);
    margin-right: 0.5rem;
}
.step-note {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-dim);
    margin: 0 0 0.75rem;
    line-height: 1.5;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid var(--line-bright);
}

.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--text-faint);
    padding: 0.5rem 1.5rem;
    transition: color 0.15s, text-shadow 0.15s;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--amber-dim);
}

.stTabs [aria-selected="true"] {
    color: var(--amber) !important;
    text-shadow: 0 0 10px var(--amber-glow);
}

div[data-testid="stButton"] > button {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    letter-spacing: 0.1em;
    border-radius: 0;
    transition: all 0.15s;
    font-size: 0.75rem;
}

div[data-testid="stButton"] > button[kind="primary"] {
    background: transparent;
    border: 1px solid var(--amber);
    color: var(--amber);
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    text-shadow: 0 0 8px var(--amber-glow);
}

div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: var(--amber);
    color: #050705;
    text-shadow: none;
    box-shadow: 0 0 18px var(--amber-glow);
}

div[data-testid="stButton"] > button:not([kind="primary"]) {
    background: transparent;
    border: 1px solid var(--line-bright);
    color: var(--text-dim);
    text-transform: uppercase;
}

div[data-testid="stButton"] > button:not([kind="primary"]):hover {
    border-color: var(--amber-dim);
    color: var(--amber);
    background: transparent;
}

/* card-like containers */
div[data-testid="stExpander"] {
    border: 1px solid var(--line);
    border-radius: 0;
    background: var(--surface);
    margin-bottom: 0.75rem;
}

div[data-testid="stExpander"] summary {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 500;
}

div[data-testid="stExpander"] summary span {
    font-size: 0.78rem;
}

.stTextArea textarea, div[data-testid="stTextInput"] input {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    color: var(--text);
    line-height: 1.6;
}

.stTextArea textarea:focus, div[data-testid="stTextInput"] input:focus {
    border-color: var(--amber-dim);
    box-shadow: 0 0 0 1px var(--amber-glow);
}

.stTextArea textarea::placeholder {
    color: var(--text-faint);
}

div[data-testid="stRadio"] label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
}

div[role="progressbar"] > div {
    background: var(--amber) !important;
    border-radius: 0;
    box-shadow: 0 0 8px var(--amber-glow);
}

div[role="progressbar"] {
    background: var(--line) !important;
    border-radius: 0;
}

/* Streamlit ships cool blue/green/red alert palettes. Retint them so the page
   stays monochrome — the stock blue info banner was the one cool hue left. */
div[role="alert"] {
    border-left: 2px solid var(--amber);
    border-radius: 0;
    background: var(--surface);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: var(--text);
}
div[data-testid="stAlertContainer"] {
    background: var(--surface) !important;
    border: 1px solid var(--line);
    border-left: 2px solid var(--amber);
    border-radius: 0;
    color: var(--text) !important;
}
div[data-testid="stAlertContainer"] p,
div[data-testid="stAlertContainer"] span {
    color: var(--text) !important;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
}
div[data-testid="stNotification"] {
    background: var(--surface) !important;
    border: 1px solid var(--line-bright);
    color: var(--text) !important;
    font-family: 'IBM Plex Mono', monospace;
    border-radius: 0;
}
div[data-testid="stToast"] {
    background: var(--surface) !important;
    border: 1px solid var(--amber-dim);
    color: var(--amber) !important;
    font-family: 'IBM Plex Mono', monospace;
    border-radius: 0;
}
/* Widget labels (Voice, Style, Image model…) inherit the terminal type. */
div[data-testid="stWidgetLabel"] p, div[data-testid="stWidgetLabel"] span {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.74rem;
    color: var(--text-dim);
    letter-spacing: 0.04em;
}
div[data-testid="stCaptionContainer"] p {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-dim);
}

.stCodeBlock {
    background: var(--surface) !important;
    border: 1px solid var(--line);
    border-radius: 0;
}

.stSelectBox [data-baseweb="select"] {
    border-radius: 0;
}

.stSpinner {
    color: var(--amber);
}

/* stage chips — the live pipeline view */
.stage-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 0.25rem 0 0.75rem;
}
.stage {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.66rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 3px 10px;
    border: 1px solid var(--line);
    border-radius: 0;
    color: var(--text-faint);
    white-space: nowrap;
}
.stage.done {
    border-color: #1f5c2e;
    color: var(--green);
}
/* The active stage is the one thing worth animating: a slow pulse reads as
   "working" without competing with the video. */
.stage.active {
    border-color: var(--amber);
    color: var(--amber);
    animation: pulse 1.6s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 rgba(255, 176, 0, 0); }
    50%      { box-shadow: 0 0 12px var(--amber-glow); }
}
@media (prefers-reduced-motion: reduce) {
    .stage.active { animation: none; box-shadow: 0 0 12px var(--amber-glow); }
}
.stage.failed {
    border-color: #7a2a24;
    color: var(--red);
}

/* live progress panel */
.prog-bar {
    height: 6px;
    background: var(--line);
    margin: 0 0 0.4rem;
    overflow: hidden;
}
.prog-bar > i {
    display: block;
    height: 100%;
    background: var(--amber);
    box-shadow: 0 0 10px var(--amber-glow);
    transition: width 0.3s linear;
}
.prog-detail {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.74rem;
    color: var(--text-dim);
    margin: 0 0 0.6rem;
}
.prog-time {
    color: var(--amber-dim);
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
}

/* project gallery cards */
.proj-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.74rem;
    font-weight: 600;
    color: var(--text);
    margin: 0.35rem 0 0.1rem;
    letter-spacing: 0.04em;
}
.proj-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: var(--text-faint);
    margin-bottom: 0.35rem;
}

/* Equal-height gallery cards. Streamlit wraps each card in a layout wrapper that
   sizes to its own content (flex: 0 1 auto), so a card with a short preview ends
   far above its neighbours and the row bottom is ragged. The wrapper's parent is
   the column's vertical block, which IS a flex container — that is where the
   wrapper has to be told to grow. Each level down to the card must stretch, or
   the card never fills the column.
   `st.container(key=...)` gives each card a stable `st-key-projcard_<id>` class,
   which is the only reliable hook. */
div[data-testid="stHorizontalBlock"]:has([class*="st-key-projcard_"]) {
    align-items: stretch;
}
div[data-testid="stVerticalBlock"]:has(> div[data-testid="stLayoutWrapper"] > [class*="st-key-projcard_"]) {
    align-items: stretch;
}
div[data-testid="stLayoutWrapper"]:has(> [class*="st-key-projcard_"]) {
    flex: 1 1 auto;
    display: flex;
    align-items: stretch;
}
[class*="st-key-projcard_"] {
    /* Streamlit's own rule sets `flex: 1 1 0%` on this element, and in flexbox
       flex-basis wins over `height` — which is why a plain height was ignored and
       rows came out ragged (480px beside 727px). Pin the basis instead, and stop
       the parent's `align-items: stretch` from re-stretching it.
       !important is needed: the competing rule is generated with equal specificity. */
    flex: 0 0 505px !important;
    align-self: flex-start;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    border: 1px solid var(--line);
    border-radius: 0;
    background: var(--surface);
    padding: 0.7rem;
    box-sizing: border-box;
}
/* Reserve a fixed preview box so every card shows the same size image,
   whether it has a video, a still, or nothing at all. Streamlit renders the
   video as the <video> element itself, not a wrapper div. */
[class*="st-key-projcard_"] video,
[class*="st-key-projcard_"] img[data-testid="stImage"] {
    width: 100%;
    height: 250px;
    object-fit: cover;
    display: block;
    border-radius: 2px;
    background: #000;
}
[class*="st-key-projcard_"] div[data-testid="stElementContainer"]:has(video),
[class*="st-key-projcard_"] div[data-testid="stElementContainer"]:has(img) {
    height: 250px;
    margin-bottom: 0.5rem;
    overflow: hidden;
}
.proj-empty {
    height: 250px;
    margin-bottom: 0.5rem;
    border: 1px dashed var(--line-bright);
    border-radius: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-faint);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}
/* The expander sits last; the card already stretches to the row height, so the
   actions line up without pushing the content apart. `margin-top: auto` here
   stretched any card that had a Resume button far taller than its neighbours. */
[class*="st-key-projcard_"] div[data-testid="stExpander"] {
    margin-bottom: 0;
}
[class*="st-key-projcard_"] .stage-row {
    margin: 0.15rem 0 0.5rem;
}

/* settings popover */
div[data-testid="stPopoverBody"] {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 0;
    min-width: 280px;
}
"""


def _save_project(path, data):
    with open(f"{path}/project.json", "w") as f:
        json.dump(data, f, indent=2)


def _load_or_backfill(path):
    pid = os.path.basename(path)
    steps = {s: {"status": "pending"} for s in STEP_NAMES}
    steps["script"] = {"status": "done"}
    if os.path.exists(f"{path}/audio.mp3"):
        steps["audio"] = {"status": "done"}
    if os.path.exists(f"{path}/transcript.json"):
        steps["transcribe"] = {"status": "done"}
    existing = sorted(glob.glob(f"{path}/img_*.png"))
    if os.path.exists(f"{path}/project.json"):
        expected = len(json.load(open(f"{path}/project.json")).get("image_prompts", []))
        if expected > 0:
            if len(existing) >= expected:
                steps["images"] = {"status": "done"}
        elif existing:
            steps["images"] = {"status": "done"}
    elif existing:
        steps["images"] = {"status": "done"}
    if os.path.exists(f"{path}/final.mp4"):
        steps["assemble"] = {"status": "done"}
    complete = all(steps[s]["status"] == "done" for s in STEP_NAMES)
    if os.path.exists(f"{path}/project.json"):
        proj = json.load(open(f"{path}/project.json"))
        proj["steps"] = steps
        proj["status"] = "completed" if complete else "in_progress"
        _save_project(path, proj)
        return proj
    proj = {
        "id": pid, "status": "completed" if complete else "in_progress", "created": "",
        "source_text": "", "narration": "", "image_prompts": [],
        "voice_id": "", "image_model": "", "image_model_label": "",
        "image_provider": "openrouter",
        "image_style": "", "llm_provider": "",
        "steps": steps,
    }
    _save_project(path, proj)
    return proj


def _active_stage(project):
    """The first stage that is not done — i.e. the one a failure came from."""
    steps = project.get("steps", {})
    for key, _ in STAGE_LABELS:
        if steps.get(key, {}).get("status") != "done":
            return key
    return None


def _music_path(project):
    """Resolve a project's stored music track, or None if it had none."""
    name = project.get("music")
    if not name:
        return None
    path = os.path.join(MUSIC_DIR, name)
    if not os.path.exists(path):
        raise RuntimeError(f"Music track for this project is missing: {path}")
    return path


def _stages_html(project, active=None, failed=None):
    steps = project.get("steps", {})
    chips = []
    for key, label in STAGE_LABELS:
        if failed and key == failed:
            cls = "failed"
        elif active == key:
            cls = "active"
        elif steps.get(key, {}).get("status") == "done":
            cls = "done"
        else:
            cls = ""
        chips.append(f'<span class="stage {cls}">{label}</span>')
    return f'<div class="stage-row">{"".join(chips)}</div>'


def _percent(project, active=None, fraction=0.0):
    """Overall completion as a percentage (0-100).

    Weighted by how long each stage actually takes. An unweighted bar would jump
    to 40% after the two fast stages and then crawl, which reads as a stall. The
    weights follow measured durations, so the bar advances at roughly a constant
    rate.
    """
    steps = project.get("steps", {})
    done = sum(w for k, w in _STAGE_WEIGHTS.items()
               if steps.get(k, {}).get("status") == "done")
    if active in _STAGE_WEIGHTS:
        done += _STAGE_WEIGHTS[active] * max(0.0, min(1.0, fraction))
    return min(100.0, done * 100)


def _progress_html(project, detail, started, active=None, failed=None, fraction=0.0):
    """The live progress panel: a percentage bar, stage chips, and elapsed time."""
    elapsed = int(time.time() - started)
    mins, secs = divmod(elapsed, 60)
    pct = _percent(project, active, fraction)
    return (
        _stages_html(project, active=active, failed=failed)
        + f'<div class="prog-bar"><i style="width:{pct:.1f}%"></i></div>'
        + f'<div class="prog-detail"><span>{detail}</span>'
          f'<span class="prog-time">{pct:.0f}% · {mins}:{secs:02d}</span></div>'
    )


def _run_pipeline(project, out_dir, slot):
    """Run every stage that is not done yet, in order.

    Both the first run and Resume call this: a fresh project has every step
    pending, so it executes all five, and a resumed one picks up where it
    stopped. The filesystem stays the source of truth — each completed stage is
    persisted before the next begins.

    `slot` is a single `st.empty()` holding the progress panel; warnings go to
    `st.warning` so they survive the next panel redraw.
    """
    steps = project["steps"]
    started = time.time()

    def mark(key, detail="", fraction=0.0):
        _save_project(out_dir, project)
        slot.markdown(_progress_html(project, detail, started, active=key,
                                     fraction=fraction),
                      unsafe_allow_html=True)

    mark("audio", "Generating voiceover…")
    if steps["audio"]["status"] != "done":
        synthesize_edge(project["narration"], project["voice_id"], f"{out_dir}/audio.mp3")
        steps["audio"] = {"status": "done"}
    mark("transcribe", "Aligning word timings…")

    if steps["transcribe"]["status"] != "done":
        words = transcribe(f"{out_dir}/audio.mp3")
        json.dump(words, open(f"{out_dir}/transcript.json", "w"))
        steps["transcribe"] = {"status": "done"}
    else:
        words = json.load(open(f"{out_dir}/transcript.json"))
    mark("images", "Preparing images…")

    prompts = project.get("image_prompts", [])
    have = [p for p in sorted(glob.glob(f"{out_dir}/img_*.png")) if os.path.getsize(p) > 0]
    if steps["images"]["status"] == "done" and (len(have) == len(prompts) or not prompts):
        images = have
    elif not prompts and have:
        # Legacy project: prompts were never persisted, but the images are real.
        images = have
        steps["images"] = {"status": "done"}
    else:
        if not prompts:
            raise RuntimeError("This project has no image prompts to generate from.")
        # A project can be marked "images: done" with nothing on disk — a stage that
        # failed after the provider was paid leaves exactly that state. Say so before
        # spending again, rather than silently re-buying every image.
        if have:
            st.warning(f"Only {len(have)} of {len(prompts)} images are on disk. "
                       f"The rest will be generated again and charged again.")

        def img_progress(i, n, action):
            mark("images", f"Generating image {i} of {n}…", fraction=i / n)

        provider = project.get("image_provider") or "openrouter"
        images, cost = generate_images(prompts, out_dir, project["image_model"],
                                       provider=provider, progress_cb=img_progress,
                                       seed=project.get("seed"))
        if len(images) != len(prompts):
            raise RuntimeError(f"Only {len(images)}/{len(prompts)} images generated. Check the image model.")
        project["total_cost"] = project.get("total_cost", 0) + cost
        steps["images"] = {"status": "done"}
    mark("assemble", "Rendering video…")

    if steps["assemble"]["status"] != "done":
        if not images:
            raise RuntimeError("No images to assemble — the image step needs to run first.")

        def render_progress(done, total):
            mark("assemble", f"Rendering frame {done} of {total}…", fraction=done / total)

        assemble(images, f"{out_dir}/audio.mp3", words, f"{out_dir}/final.mp4",
                 music_path=_music_path(project), music_volume=project.get("music_volume", 0.15),
                 progress_cb=render_progress)
        steps["assemble"] = {"status": "done"}
    project["status"] = "completed"
    _save_project(out_dir, project)
    elapsed = int(time.time() - started)
    slot.markdown(
        _stages_html(project)
        + '<div class="prog-bar"><i style="width:100%"></i></div>'
        + f'<div class="prog-detail"><span>Done</span>'
          f'<span class="prog-time">100% · {elapsed // 60}:{elapsed % 60:02d}</span></div>',
        unsafe_allow_html=True,
    )


def _img_models(custom_raw):
    models = [
        ("Kenari: Nano Banana 2 Lite — 150 IDR/img", "nano-banana-2-lite", "kenari"),
        ("Kenari: Nano Banana 2 — 250 IDR/img", "nano-banana-2", "kenari"),
        ("Kenari: Nano Banana Pro — 350 IDR/img", "nano-banana-pro", "kenari"),
        ("Kenari: Grok Imagine — 300 IDR/img", "grok-imagine-image", "kenari"),
        ("Gemini 3.1 Flash Lite — ~$0.035/img", "google/gemini-3.1-flash-lite-image", "openrouter"),
        ("Grok Imagine 1K — ~$0.05/img", "x-ai/grok-imagine-image-quality", "openrouter"),
        ("Gemini 3.1 Flash — ~$0.07/img", "google/gemini-3.1-flash-image", "openrouter"),
        ("Gemini 3 Pro — ~$0.14/img", "google/gemini-3-pro-image", "openrouter"),
        ("Local SDXL — Free (10 steps)", "stabilityai/stable-diffusion-xl-base-1.0__10", "local"),
        ("Local SDXL Turbo — Free", "stabilityai/sdxl-turbo", "local"),
    ]
    for line in (custom_raw or "").strip().split("\n"):
        if line.strip():
            models.append((line.strip(), line.strip(), "openrouter"))
    return models


_PRICED_OPENROUTER = {
    "google/gemini-3.1-flash-lite-image": "openrouter_gemini_lite",
    "x-ai/grok-imagine-image-quality": "openrouter_grok_imagine",
    "google/gemini-3.1-flash-image": "openrouter_gemini_flash",
    "google/gemini-3-pro-image": "openrouter_gemini_pro",
}


def _llm_cost_usd(llm_label, tokens):
    """Chat cost. Ollama is local and Kenari's plan covers chat, so both are 0."""
    if "Ollama" in llm_label or "Kenari" in llm_label:
        return 0.0
    key = "deepseek_llm" if "DeepSeek" in llm_label else "openrouter_llm"
    return round(tokens * PRICES[key], 4)


def _estimate_costs(img_model, imgs, llm_label, tokens):
    """Return (image_cost_usd, llm_cost_usd, image_note)."""
    label, model_id, provider = img_model
    if provider == "local":
        return 0.0, 0.0, "free, runs locally"
    if provider == "kenari":
        idr = KENARI_IMAGE_IDR.get(model_id, KENARI_IMAGE_IDR_DEFAULT)
        return round(imgs * idr / IDR_PER_USD, 4), 0.0, f"{idr} IDR each"
    if model_id in _PRICED_OPENROUTER:
        return round(imgs * PRICES[_PRICED_OPENROUTER[model_id]], 4), 0.0, "OpenRouter"
    img_cost = round(imgs * PRICES["openrouter_gemini_flash"], 4)
    return img_cost, _llm_cost_usd(llm_label, tokens), "estimated"


def _section(num, title, note=""):
    st.markdown(f'<div class="step-head"><span>{num}</span>{title}</div>', unsafe_allow_html=True)
    if note:
        st.markdown(f'<div class="step-note">{note}</div>', unsafe_allow_html=True)


def _llm_choice(label):
    """Map the provider radio label to (provider, model)."""
    if "Ollama" in label:
        return "ollama", _setting("ollama")
    if "DeepSeek" in label:
        return "deepseek", _setting("deepseek")
    if "Kenari" in label:
        return "kenari", _setting("kenari")
    return "openrouter", _setting("openrouter_llm")


st.set_page_config(page_title="Faceless", layout="wide")
st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)

col_h, col_s = st.columns([5, 1])
with col_h:
    st.markdown(
        '<h1 style="margin-bottom:0">FACELESS</h1>'
        '<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.68rem;'
        'color:var(--text-dim);'
        'letter-spacing:0.22em;text-transform:uppercase;margin-top:-0.15rem">'
        'vertical video generator</p>',
        unsafe_allow_html=True,
    )

tab_create, tab_projects, tab_settings = st.tabs(["Create", "Projects", "Settings"])

# --------------------------------------------------------------------------
# Create
# --------------------------------------------------------------------------
with tab_create:
    if st.session_state.pop("_clear_source", False):
        st.session_state["source_text"] = ""
    if st.session_state.pop("_clear_style", False):
        st.session_state["style_custom"] = ""
    if st.session_state.pop("_clear_estimate", False):
        st.session_state.pop("script_result", None)

    left, mid, right = st.columns([1, 3, 1])
    with mid:
        _section(1, "Your story", "Paste anything with a narrative — the script model splits it into scenes.")
        with st.popover("Import from Reddit", use_container_width=False):
            st.caption("Paste a post URL and it fills the story box below.")
            reddit_url = st.text_input(
                "Reddit post URL", key="_reddit_url", label_visibility="collapsed",
                placeholder="https://www.reddit.com/r/AmItheAsshole/comments/...")
            if st.button("Fetch post", key="_reddit_fetch", use_container_width=True):
                if not reddit_url.strip():
                    st.warning("Paste a Reddit post link first.")
                else:
                    with st.spinner("Fetching from Reddit…"):
                        try:
                            post = fetch_post(reddit_url.strip())
                        except Exception as e:
                            post = None
                            st.error(str(e))
                    if post:
                        st.session_state["source_text"] = post["text"]
                        st.session_state["_reddit_credit"] = (
                            f"{post['subreddit']} · {post['author']}" if post["subreddit"]
                            else post["author"])
                        st.rerun()
            if st.session_state.get("_reddit_credit"):
                st.caption(f"Loaded from {st.session_state['_reddit_credit']}")

        source = st.text_area("Source script", height=160, label_visibility="collapsed",
                              placeholder="Paste your story or script here...", key="source_text")

        _section(2, "Script", "Turns your text into narration plus one image prompt per scene.")
        llm_label = st.radio(
            "Script model",
            ["Kenari (API)", "DeepSeek (API)", "Ollama (local)", "OpenRouter (API)"],
            horizontal=True, label_visibility="collapsed")
        llm_provider, llm_model = _llm_choice(llm_label)
        _key_env, _key_name, _key_url = API_KEYS[llm_provider]
        if not os.getenv(_key_env):
            st.warning(f"{_key_env} is not set in .env — add it, or pick another script model. "
                       f"Get a key at {_key_url}")

        _section(3, "Look and sound", "These decide the video's voice, art style and image model.")
        c1, c2 = st.columns(2)
        with c1:
            try:
                edge_voices = list_edge_voices()
                voice_labels = {v["name"]: v["id"] for v in edge_voices}
                default_name = next((n for n in voice_labels if "Eric" in n), list(voice_labels)[0])
                selected = st.selectbox("Voice", list(voice_labels),
                                        index=list(voice_labels).index(default_name))
                voice_id = voice_labels[selected]
            except Exception as e:
                st.error(f"Could not load voices: {e}")
                voice_id = "en-US-EricNeural"

            style_preset = st.selectbox("Style", [*STYLE_PRESETS, "Custom"])
            if style_preset == "Custom":
                style = st.text_input("Custom style tag", value="",
                                      placeholder="Describe your custom style...", key="style_custom")
            else:
                style = STYLE_PRESETS[style_preset]
        with c2:
            img_model = st.selectbox("Image model",
                                     _img_models(st.session_state.get("_settings_img_models")),
                                     format_func=lambda x: x[0])

            lock_character = st.toggle("Lock character (same seed every image)")
            if lock_character:
                seed = st.session_state.setdefault("_seed", int.from_bytes(os.urandom(4), "big"))
                if img_model[2] == "kenari":
                    st.caption(f"Seed {seed} — but Kenari's image models ignore seeds, so this "
                               f"changes nothing. Character consistency comes from the script's "
                               f"character sheet instead.")
                else:
                    st.caption(f"Seed {seed} — reuse it to reproduce the same cast.")
            else:
                seed = None

            tracks = sorted(glob.glob(f"{MUSIC_DIR}/*.mp3")) if os.path.isdir(MUSIC_DIR) else []
            track_labels = ["Off"] + [os.path.basename(t) for t in tracks]
            track = st.selectbox("Music", track_labels)
            music_path = tracks[track_labels.index(track) - 1] if track != "Off" else None
            music_volume = st.slider("Music volume", 0.0, 1.0, 0.15, 0.01,
                                     disabled=music_path is None)

        # --- estimate state -------------------------------------------------
        # Only the inputs that change the *script* invalidate the estimate. The
        # voice, image model, style and music are applied at render time, so
        # changing them must not throw the generated script away.
        fingerprint = {"source": source, "llm": llm_label, "model": llm_model}
        result = st.session_state.get("script_result")
        stale = result is not None and st.session_state.get("last_fp") != fingerprint

        _section(4, "Cost and render",
                 "Estimating runs the script model once. Nothing else is charged until you render.")
        if stale:
            st.warning("Your story or script model changed, so this estimate is out of date.")
        if result is None:
            st.info("Not estimated yet. Estimating writes the narration and the image prompts.")
        else:
            imgs = len(result["image_prompts"])
            img_cost, llm_cost, note = _estimate_costs(
                img_model, imgs, llm_label, result.get("_token_count", 0))
            m1, m2, m3 = st.columns(3)
            m1.metric("Scenes", imgs)
            m2.metric("Images", f"${img_cost:.2f}", help=note)
            m3.metric("Script", "free" if not llm_cost else f"${llm_cost:.2f}")
            st.caption(f"Estimated total **${img_cost + llm_cost:.2f}** · {note}")

        if result is None or stale:
            label = "Estimate script" if result is None else "Re-estimate script"
            clicked = st.button(label, type="primary", use_container_width=True,
                                disabled=not source)
            if not source:
                st.caption("Add your story above to enable this.")
            if clicked:
                with st.spinner("Running the script model..."):
                    out = generate_script(source, provider=llm_provider, model=llm_model)
                st.session_state["script_result"] = out
                st.session_state["last_fp"] = fingerprint
                st.session_state["_llm_cost"] = _llm_cost_usd(
                    llm_label, out.get("_token_count", 0))
                st.rerun()
        else:
            b1, b2 = st.columns([3, 1])
            with b1:
                if st.button("Render video", type="primary", use_container_width=True):
                    st.session_state["_start_render"] = True
            with b2:
                if st.button("Re-estimate", use_container_width=True):
                    st.session_state.pop("script_result", None)
                    st.rerun()

        if result is not None:
            with st.expander("Preview the generated script"):
                st.write("**Narration**")
                st.write(result["narration"])
                st.write("**Image prompts**")
                for i, p in enumerate(result["image_prompts"], 1):
                    st.write(f"{i}. {p}")

        # --- render ---------------------------------------------------------
        if st.session_state.pop("_start_render", False) and result is not None:
            project_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            out_dir = f"{PROJECTS_DIR}/{project_id}"
            os.makedirs(out_dir, exist_ok=True)

            cast = "\n".join(
                f"{c['name']}: {c['description']}"
                for c in result.get("characters", []) if c.get("name") and c.get("description")
            )

            prompts = result["image_prompts"]
            if style and "[INSERT" in style:
                # Scene prompts usually end in a period, and the template already
                # supplies one after the placeholder, so strip it to avoid "..".
                prompts = [style.replace("[INSERT YOUR SCENE / CHARACTER HERE]",
                                         p.strip().rstrip(".")) for p in prompts]
            elif style:
                prompts = [f"{p}, {style} style" for p in prompts]
            # Restate the fixed character specs on every prompt. Image models have no
            # memory between requests, so this is what actually keeps a character
            # consistent — a seed only helps when the provider honours one.
            prompts = [f"{cast}\n\nScene: {p}" for p in prompts] if cast else prompts

            project = {
                "id": project_id, "status": "in_progress", "created": datetime.now().isoformat(),
                "source_text": source, "narration": result["narration"],
                "image_prompts": prompts, "voice_id": voice_id,
                "image_model": img_model[1], "image_model_label": img_model[0],
                "image_provider": img_model[2],
                "image_style": style, "llm_provider": llm_label,
                "seed": seed, "music": os.path.basename(music_path) if music_path else None,
                "music_volume": music_volume,
                "total_cost": st.session_state.get("_llm_cost", 0),
                "steps": {s: {"status": "pending"} for s in ["script", "audio", "transcribe", "images", "assemble"]},
            }
            project["steps"]["script"] = {"status": "done"}
            _save_project(out_dir, project)
            with open(f"{out_dir}/script.txt", "w", encoding="utf-8") as f:
                f.write(result["narration"] + "\n\n")
                for i, p in enumerate(prompts, 1):
                    f.write(f"{i}. {p}\n")

            status = st.empty()
            status.markdown(_progress_html(project, "Starting…", time.time()),
                            unsafe_allow_html=True)
            try:
                _run_pipeline(project, out_dir, status)
                st.session_state["_last_output"] = out_dir
                st.session_state["_clear_source"] = True
                st.session_state["_clear_estimate"] = True
                st.toast("Video ready!")
                st.rerun()
            except Exception as e:
                project["status"] = "failed"
                _save_project(out_dir, project)
                status.markdown(
                    _stages_html(project, failed=_active_stage(project)),
                    unsafe_allow_html=True)
                st.error(f"Render failed: {e}")
                st.info("Nothing is lost — open the Projects tab and press Resume.")

        # --- last result ----------------------------------------------------
        last = st.session_state.get("_last_output")
        if last and os.path.exists(f"{last}/final.mp4"):
            _section(5, "Your video", os.path.basename(last))
            vc1, vc2 = st.columns([1, 1])
            with vc1:
                st.video(f"{last}/final.mp4")
            with vc2:
                st.download_button("Download MP4", open(f"{last}/final.mp4", "rb"),
                                   file_name=f"{os.path.basename(last)}.mp4")
                proj = json.load(open(f"{last}/project.json"))
                st.caption(f"**{len(proj.get('image_prompts', []))} scenes** · "
                           f"${proj.get('total_cost', 0):.2f} · `{os.path.basename(last)}`")

# --------------------------------------------------------------------------
# Projects
# --------------------------------------------------------------------------
with tab_projects:
    dirs = sorted([
        d for d in os.listdir(PROJECTS_DIR)
        if os.path.isdir(f"{PROJECTS_DIR}/{d}")
        and (os.path.exists(f"{PROJECTS_DIR}/{d}/project.json")
             or len(os.listdir(f"{PROJECTS_DIR}/{d}")) > 0)
    ], reverse=True)

    if not dirs:
        st.info("No projects yet. Create one in the Create tab.")
    else:
        projects = [(pid, _load_or_backfill(f"{PROJECTS_DIR}/{pid}")) for pid in dirs]
        done_count = sum(1 for _, p in projects if p.get("status") == "completed")
        spent = sum(p.get("total_cost", 0) or 0 for _, p in projects)
        s1, s2, s3 = st.columns(3)
        s1.metric("Projects", len(projects))
        s2.metric("Completed", done_count)
        s3.metric("Total spent", f"${spent:.2f}")
        st.divider()

        cols = st.columns(3)
        # --- resume ---------------------------------------------------------
        # The panel is rendered inside the card that was clicked, not after the
        # grid. It used to land below every card, which with three rows is well
        # off-screen, so pressing Resume looked like nothing happened.
        for i, (pid, proj) in enumerate(projects):
            with cols[i % 3]:
                with st.container(key=f"projcard_{pid}"):
                    steps = proj.get("steps", {})
                    done = sum(1 for s in STEP_NAMES if steps.get(s, {}).get("status") == "done")
                    pct = int(done / len(STEP_NAMES) * 100)
                    status = proj.get("status")
                    badge = {"completed": "Ready", "failed": "Failed",
                             "in_progress": "Unfinished"}.get(status, "Unknown")

                    video_path = f"{PROJECTS_DIR}/{pid}/final.mp4"
                    images = sorted(glob.glob(f"{PROJECTS_DIR}/{pid}/img_*.png"))

                    # Every card reserves the same preview height, so the grid rows
                    # line up whether the project has a video, a still, or nothing.
                    if os.path.exists(video_path):
                        st.video(video_path)
                    elif images:
                        st.image(images[0])
                    else:
                        st.markdown('<div class="proj-empty">No preview yet</div>',
                                    unsafe_allow_html=True)

                    st.markdown(
                        f'<div class="proj-title">{badge} · {pct}%</div>'
                        f'<div class="proj-meta">{pid} · ${proj.get("total_cost", 0) or 0:.2f}</div>',
                        unsafe_allow_html=True,
                    )
                    st.progress(pct / 100)

                    b1, b2 = st.columns(2)
                    with b1:
                        if status in ("failed", "in_progress"):
                            if st.button("Resume", key=f"resume_{pid}", use_container_width=True):
                                st.session_state["_resume_pid"] = pid
                    with b2:
                        if os.path.exists(video_path):
                            st.download_button("Download", open(video_path, "rb"),
                                               file_name=f"{pid}.mp4", key=f"dl_{pid}",
                                               use_container_width=True)

                    if st.session_state.get("_resume_pid") == pid:
                        slot = st.empty()
                        slot.markdown(
                            _progress_html(proj, "Resuming…", time.time()),
                            unsafe_allow_html=True)
                        try:
                            _run_pipeline(proj, f"{PROJECTS_DIR}/{pid}", slot)
                            st.session_state.pop("_resume_pid", None)
                            st.session_state["_last_output"] = f"{PROJECTS_DIR}/{pid}"
                            st.toast("Video ready!")
                            st.rerun()
                        except Exception as e:
                            proj["status"] = "failed"
                            _save_project(f"{PROJECTS_DIR}/{pid}", proj)
                            slot.markdown(
                                _stages_html(proj, failed=_active_stage(proj)),
                                unsafe_allow_html=True)
                            st.session_state.pop("_resume_pid", None)
                            st.error(f"Resume failed: {e}")

                    with st.expander("Details"):
                        if os.path.exists(f"{PROJECTS_DIR}/{pid}/audio.mp3"):
                            st.audio(f"{PROJECTS_DIR}/{pid}/audio.mp3")
                        st.markdown(_stages_html(proj), unsafe_allow_html=True)
                        st.caption(f"Model: {proj.get('image_model_label') or '—'}")
                        st.caption(f"Music: {proj.get('music') or 'none'}")
                        if proj.get("seed"):
                            st.caption(f"Seed: {proj['seed']}")
                        if st.button("Delete project", key=f"del_{pid}", use_container_width=True):
                            shutil.rmtree(f"{PROJECTS_DIR}/{pid}")
                            st.rerun()

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
with tab_settings:
    left, mid, right = st.columns([1, 3, 1])
    with mid:
        _section("", "API keys", "Read from .env at startup. Restart the app after editing.")
        for provider, (env, name, url) in API_KEYS.items():
            present = bool(os.getenv(env))
            st.markdown(f"{'✅' if present else '⚠️'} **{name}** — "
                        f"{'connected' if present else f'not set · [{url}]({url})'}")

        _section("", "Script models", "Defaults used when a provider is selected.")
        st.text_input("Ollama", value=DEFAULT_MODELS["ollama"], key="_settings_ollama")
        st.text_input("DeepSeek", value=DEFAULT_MODELS["deepseek"], key="_settings_deepseek")
        st.text_input("Kenari", value=DEFAULT_MODELS["kenari"], key="_settings_kenari")
        st.text_input("OpenRouter", value=DEFAULT_MODELS["openrouter"], key="_settings_openrouter_llm")

        _section("", "Extra image models", "One OpenRouter model ID per line.")
        st.text_area("custom image models", placeholder="black-forest-labs/flux-1.1-pro",
                     label_visibility="collapsed", key="_settings_img_models")

        _section("", "Storage")
        st.caption(f"Projects live in `{os.path.abspath(PROJECTS_DIR)}`")
        st.caption(f"Music tracks are read from `{MUSIC_DIR}`")

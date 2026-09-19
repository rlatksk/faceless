# Repository Guidelines

Streamlit app that turns source text into a 1080×1920 vertical video. Five stages run synchronously: LLM script → Edge TTS audio → Whisper word timestamps → image generation → MoviePy assembly.

13 tracked files, ~1,100 lines. `app.py` is the whole UI and orchestration; `pipeline/` holds the five stage modules.

## Architecture & Data Flow

**Single top-level script.** `app.py` has no `main()` and no `if __name__` guard — Streamlit executes it top to bottom on every rerun. `load_dotenv()` runs first (`app.py:10`), then pipeline imports are deferred with `# noqa: E402` (`app.py:12-17`).

**Strict dependency direction.** `app.py` imports `pipeline.*`; no pipeline module ever imports `app`. Keep it that way — the pipeline modules are independently runnable.

**Stage contract.** Each stage is a plain function call, in fixed order:

| Stage | Call | Writes |
|---|---|---|
| script | `generate_script(source, provider, model)` | `script.txt`, `project.json` |
| audio | `synthesize_edge(narration, voice, out)` | `audio.mp3` |
| transcribe | `transcribe(audio_path)` | `transcript.json` |
| images | `generate_images(prompts, out_dir, model, provider, progress_cb)` | `img_NNN.png` |
| assemble | `assemble(images, audio, timestamps, out)` | `final.mp4` |

`generate_images` is the only stage returning a tuple: `(paths, cost)`. All others return `None` or a value. Timestamps are `{"word": str, "start": float, "end": float}`; `assemble` consumes them directly.

**State is the filesystem.** One run = one directory `output/<YYYY-MM-DD_HHMMSS>/` containing `project.json`, `script.txt`, `audio.mp3`, `transcript.json`, `img_000.png…`, `final.mp4`. `project.json` holds `narration`, `image_prompts`, `voice_id`, `image_model`, `total_cost`, and `steps` — a dict of `{stage: {"status": "pending"|"done"}}`.

There is no database and no in-memory run state. Any stage can be re-entered from disk, which is what makes resume work.

**Provider dispatch is plain strings.** LLM: `"ollama"` | `"deepseek"` | `"openrouter"`, chosen by substring-matching the radio label (`app.py:367-372`) — the label text and the provider string are coupled. Images: `"local"` | `"openrouter"`, carried as the third element of each entry in the model list (`app.py:330-336`). Local model IDs encode inference steps as a `__N` suffix, parsed at `images.py:76-78`: `stabilityai/stable-diffusion-xl-base-1.0__10` = 10 steps.

**Resume is a file-existence heuristic.** `_load_or_backfill` (`app.py:238-270`) infers each step's status from which artifacts exist, then rewrites `project.json`. It runs for every project on every render of the Projects tab.

**Two process-global caches**, both intentional and both unbounded in lifetime: `_pipe` in `images.py:13` (diffusers pipeline, loaded lazily on first local generation) and `model` in `transcribe.py:5` (`WhisperModel`, loaded at *import* time). Importing `pipeline.transcribe` triggers a model download.

## Key Directories

```
app.py                  # Entire UI + orchestration (601 lines)
pipeline/               # One module per stage; no cross-imports
  script.py             # LLM → {narration, image_prompts}
  audio.py              # Edge TTS (async under the hood)
  transcribe.py         # faster-whisper word timestamps
  images.py             # OpenRouter API or local diffusers
  assemble.py           # MoviePy composite + word-highlight subtitles
  cost.py               # PRICES dict only — no functions
.streamlit/config.toml  # Theme + telemetry off
output/                 # Runtime artifacts, gitignored
tests/                  # Empty — see Testing & QA
```

## Development Commands

```bash
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py     # UI at localhost:8501
ruff check .             # Lint (no config file — runs on defaults)
ruff format .
```

Launch from the repo root: `load_dotenv()` and `PROJECTS_DIR = "output"` are both CWD-relative.

## Code Conventions & Common Patterns

**Style.** No type hints anywhere — match that; adding them to one module would be inconsistent. Plain module-level functions, no classes in `pipeline/`. Constants are `UPPER_SNAKE` at module top (`_FONT`, `_SUB_Y_RATIO`, `PROJECTS_DIR`, `DEFAULT_STYLE`); private helpers and private constants take a leading underscore (`_chat_script`, `_generate_local`, `_render_text`). `pipeline/__init__.py` is empty — import submodules directly, never re-export.

**Error handling.** Wrap third-party failures in `RuntimeError` with the cause interpolated: `raise RuntimeError(f"Edge TTS error: {e}")`. Use `ValueError` for bad local state (missing API key, empty image list). Catch narrowly where the failure mode is known (`requests.ConnectionError`, `json.JSONDecodeError`), broadly where it is not. User-facing text always says what to do — `"Cannot reach Ollama at http://localhost:11434. Is it running?"`.

**No retries, no backoff, no threads anywhere.** A failed stage raises and the project is marked `"failed"`. Recovery is manual, via the Resume button.

**Progress reporting.** `progress_cb(i, n, action)` is passed only into `generate_images`, called once per image *before* the request. The UI passes an inline closure capturing `bar` and `status` (`app.py:463-466`). The `action` argument is currently ignored by both closures — it is part of the signature, not a live feature.

**No Streamlit caching.** There is no `@st.cache_data` / `@st.cache_resource` anywhere. `list_edge_voices()` is a network call that re-runs on every rerun. Adding caching is a real improvement, but be deliberate: `st.cache_resource` is the right decorator for the model globals.

**Windows assumptions are load-bearing.** `_FONT = "C:/Windows/Fonts/Arial.ttf"` (`assemble.py:9`) is a hardcoded absolute path — two commits were spent fixing it. Forward-slash f-string paths (`f"{out_dir}/audio.mp3"`) and `os.path.join` are mixed freely; both work. Do not write `encoding=`-less `open()` calls for text (see Pitfalls).

**Commit messages** are bare lowercase imperative subjects — `resume: only mark images done when count matches prompts`. They are *not* Conventional Commits (15 of 16). Match the existing style unless asked otherwise.

## Important Files

| File | Why it matters |
|---|---|
| `app.py:19-21` | `PROJECTS_DIR`, `STEP_NAMES`, `DEFAULT_STYLE`. Note `STEP_NAMES` omits `"script"` while `project.json["steps"]` includes it — intentional, `script` is always `done`. |
| `app.py:238-270` | `_load_or_backfill` — the resume brain. Change here affects every project render. |
| `app.py:413-489` | Main run path: dir creation → 5 stages → failure handling. |
| `app.py:529-597` | Resume path. Mirrors the run path but gates each stage on persisted status. |
| `pipeline/images.py:16-19` | Provider fork; `local` returns `(paths, 0.0)`. |
| `pipeline/images.py:56-88` | Local diffusers path incl. `__N` step parsing. |
| `pipeline/assemble.py:17-62` | `assemble` — segment boundaries, composite, subtitle overlay. |
| `pipeline/assemble.py:125-184` | Per-word subtitle rendering: yellow highlight clip + white remainder, pop scale effect. |
| `pipeline/cost.py` | `PRICES` — 4 image keys, 3 LLM keys. No functions; estimate math lives inline in `app.py:382-398`. |
| `.env.example` | Two keys — `OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY` — plus a commented optional `HF_HOME`. |

## Runtime/Tooling Preferences

**Python 3.13** (`.venv` reports 3.13.11). No `pyproject.toml`, no `setup.py`, no lockfile, no CI, no packaging manifest of any kind — `requirements.txt` is 11 unpinned lines. Adding a pin is a behavior change; adding a *file* is a new convention, so ask first.

**FFmpeg is NOT required on PATH.** MoviePy 2.x resolves the `imageio-ffmpeg` bundled binary by default. (An `ffmpeg` on PATH happens to exist here via winget, but nothing depends on it.) Override with `FFMPEG_BINARY` or `IMAGEIO_FFMPEG_EXE` if ever needed.

**External services.** Ollama at `localhost:11434` — only for the Ollama provider, hardcoded, not env-configurable. Outbound HTTPS for Edge TTS, OpenRouter, DeepSeek, and Hugging Face model downloads. Google Fonts is fetched at render time by `_CSS` (`app.py:24`).

**GPU is CPU-only.** Installed torch is `2.12.1+cpu`, so `torch.cuda.is_available()` is always false and the `"cuda"` branch at `images.py:64` is dead in practice. Local SDXL is therefore slow; SDXL Turbo (4 steps) exists for that reason.

**Two dependencies are imported but undeclared**: `numpy` and `PIL` (`assemble.py:3-4`). They resolve transitively via `moviepy`. If you ever pin dependencies, declare them explicitly. Conversely `transformers`, `ruff`, and `pytest` are declared but never imported by source — `transformers` is a deliberate runtime gate for `diffusers`, the other two are CLI-only.

**Git identity.** Repo-local config is personal (`rlatksk <justinsalim73@gmail.com>`); the *global* config is a work address. Keep new commits on the personal identity — do not remove the repo-local `[user]` block.

## Testing & QA

**There is no test suite.** `tests/` contains zero `.py` files — only `tests/__pycache__/*.pyc` left over from a deleted suite. `pytest` collects nothing and exits 5.

The deleted `tests/test_cost.py` imported `estimate_image_cost`, `estimate_llm_cost`, and `format_estimate` from `pipeline.cost`. Those functions no longer exist (the module was reduced to the `PRICES` dict in commit `1be6159`), so the old tests are not recoverable — the API they tested is gone.

**Lint is the only working gate:** `ruff check .` covers 7 Python files on default rules and is expected to pass.

**Verifying a change** therefore means exercising the app, not running tests:

```bash
streamlit run app.py
```

Then walk the affected path in the UI. The cheapest end-to-end check is the Generate tab with the Ollama provider (free, no API key) and the local SDXL Turbo image model (free). For a stage in isolation, import and call it against an existing `output/<run>/` directory — the artifacts on disk are the real fixtures.

Streamlit hot-reloads `app.py`, but **not** imported modules. Restart the server after editing anything in `pipeline/`.

If you add tests, they need `pipeline.cost`-style pure functions to be worth writing; most of this codebase is I/O-bound and its correctness shows up in the produced artifacts.

## Pitfalls

Known live bugs and landmines. Not a backlog — just things that will bite.

- **Resume hardcodes `provider="openrouter"`** (`app.py:566`), ignoring the stored `image_model`. A local-model project cannot be resumed locally.
- **`"failed"` status is transient.** `_load_or_backfill` rewrites it to `"in_progress"` on the next render, so the ❌ icon usually disappears before you see it.
- **Local SDXL base is mispriced** at $0.07/img in the estimate: `_default_ids` (`app.py:382`) omits the `__10` variant, so it takes the custom-model branch despite running locally for free.
- **`_pipe` is never invalidated** when the selected local model changes — switching models reuses the first-loaded pipeline.
- **`script.txt` is written cp1252** on Windows (`app.py:438`, no `encoding=`). Em dashes land as `\x97`. `project.json` and `transcript.json` are safe via `json.dump`'s `ensure_ascii=True`.
- **`list_edge_voices()` is called every rerun** and only `RuntimeError` is caught (`app.py:326`) — other exception types escape the handler.

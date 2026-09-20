# Checkpoint — 2026-09-20

Handoff state for a fresh session. Everything below is verified, not assumed.

---

## Repo state

| | |
|---|---|
| Branch | `master`, **pushed**, 0 unpushed commits |
| HEAD | `f95464e assemble: render at 30fps for short-form platforms` |
| Tests | **104 passed** (`pytest`) |
| Lint | `ruff check .` clean |
| Tracked files | 27 |
| Uncommitted | upload pack (§5.5 / issue #20) — `app.py`, `pipeline/script.py`, `pipeline/assemble.py`, `tests/test_post.py`, `AGENTS.md` |
| Untracked | `plan.md` only (the original dev roadmap — deliberately not committed) |

19 commits this session, from `4a69a02` to `e597014`.

---

## What was built

### Upload pack — issue #20 / plan §5.5 (uncommitted)

One LLM call after `assemble` writes the upload metadata, plus a thumbnail composited from a
scene the video already contains (no extra image spend).

| Piece | Where |
|---|---|
| `generate_post_pack(narration, source_text, n_scenes, provider, model)` | `pipeline/script.py` |
| `make_thumbnail(image_path, text, out)` + `_wrap` | `pipeline/assemble.py` |
| `_make_post_pack` / `_write_post_pack` / `_post_panel` | `app.py` |
| Artifacts | `post.json`, `post.md`, `thumbnail.png` in the run directory |

Deliberately **not** a pipeline stage: it is outside `steps`, wrapped in a bare `except`, and
gated on `project.get("narration")`. The video is rendered and paid for by then, so a metadata
failure warns instead of failing the run.

Verified end to end (real Kenari, ~2.4s per pack): a full resume rendered the video, printed
"Writing the upload pack…", and wrote all three files. Pointed at a dead Ollama the run still
finished `Ready · 100%` with the warning shown — that is the "never a gate" property, proven.

UI: copy-ready Title / Description / Hashtags text fields plus JSON and PNG downloads, in the
Create tab's "Your video" section and in each project card's Details.

### Issues closed on GitHub (#9, #17, #18)

| Issue | Fix | Commits |
|---|---|---|
| #9 | Parallel image generation, `MAX_WORKERS=4` | `4a69a02` |
| #17 | Seed locking through all providers + `project.json` | `4a69a02` |
| #18 | Music bed, −18 dB ducking from word timestamps | `f1b717b` |

Each issue carries a comment naming its commit and stating what was **not** done
(#17's reference-image half is outstanding). Verified per-commit in a git worktree.

### UI

- **Full redesign** (`3251ead`): four numbered sections, always-visible primary
  action, cost as metrics, live stage chips. The run and resume paths were two
  ~100-line copies of the same pipeline — both now call `_run_pipeline`.
- **Amber phosphor theme** (`9e79f44`): CSS variables, IBM Plex Mono + Share Tech
  Mono, scanlines, contrast measured (fixed a 2.65 WCAG failure).
- **Gallery cards** (`ba2b14b`): equal height via `flex: 0 0 505px !important`
  (Streamlit's `flex: 1 1 0%` beats a plain `height`).
- **Weighted progress bar** (`33b2b27`): percentages weighted by real stage
  duration; resume panel renders *inside the clicked card* (it used to render
  below the whole grid, off-screen).
- **Reddit import** (`ff5bdf5`, `e597014`): post URL → story box.

### Performance — 415s → ~45s per render (9×)

| Commit | Change |
|---|---|
| `186b25b` | numpy compositor piped to ffmpeg, replacing MoviePy's per-frame chain |
| `617098c` | Pillow native `stroke_width` (was a 289-iteration Python loop) |

Measured: `_subtitle_states` 51.9s → 2.0s. Encoding is only ~9s of the total;
the rest was per-frame compositing.

**Hardware encoding was tested and rejected** — `h264_amf` on the RX 6600 gave
2.6% for 16× the file size. Also rejected: pre-resizing images (1%),
single-overlay-clip subtitles (2.3× *slower*).

---

## Bugs found and fixed

| Bug | Impact | Commit |
|---|---|---|
| Kenari returns images as `data:` URIs | Every image paid for then discarded | `60ae27a` |
| `ex.map` raises at first failure | Later successes lost, re-charged on resume | `7f8de49` |
| No retry on transient failures | One blip killed a 10-image run | `7f8de49` |
| Output was 1920×1920, not vertical | `Resize(height=1920)` + no crop | `186b25b` |
| Fenced JSON from the model | Script stage failed on valid responses | `165d49f` |
| Seed does nothing on Kenari | Silently ignored, returns 200 | `165d49f` |
| Estimate stale immediately | Settings render after the Create tab | `3251ead` |
| `_percent` returned a fraction | Bar sat at 0–1% then jumped to 100 | `33b2b27` |

---

## Known limitations (current, honest)

- **Resume cannot use a local image model** — raises a clear error.
- **`"failed"` status is transient** — `_load_or_backfill` rewrites it on render.
- **Seeds only work where the provider honours them.** Kenari ignores `seed`;
  character consistency comes from the cast sheet in the prompt instead.
- **Delete has no guard against an in-flight render.** Deleting a project mid-run
  kills the pipeline with a confusing error. Not fixed.
- **Renders block the browser tab.** A refresh loses progress; Resume recovers
  from disk, but the page does not survive.
- **Reddit rate-limits.** Importing several posts quickly returns 429 with a
  "wait a minute" message.

---

## Open items

1. **Delete guard** — no protection against deleting while rendering. Small fix.
2. **Character consistency unverified visually.** The cast sheet is in the
   prompts, but no render has been eyeballed since. One `nano-banana-2-lite`
   render (~1,500 IDR) would settle it.
3. **Reference-image consistency** for Gemini/Nano Banana — the other half of #17.

---

## Environment

| | |
|---|---|
| Python | 3.13.12 in `.venv` |
| GPU | AMD Radeon RX 6600 (no CUDA; `torch` is `+cpu`) |
| ffmpeg | bundled via `imageio-ffmpeg` |
| Kenari | Indie plan, week 26,242 IDR / month 251,242 IDR remaining |
| Services | **not running** — start with `streamlit run app.py` |

**Kenari billing:** chat draws from the plan window; **images bill from PAYG
balance**. A 10-image run costs 1,500 IDR on `nano-banana-2-lite` (150/img) or
3,000 IDR on `grok-imagine-image` (300/img).

---

## Gotchas for the next session

- **Streamlit hot-reloads `app.py` but NOT `pipeline/*.py`.** Restart after
  editing a pipeline module or you will test stale code.
- **`old.reddit.com` is a login wall**; `www.reddit.com/…/.json` returns 403 to
  scripted requests. The Atom feed at `/comments/<id>/.rss` is the working path.
- **Streamlit sets `flex: 1 1 0%`** on keyed containers — flex-basis beats
  `height`, so card sizing needs `!important`.
- **Streamlit's own CSS (`.st-bp`) loads after ours** with equal specificity;
  bare `h1` loses, so theme rules are scoped under `.stApp`.
- **`_render_text`'s stroke must use Pillow's native `stroke_width`** — the
  hand-rolled loop cost 52s per render.
- **Don't run paid image generation to verify UI or logic.** Stub
  `generate_images` and reuse images already in `output/`.

---

## Verification fixtures

Real artifacts in `output/`, usable without spending anything:

| Directory | Contents |
|---|---|
| `2026-07-06_101927` | 10 images, 230 words, 93s audio — the main test fixture |
| `2026-09-19_215216` | AITA render, 10 Kenari images, music |
| `2026-09-20_072812` | Grok run, completed |

`pipeline/assemble.py` accepts an existing run directory directly, so stages can
be exercised without the UI.

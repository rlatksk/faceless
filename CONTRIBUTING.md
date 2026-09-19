# Contributing

Thanks for wanting to help. This is a small, single-author project — the process is deliberately light. Read this before opening a pull request and you'll save us both a round trip.

## Ways to contribute

| | |
|---|---|
| **Bug reports** | Most valuable. Include the traceback, the provider/model you selected, and whether it reproduces on a fresh run. |
| **Bug fixes** | Welcome. Keep them narrow — one bug, one PR. |
| **New providers** | Welcome if they follow the existing dispatch pattern (see below). |
| **New image models** | Usually just an entry in the model list in `app.py`. Probably not worth a PR — add it under **⚙ Settings → Image Models** at runtime instead. |
| **Refactors** | Please open an issue first. This codebase is intentionally small and plain; restructuring needs a reason. |
| **Formatting-only PRs** | No. They create noise and merge conflicts. |

## Before you start

For anything larger than a one-line fix, **open an issue first**. It's much cheaper to agree on an approach than to have a PR rejected for doing the wrong thing well.

## Development setup

```bash
git clone https://github.com/rlatksk/faceless.git
cd faceless

py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # then fill in the keys you need
streamlit run app.py
```

If you don't want to spend money testing, use the free path: **Ollama** for scripting and **local SDXL Turbo** for images. That covers the full pipeline end to end with no API keys.

## Architecture rules

These aren't style preferences — the design depends on them.

**`app.py` is the only orchestrator.** It imports `pipeline.*`. No module inside `pipeline/` may import `app`, and pipeline modules must not import each other. Each one should stay independently callable.

**State lives on the filesystem.** A run is a directory in `output/`, and `project.json` plus the artifacts beside it *are* the run state. Don't introduce a database, a global registry, or in-memory run tracking — resume works precisely because everything is reconstructible from disk.

**Provider dispatch is by plain string.** LLM providers are `"ollama"` / `"deepseek"` / `"openrouter"`; image providers are `"local"` / `"openrouter"`. Adding a provider means adding a branch in the dispatch function, not building an abstraction layer. Do not introduce a plugin system or a base class for two implementations.

**Pipeline stages are plain functions.** No classes inside `pipeline/`. Each stage takes explicit arguments and returns its result — `generate_images` returns `(paths, cost)`, the rest return `None` or a value.

## Code conventions

Match the surrounding code. The codebase is consistent, and consistency beats your personal preference.

- **No type hints.** There are none anywhere. Adding them to one module makes the rest look wrong.
- **Module-level functions**, not classes, in `pipeline/`.
- **`UPPER_SNAKE` constants** at the top of the module. Private helpers and private constants get a leading underscore (`_chat_script`, `_render_text`, `_FONT`).
- **`pipeline/__init__.py` stays empty.** Import submodules directly; never re-export.
- **Error handling:** wrap third-party failures in `RuntimeError` with the cause interpolated — `raise RuntimeError(f"Edge TTS error: {e}")`. Use `ValueError` for bad local state. Catch narrowly where the failure mode is known, broadly where it isn't.
- **Error messages say what to do**, not just what broke. `"Cannot reach Ollama at http://localhost:11434. Is it running?"` is the standard to aim for.
- **No retries or backoff.** A stage that fails raises, and the project is marked failed so the user can resume. Adding silent retry loops changes the cost and failure model — raise it in an issue first.
- **Windows is the primary platform.** Don't break it. If you make something cross-platform, keep the Windows path working.

## Commit messages

Bare lowercase imperative subjects, no type prefix:

```
resume: only mark images done when count matches prompts
fix font path, aspect ratio, remove bounce effect
```

This is not Conventional Commits — the original commits all use this bare form, and consistency matters more than the convention. Keep the subject under ~72 characters. Use a body when the change spans multiple modules.

## Testing

There is currently **no test suite**. `tests/` contains only stale bytecode from a suite that was deleted along with the functions it tested. `pytest` collects nothing.

That means verification is manual. Before opening a PR:

```bash
ruff check .        # must pass — this is the only automated gate
streamlit run app.py
```

Then walk the path you changed in the UI. The cheapest full-pipeline check is Ollama + local SDXL Turbo. To exercise one stage in isolation, call it directly against an existing `output/<run>/` directory — the artifacts on disk are the real fixtures.

Remember that Streamlit hot-reloads `app.py` but not `pipeline/*.py`. Restart the server after editing pipeline modules, or you'll test stale code.

If you want to add tests: `pipeline/cost.py`-style pure functions are worth testing, and the segment-boundary logic in `pipeline/assemble.py` (`_segment_boundaries`, `_group_into_segments`) is genuinely worth testing because it's fiddly and deterministic. Most of the rest is I/O-bound, and its correctness shows up in the produced video. Please don't add tests that assert implementation details or just re-assert that a function was called.

## Pull requests

- One concern per PR. A bug fix shouldn't also reformat a file.
- Run `ruff check .` before pushing.
- Describe **what you tested and how** — which provider, which model, what you saw. "Should work" isn't testable.
- Note any change to the on-disk format of `project.json` or the output directory layout. Those are the compatibility surface; people have existing runs in `output/`.
- Don't bump dependencies as a drive-by. `requirements.txt` is intentionally unpinned.

## Reporting bugs

Include:

1. What you did, and what you expected.
2. The full traceback. The Resume path prints one via `st.code`; the main run path doesn't, so check your terminal.
3. Which LLM provider and image model you selected.
4. Your OS and Python version.
5. Whether the run directory in `output/` still exists — its `project.json` usually explains the state.

**Never paste API keys** in an issue. Redact anything starting with `sk-`.

## Security

Don't open a public issue for a security problem — email the maintainer instead. Note that this app reads API keys from `.env` and sends user-supplied text to whichever LLM and image providers you configure; if your source text is sensitive, that's worth understanding before you paste it.

## License

No license is currently declared, so contributions can't be formally licensed for redistribution yet. By opening a PR you're agreeing that your contribution can be distributed under whatever license the project adopts.

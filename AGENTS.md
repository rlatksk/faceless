# Faceless — Agent Guide

## Project

Python app that generates vertical videos via a hybrid pipeline:
- **Local**: LLM scripting (Ollama), TTS (Edge), image gen (SDXL Turbo/SDXL), transcription (faster-whisper), video assembly (MoviePy/FFmpeg)
- **API**: Image gen (OpenRouter), LLM (OpenRouter)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add `OPENROUTER_API_KEY=`.

## Commands

| Action | Command |
|--------|---------|
| Run UI | `streamlit run app.py` |
| Lint | `ruff check .` |
| Format | `ruff format .` |
| Test | `pytest` |

## Pipeline (synchronous order)

1. **Scripting** — source text → { narration, image_prompts[] } via Ollama or OpenRouter
2. **Audio** — narration → MP3 via Edge TTS (free, 47 English voices)
3. **Transcription** — MP3 → word timestamps via faster-whisper (CPU)
4. **Assets** — prompts → PNGs via OpenRouter (Nano Banana 2/Pro) or local SDXL/SDXL Turbo
5. **Assembly** — images + audio + timestamps → 1080×1920 MP4 with subtitles via MoviePy

Each run creates `output/YYYY-MM-DD_HHMMSS/final.mp4`.

## Architecture notes

- **LLM providers**: Ollama (local, `llama3`), OpenRouter (`openai/gpt-4o-mini`)
- **TTS**: Edge TTS only (`synthesize_edge`, `list_edge_voices`). Default voice "Microsoft Eric".
- **Image providers**: OpenRouter (Gemini 3.1 Flash Lite $0.035/img, Grok Imagine $0.05/img, Gemini 3.1 Flash $0.07/img, Gemini 3 Pro $0.14/img), local SDXL/SDXL Turbo (free, slow on CPU)
- **Style input**: Template style with `[INSERT YOUR SCENE / CHARACTER HERE]` placeholder, falls back to `, {style} style`
- **Cost estimate**: Runs LLM first (free for Ollama, ~$0.0002 for OpenRouter) to get image count, then shows image + LLM cost. Edge TTS is free.
- **Edge TTS**: Uses `asyncio.run()` internally — `edge_tts.list_voices()` and `edge_tts.Communicate().save()` are async
- **Windows**: Avoid Unicode box-drawing in output. `faster-whisper` symlink warning is harmless. PyTorch CUDA unavailable on AMD GPUs — CPU fallback.
- **`.env`** only needs `OPENROUTER_API_KEY`. No quotes unless key has `=`, `#`, or spaces.
- Streamlit hot-reloads `app.py` changes but doesn't always catch imported module changes — restart server after editing `pipeline/*.py`.
- **diffusers local model**: Uses `float32` on CPU. First run downloads ~7GB model. SDXL full uses 30 steps/guidance 7.5, SDXL Turbo uses 4 steps/guidance 0.0.
- **progress_cb**: `images.py` functions accept `progress_cb(i, n, action)` for per-image status updates. Passed from `app.py` via closure.

## Dependencies

`streamlit`, `requests`, `faster-whisper`, `edge-tts`, `diffusers`, `transformers`, `torch`, `torchvision`, `moviepy`, `python-dotenv`, `accelerate`, `ruff`, `pytest`

FFmpeg on PATH. Ollama running locally for Ollama provider. `streamlit run app.py` requires venv activation.

## Project structure

```
faceless/
├── app.py                  # Streamlit UI + pipeline orchestration
├── pipeline/
│   ├── __init__.py
│   ├── cost.py             # Pricing constants + estimate helpers
│   ├── script.py           # Ollama / OpenRouter LLM
│   ├── audio.py            # Edge TTS (free)
│   ├── transcribe.py       # faster-whisper word timestamps
│   ├── images.py           # OpenRouter / local SDXL
│   └── assemble.py         # MoviePy video + subtitles
├── tests/
│   └── test_cost.py        # 7 tests for cost module
├── .env.example
├── .gitignore
├── requirements.txt
└── AGENTS.md
```

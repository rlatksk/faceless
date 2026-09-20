import json
import os

import requests


def generate_script(source_text, provider="ollama", model="llama3"):
    prompt = f"""You are a video scriptwriter. Given this source text, create a narration and visual scenes.

Source: {source_text}

Return JSON:
{{"narration": "voiceover text",
  "characters": [{{"name": "short name", "description": "fixed visual description"}}],
  "image_prompts": ["prompt1", "prompt2", ...]}}

"characters" lists every recurring person in the story. Each description must be a fixed,
concrete visual spec that can be reused verbatim for every scene: apparent age, hair colour
and style, build, and the clothing they wear throughout. Invent plausible specifics where the
source does not say, then never vary them. Do not list one-off background people.

Each image_prompt describes one scene for AI image generation, and must refer to characters
by the name from "characters" rather than re-describing them, so the description stays fixed.

Describe only what is IN the scene: the subjects, their expressions and body language, the
setting, and the action. Do not name an art style, medium, rendering technique, lighting
setup, camera lens, or image quality — those are applied separately and will conflict.
Write each prompt as a single plain descriptive sentence.

Keep narration under 500 words. Generate at most 10 image prompts (one per sentence or scene change)."""

    if provider == "ollama":
        return _ollama_script(prompt, model)
    if provider == "deepseek":
        return _deepseek_script(prompt, model)
    if provider == "kenari":
        return _kenari_script(prompt, model)
    return _openrouter_script(prompt, model)


_POST_PROMPT = """You are writing the upload metadata for a short vertical video.

The narration is:
{narration}

The original source was:
{source}

Return JSON:
{{"title": "the hook, under 90 characters, no hashtags, no quotes",
  "description": "2-3 sentences that make someone want to watch. No hashtags — those are a separate field",
  "hashtags": ["#shorts", "#storytime"],
  "thumbnail_text": "2 to 4 punchy words, all caps, no punctuation",
  "thumbnail_scene": {scene_hint}}}

"thumbnail_scene" is the 1-based index of the image that would make the most
compelling thumbnail — pick the most striking, dramatic or emotionally clear
moment, not the opening establishing shot. There are {n_scenes} images.

The title must work as a hook on its own. Do not start it with "AITA" unless the
source is an AITA post. Do not use hashtags in the title."""


def generate_post_pack(narration, source_text, n_scenes, provider="ollama", model="llama3"):
    """Return upload metadata for a finished video: title, description, hashtags,
    thumbnail text, and which scene makes the best thumbnail.

    Raises RuntimeError if the model returns nothing usable, so the caller can
    decide whether that matters — it never should for an already-rendered video.
    """
    prompt = _POST_PROMPT.format(
        narration=narration, source=source_text[:1500],
        n_scenes=n_scenes, scene_hint=max(1, n_scenes),
    )
    if provider == "ollama":
        result = _ollama_script(prompt, model)
    elif provider == "deepseek":
        result = _deepseek_script(prompt, model)
    elif provider == "kenari":
        result = _kenari_script(prompt, model)
    else:
        result = _openrouter_script(prompt, model)

    title = str(result.get("title", "")).strip().strip('"')
    if not title:
        raise RuntimeError("The model returned no title for the post.")

    # Models return the tags as a list about as often as they return one string,
    # and iterating a string yields characters — so normalise before looping.
    raw_tags = result.get("hashtags", []) or []
    if isinstance(raw_tags, str):
        raw_tags = raw_tags.split()

    hashtags = []
    for tag in raw_tags:
        tag = str(tag).strip()
        if tag:
            hashtags.append(tag if tag.startswith("#") else f"#{tag}")

    try:
        scene = max(1, min(int(result.get("thumbnail_scene", 1)), max(1, n_scenes)))
    except (TypeError, ValueError):
        scene = 1

    return {
        "title": title,
        "description": str(result.get("description", "")).strip(),
        "hashtags": hashtags,
        "thumbnail_text": str(result.get("thumbnail_text", "")).strip().upper(),
        "thumbnail_scene": scene,
    }


def _ollama_script(prompt, model):
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return json.loads(resp.json()["response"])
    except (requests.ConnectionError, requests.Timeout):
        raise RuntimeError("Cannot reach Ollama at http://localhost:11434. Is it running?")
    except (json.JSONDecodeError, KeyError) as e:
        raise RuntimeError(f"Ollama returned invalid JSON: {e}")


def _deepseek_script(prompt, model="deepseek-v4-flash"):
    return _chat_script(prompt, model, "DeepSeek", "DEEPSEEK_API_KEY",
                        "https://api.deepseek.com/v1/chat/completions")


def _openrouter_script(prompt, model):
    return _chat_script(prompt, model, "OpenRouter", "OPENROUTER_API_KEY",
                        "https://openrouter.ai/api/v1/chat/completions")


def _kenari_script(prompt, model):
    return _chat_script(prompt, model, "Kenari", "KENARI_API_KEY",
                        "https://kenari.id/v1/chat/completions")


def _parse_json(content, name):
    """Parse the model's JSON, tolerating a ```json fence around it.

    Models frequently wrap the object in a fenced block even when asked for bare
    JSON, which json.loads rejects outright.
    """
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError as e:
        raise RuntimeError(f"{name} returned invalid JSON: {e}")


def _chat_script(prompt, model, name, env_key, url):
    api_key = os.environ.get(env_key)
    if not api_key:
        raise ValueError(f"{env_key} not set in .env")
    try:
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}]},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        result = _parse_json(content, name)
        result["_token_count"] = data["usage"]["total_tokens"]
        return result
    except requests.RequestException as e:
        raise RuntimeError(f"{name} API error: {e}")
    except KeyError as e:
        raise RuntimeError(f"{name} returned invalid response: {e}")

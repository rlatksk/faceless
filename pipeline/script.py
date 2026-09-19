import json
import os

import requests


def generate_script(source_text, provider="ollama", model="llama3"):
    prompt = f"""You are a video scriptwriter. Given this source text, create a narration and visual scenes.

Source: {source_text}

Return JSON: {{"narration": "voiceover text", "image_prompts": ["prompt1", "prompt2", ...]}}
Each image_prompt describes one scene for AI image generation.

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
        result = json.loads(content)
        result["_token_count"] = data["usage"]["total_tokens"]
        return result
    except requests.RequestException as e:
        raise RuntimeError(f"{name} API error: {e}")
    except (json.JSONDecodeError, KeyError) as e:
        raise RuntimeError(f"{name} returned invalid response: {e}")

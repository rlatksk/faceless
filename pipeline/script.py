import json
import os

import requests


def generate_script(source_text, provider="ollama", model="llama3"):
    prompt = f"""You are a video scriptwriter. Given this source text, create a narration and visual scenes.

Source: {source_text}

Return JSON: {{"narration": "voiceover text", "image_prompts": ["prompt1", "prompt2", ...]}}
Each image_prompt describes one scene for AI image generation.
Keep narration under 500 words. Generate 5-12 image prompts depending on story length."""

    if provider == "ollama":
        return _ollama_script(prompt, model)
    if provider == "deepseek":
        return _deepseek_script(prompt, model)
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


def _deepseek_script(prompt, model="deepseek-chat"):
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not set in .env")
    try:
        resp = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
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
        raise RuntimeError(f"DeepSeek API error: {e}")
    except (json.JSONDecodeError, KeyError) as e:
        raise RuntimeError(f"DeepSeek returned invalid response: {e}")


def _openrouter_script(prompt, model):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set in .env")
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
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
        raise RuntimeError(f"OpenRouter API error: {e}")
    except (json.JSONDecodeError, KeyError) as e:
        raise RuntimeError(f"OpenRouter returned invalid response: {e}")

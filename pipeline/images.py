import base64
import os

import requests
from dotenv import load_dotenv

load_dotenv()
hf_home = os.environ.get("HF_HOME")
if hf_home:
    os.environ["HF_HOME"] = hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(hf_home, "hub")

_pipe = None


def generate_images(prompts, output_dir, model, provider="openrouter", progress_cb=None):
    if provider == "local":
        return _generate_local(prompts, output_dir, model, progress_cb), 0.0
    return _generate_openrouter(prompts, output_dir, model, progress_cb)


def _generate_openrouter(prompts, output_dir, model, progress_cb=None):
    api_key = os.environ["OPENROUTER_API_KEY"]
    paths = []
    total_cost = 0.0
    for i, prompt in enumerate(prompts):
        path = os.path.join(output_dir, f"img_{i:03d}.png")
        if os.path.exists(path):
            paths.append(path)
            continue
        if progress_cb:
            progress_cb(i, len(prompts), "Generating")
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/images",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "prompt": prompt, "aspect_ratio": "9:16"},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            images = data.get("data", [])
            if not images:
                raise RuntimeError("No image in OpenRouter response")
            b64 = images[0]["b64_json"]
            with open(path, "wb") as f:
                f.write(base64.b64decode(b64))
            paths.append(path)
            total_cost += data.get("usage", {}).get("cost", 0)
        except requests.RequestException as e:
            body = e.response.text if e.response is not None else "no response"
            raise RuntimeError(f"OpenRouter image gen failed for prompt {i}: {e} — {body}")
    return paths, round(total_cost, 4)


def _generate_local(prompts, output_dir, model, progress_cb=None):
    global _pipe
    model_id = model.split("__", 1)[0]
    if _pipe is None:
        import torch
        from diffusers import AutoPipelineForText2Image
        _pipe = AutoPipelineForText2Image.from_pretrained(
            model_id, torch_dtype=torch.float32,
        )
        _pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    paths = []
    for i, prompt in enumerate(prompts):
        path = os.path.join(output_dir, f"img_{i:03d}.png")
        if os.path.exists(path):
            paths.append(path)
            continue
        if progress_cb:
            progress_cb(i, len(prompts), "Generating")
        try:
            if "__" in model:
                steps = int(model.split("__", 1)[1])
                guidance = 7.5
            else:
                turbo = "turbo" in model or "sdxl-turbo" in model
                steps = 4 if turbo else 30
                guidance = 0.0 if turbo else 7.5
            images = _pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=guidance).images
            images[0].save(path)
            paths.append(path)
        except Exception as e:
            raise RuntimeError(f"Local image gen failed for prompt {i}: {e}")
    return paths

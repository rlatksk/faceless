import base64
import os

import requests

_pipe = None


def generate_images(prompts, output_dir, model, provider="openrouter", progress_cb=None):
    if provider == "local":
        return _generate_local(prompts, output_dir, model, progress_cb)
    return _generate_openrouter(prompts, output_dir, model, progress_cb)


def _generate_openrouter(prompts, output_dir, model, progress_cb=None):
    api_key = os.environ["OPENROUTER_API_KEY"]
    paths = []
    for i, prompt in enumerate(prompts):
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
            path = os.path.join(output_dir, f"img_{i:03d}.png")
            with open(path, "wb") as f:
                f.write(base64.b64decode(b64))
            paths.append(path)
        except requests.RequestException as e:
            raise RuntimeError(f"OpenRouter image gen failed for prompt {i}: {e}")
    return paths


def _generate_local(prompts, output_dir, model, progress_cb=None):
    global _pipe
    if _pipe is None:
        import torch
        from diffusers import AutoPipelineForText2Image
        _pipe = AutoPipelineForText2Image.from_pretrained(
            model, torch_dtype=torch.float32,
        )
        _pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    paths = []
    for i, prompt in enumerate(prompts):
        if progress_cb:
            progress_cb(i, len(prompts), "Generating")
        try:
            turbo = "turbo" in model or "sdxl-turbo" in model
            steps = 4 if turbo else 30
            guidance = 0.0 if turbo else 7.5
            images = _pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=guidance).images
            path = os.path.join(output_dir, f"img_{i:03d}.png")
            images[0].save(path)
            paths.append(path)
        except Exception as e:
            raise RuntimeError(f"Local image gen failed for prompt {i}: {e}")
    return paths

import base64
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import requests
from dotenv import load_dotenv

from pipeline.cost import KENARI_IMAGE_IDR, KENARI_IMAGE_IDR_DEFAULT, IDR_PER_USD

load_dotenv()
hf_home = os.environ.get("HF_HOME")
if hf_home:
    os.environ["HF_HOME"] = hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(hf_home, "hub")

_pipe = None

# Concurrent in-flight image requests for the HTTP providers. Kept low because
# providers may throttle, and a 429 costs more wall-clock than it saves.
MAX_WORKERS = 4


def generate_images(prompts, output_dir, model, provider="openrouter",
                    progress_cb=None, seed=None, workers=None):
    if provider == "local":
        return _generate_local(prompts, output_dir, model, progress_cb, seed), 0.0
    if provider == "kenari":
        return _generate_kenari(prompts, output_dir, model, progress_cb, seed, workers)
    return _generate_openrouter(prompts, output_dir, model, progress_cb, seed, workers)


def _attach_script_ctx():
    """Give a worker thread the Streamlit script context.

    Streamlit widgets may only be touched from the thread running the script, so
    without this the progress bar updates from pool threads are dropped with a
    warning. Imported lazily and guarded: `pipeline/` stays runnable standalone.
    """
    try:
        from streamlit.runtime.scriptrunner import add_script_run_ctx
    except Exception:
        return
    add_script_run_ctx()


def _run_parallel(prompts, output_dir, one, progress_cb=None, workers=None):
    """Run `one(i, prompt, path)` for every prompt, at most `workers` at a time.

    Images already on disk are skipped without calling `one` — that is what makes
    resume work. `progress_cb(done, total, action)` fires once per finished image
    under a lock, and `done` counts completions rather than the prompt index, so
    concurrent finishes can never report progress going backwards.

    The first failure propagates out of the pool, matching the serial behaviour:
    the run fails and the caller resumes, reusing whatever landed on disk.
    """
    n = len(prompts)
    paths = [None] * n
    costs = [0.0] * n
    lock = threading.Lock()
    done = 0

    def task(i):
        nonlocal done
        if progress_cb:
            _attach_script_ctx()
        path = os.path.join(output_dir, f"img_{i:03d}.png")
        cost = 0.0
        if not os.path.exists(path):
            cost = one(i, prompts[i], path)
        with lock:
            paths[i] = path
            costs[i] = cost
            done += 1
            if progress_cb:
                progress_cb(done, n, "Generating")

    with ThreadPoolExecutor(max_workers=workers or MAX_WORKERS) as ex:
        list(ex.map(task, range(n)))
    return paths, round(sum(costs), 4)


def _post_image(url, headers, body, seed=None, timeout=120):
    """POST an image request, retrying once without `seed` if it is rejected.

    Seed support varies by model, so an unsupported `seed` field degrades to a
    seedless request instead of failing the run. Status is not raised here:
    callers inspect it first (Kenari's 402 needs its own message).
    """
    resp = requests.post(url, headers=headers, json=body, timeout=timeout)
    if seed is not None and resp.status_code == 400:
        retry = {k: v for k, v in body.items() if k != "seed"}
        resp = requests.post(url, headers=headers, json=retry, timeout=timeout)
    return resp


def _generate_openrouter(prompts, output_dir, model, progress_cb=None, seed=None, workers=None):
    api_key = os.environ["OPENROUTER_API_KEY"]
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def one(i, prompt, path):
        body = {"model": model, "prompt": prompt, "aspect_ratio": "9:16"}
        if seed is not None:
            body["seed"] = seed
        try:
            resp = _post_image("https://openrouter.ai/api/v1/images", headers, body, seed)
            resp.raise_for_status()
            data = resp.json()
            images = data.get("data", [])
            if not images:
                raise RuntimeError("No image in OpenRouter response")
            with open(path, "wb") as f:
                f.write(base64.b64decode(images[0]["b64_json"]))
            return data.get("usage", {}).get("cost", 0)
        except requests.RequestException as e:
            body_txt = e.response.text if e.response is not None else "no response"
            raise RuntimeError(f"OpenRouter image gen failed for prompt {i}: {e} — {body_txt}")

    return _run_parallel(prompts, output_dir, one, progress_cb, workers)


def _generate_kenari(prompts, output_dir, model, progress_cb=None, seed=None, workers=None):
    api_key = os.environ.get("KENARI_API_KEY")
    if not api_key:
        raise RuntimeError("KENARI_API_KEY not set in .env")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def one(i, prompt, path):
        body = {"model": model, "prompt": prompt, "n": 1, "size": "1024x1792"}
        if seed is not None:
            body["seed"] = seed
        try:
            resp = _post_image("https://kenari.id/v1/images/generations", headers, body,
                               seed, timeout=600)
            if resp.status_code == 402:
                raise RuntimeError(
                    "Kenari image generation needs PAYG balance — a subscription plan only "
                    "covers chat. Top up at https://kenari.id/pay (from Rp 1.000)."
                )
            resp.raise_for_status()
            images = resp.json().get("data", [])
            if not images:
                raise RuntimeError("No image in Kenari response")
            b64 = images[0].get("b64_json")
            if b64:
                with open(path, "wb") as f:
                    f.write(base64.b64decode(b64))
            else:
                img_url = images[0].get("url")
                if not img_url:
                    raise RuntimeError("Kenari response had neither b64_json nor url")
                img_resp = requests.get(img_url, timeout=120)
                img_resp.raise_for_status()
                with open(path, "wb") as f:
                    f.write(img_resp.content)
        except requests.RequestException as e:
            body_txt = e.response.text if e.response is not None else "no response"
            raise RuntimeError(f"Kenari image gen failed for prompt {i}: {e} — {body_txt}")
        return 0.0

    paths, _ = _run_parallel(prompts, output_dir, one, progress_cb, workers)
    return paths, round(len(paths) * KENARI_IMAGE_IDR.get(model, KENARI_IMAGE_IDR_DEFAULT) / IDR_PER_USD, 4)


def _generate_local(prompts, output_dir, model, progress_cb=None, seed=None):
    """Serial by design: one diffusers pipeline, no thread-safe inference."""
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
    n = len(prompts)
    for i, prompt in enumerate(prompts):
        path = os.path.join(output_dir, f"img_{i:03d}.png")
        if os.path.exists(path):
            paths.append(path)
            if progress_cb:
                progress_cb(i + 1, n, "Generating")
            continue
        try:
            if "__" in model:
                steps = int(model.split("__", 1)[1])
                guidance = 7.5
            else:
                turbo = "turbo" in model or "sdxl-turbo" in model
                steps = 4 if turbo else 30
                guidance = 0.0 if turbo else 7.5
            kwargs = {}
            if seed is not None:
                import torch
                kwargs["generator"] = torch.Generator(device=_pipe.device).manual_seed(seed)
            images = _pipe(prompt=prompt, num_inference_steps=steps,
                           guidance_scale=guidance, **kwargs).images
            images[0].save(path)
            paths.append(path)
            if progress_cb:
                progress_cb(i + 1, n, "Generating")
        except Exception as e:
            raise RuntimeError(f"Local image gen failed for prompt {i}: {e}")
    return paths

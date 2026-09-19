import glob
import json
import os
import shutil
from datetime import datetime

from dotenv import load_dotenv
import streamlit as st

load_dotenv()

from pipeline.audio import synthesize_edge, list_edge_voices  # noqa: E402
from pipeline.script import generate_script  # noqa: E402
from pipeline.transcribe import transcribe  # noqa: E402
from pipeline.images import generate_images  # noqa: E402
from pipeline.assemble import assemble  # noqa: E402
from pipeline.cost import PRICES, KENARI_IMAGE_IDR, KENARI_IMAGE_IDR_DEFAULT, IDR_PER_USD  # noqa: E402

PROJECTS_DIR = "output"
STEP_NAMES = ["audio", "transcribe", "images", "assemble"]
os.makedirs(PROJECTS_DIR, exist_ok=True)
DEFAULT_STYLE = "A dark graphic novel illustration of [INSERT YOUR SCENE / CHARACTER HERE]. Gritty indie comic book art style, thick clean black ink outlines, digital cel-shading. Dramatic cinematic lighting with deep shadows and high contrast. The characters must have large, wide-open anxious eyes with tiny pinpoint pupils, expressing shock. Suspenseful true-crime storybook aesthetic, high quality, 9:16 vertical aspect ratio."

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Teko:wght@400;600;700&family=Inter:wght@400;500;600;700&display=swap');

.stApp {
    background: #000;
}

/* film grain overlay */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
    background-repeat: repeat;
    background-size: 256px 256px;
    opacity: 0.035;
    pointer-events: none;
    z-index: 9999;
}

h1 {
    font-family: 'Teko', sans-serif;
    font-weight: 700;
    font-size: 3rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    line-height: 1;
    margin-bottom: 0;
}

h2, h3 {
    font-family: 'Teko', sans-serif;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 2px solid #1a1a1a;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Teko', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #555;
    padding: 0.5rem 1.5rem;
}

.stTabs [aria-selected="true"] {
    color: #ff2b2b !important;
}

div[data-testid="stButton"] > button {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    letter-spacing: 0.04em;
    border-radius: 2px;
    transition: all 0.15s;
    font-size: 0.8rem;
}

div[data-testid="stButton"] > button[kind="primary"] {
    background: #ff2b2b;
    border: none;
    color: #fff;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #cc0000;
    box-shadow: none;
}

div[data-testid="stButton"] > button:not([kind="primary"]) {
    background: transparent;
    border: 1px solid #333;
    color: #888;
    text-transform: uppercase;
}

div[data-testid="stButton"] > button:not([kind="primary"]):hover {
    border-color: #ff2b2b;
    color: #ff2b2b;
    background: transparent;
}

/* card-like containers */
div[data-testid="stExpander"] {
    border: 1px solid #1a1a1a;
    border-radius: 2px;
    background: #0a0a0a;
    margin-bottom: 0.75rem;
}

div[data-testid="stExpander"] summary {
    font-family: 'Inter', sans-serif;
    font-weight: 500;
}

div[data-testid="stExpander"] summary span {
    font-size: 0.85rem;
}

.stTextArea textarea, div[data-testid="stTextInput"] input {
    background: #0a0a0a;
    border: 1px solid #1a1a1a;
    border-radius: 2px;
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    color: #f5f5f5;
}

.stTextArea textarea:focus, div[data-testid="stTextInput"] input:focus {
    border-color: #ff2b2b;
    box-shadow: none;
}

.stTextArea textarea::placeholder {
    color: #444;
}

div[data-testid="stRadio"] label {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
}

div[role="progressbar"] > div {
    background: #ff2b2b !important;
    border-radius: 1px;
}

div[role="progressbar"] {
    background: #1a1a1a !important;
    border-radius: 1px;
}

div[role="alert"] {
    border-left: 3px solid #ff2b2b;
    border-radius: 0;
    background: #0a0a0a;
}

.stCodeBlock {
    background: #0a0a0a !important;
    border: 1px solid #1a1a1a;
    border-radius: 2px;
}

.stSelectBox [data-baseweb="select"] {
    border-radius: 2px;
}

.stSpinner {
    color: #ff2b2b;
}

/* settings popover */
div[data-testid="stPopover"] > div[data-testid="stButton"] > button {
    background: transparent;
    border: 1px solid #333;
    color: #888;
    font-size: 0.8rem;
    font-family: 'Inter', sans-serif;
    border-radius: 2px;
    padding: 0.25rem 0.75rem;
}
div[data-testid="stPopover"] > div[data-testid="stButton"] > button:hover {
    border-color: #ff2b2b;
    color: #ff2b2b;
}
div[data-testid="stPopoverBody"] {
    background: #0a0a0a;
    border: 1px solid #1a1a1a;
    border-radius: 2px;
    min-width: 280px;
}
div[data-testid="stPopoverBody"] a {
    color: #ff2b2b;
    text-decoration: none;
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
}
div[data-testid="stPopoverBody"] a:hover {
    color: #cc0000;
}
div[data-testid="stPopoverBody"] h5 {
    font-family: 'Teko', sans-serif;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 0.25rem;
}
div[data-testid="stPopoverBody"] hr {
    border-color: #1a1a1a;
    margin: 0.5rem 0;
}
div[data-testid="stPopoverBody"] .stCaption {
    color: #555;
    font-size: 0.75rem;
}
"""



def _save_project(path, data):
    with open(f"{path}/project.json", "w") as f:
        json.dump(data, f, indent=2)


def _load_or_backfill(path):
    pid = os.path.basename(path)
    steps = {s: {"status": "pending"} for s in STEP_NAMES}
    steps["script"] = {"status": "done"}
    if os.path.exists(f"{path}/audio.mp3"):
        steps["audio"] = {"status": "done"}
    if os.path.exists(f"{path}/transcript.json"):
        steps["transcribe"] = {"status": "done"}
    existing = sorted(glob.glob(f"{path}/img_*.png"))
    if os.path.exists(f"{path}/project.json"):
        expected = len(json.load(open(f"{path}/project.json")).get("image_prompts", []))
        if len(existing) >= expected > 0:
            steps["images"] = {"status": "done"}
    elif existing:
        steps["images"] = {"status": "done"}
    if os.path.exists(f"{path}/final.mp4"):
        steps["assemble"] = {"status": "done"}
    complete = all(steps[s]["status"] == "done" for s in STEP_NAMES)
    if os.path.exists(f"{path}/project.json"):
        proj = json.load(open(f"{path}/project.json"))
        proj["steps"] = steps
        proj["status"] = "completed" if complete else "in_progress"
        _save_project(path, proj)
        return proj
    proj = {
        "id": pid, "status": "completed" if complete else "in_progress", "created": "",
        "source_text": "", "narration": "", "image_prompts": [],
        "voice_id": "", "image_model": "", "image_model_label": "",
        "image_provider": "openrouter",
        "image_style": "", "llm_provider": "",
        "steps": steps,
    }
    _save_project(path, proj)
    return proj


st.set_page_config(page_title="Faceless", layout="centered")
st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)

col_h, col_s = st.columns([3, 1])
with col_h:
    st.markdown(
        '<h1 style="margin-bottom:0">FACELESS</h1>'
        '<p style="font-family:Inter,sans-serif;font-size:0.75rem;color:#555;'
        'letter-spacing:0.15em;text-transform:uppercase;margin-top:-0.25rem">'
        'vertical video generator</p>',
        unsafe_allow_html=True,
    )
with col_s:
    with st.popover("\u2699 Settings"):
        st.markdown("##### API Keys")
        st.markdown("[OpenRouter](https://openrouter.ai/keys)")
        st.markdown("[DeepSeek](https://platform.deepseek.com/api_keys)")
        st.divider()
        st.markdown("##### Image Models")
        st.caption("Extra OpenRouter model IDs (one per line)")
        st.text_area("custom image models", placeholder="black-forest-labs/flux-1.1-pro\n...",
                     label_visibility="collapsed", key="_settings_img_models")
        st.divider()
        st.markdown("##### LLM Models")
        st.text_input("Ollama", value="llama3", key="_settings_ollama")
        st.text_input("DeepSeek", value="deepseek-v4-flash", key="_settings_deepseek")
        st.text_input("OpenRouter", value="openai/gpt-4o-mini", key="_settings_openrouter_llm")
        st.text_input("Kenari", value="agnes-3-0-flash:free", key="_settings_kenari")

if not os.getenv("OPENROUTER_API_KEY"):
    st.warning("OPENROUTER_API_KEY not set in .env")

tab_gen, tab_projects = st.tabs(["Generate", "Projects"])

with tab_gen:
    if st.session_state.pop("_clear_source", False):
        st.session_state["source_text"] = ""
    if st.session_state.pop("_clear_style", False):
        st.session_state["style_custom"] = ""
    if st.session_state.pop("_clear_estimate", False):
        st.session_state.pop("script_result", None)
    source = st.text_area("Source script", height=140, label_visibility="collapsed",
                          placeholder="Paste your story or script here...", key="source_text")

    col1, col2 = st.columns(2)
    with col1:
        try:
            edge_voices = list_edge_voices()
            labels = {v["name"]: v["id"] for v in edge_voices}
            default_name = next((n for n in labels if "Eric" in n), list(labels.keys())[0])
            default_idx = list(labels.keys()).index(default_name)
            selected = st.selectbox("Voice", list(labels.keys()), index=default_idx)
            voice_id = labels[selected]
        except RuntimeError as e:
            st.error(f"Could not load voices: {e}")
            voice_id = "en-US-EricNeural"

    with col2:
        _img_models = [
            ("Gemini 3.1 Flash Lite \u2014 ~$0.035/img", "google/gemini-3.1-flash-lite-image", "openrouter"),
            ("Grok Imagine 1K \u2014 ~$0.05/img",          "x-ai/grok-imagine-image-quality",     "openrouter"),
            ("Gemini 3.1 Flash \u2014 ~$0.07/img",         "google/gemini-3.1-flash-image",       "openrouter"),
            ("Gemini 3 Pro \u2014 ~$0.14/img",             "google/gemini-3-pro-image",           "openrouter"),
            ("Kenari: Nano Banana 2 Lite \u2014 150 IDR/img",  "nano-banana-2-lite",       "kenari"),
            ("Kenari: Nano Banana 2 \u2014 250 IDR/img",       "nano-banana-2",            "kenari"),
            ("Kenari: Nano Banana Pro \u2014 350 IDR/img",     "nano-banana-pro",          "kenari"),
            ("Kenari: Grok Imagine \u2014 300 IDR/img",        "grok-imagine-image",       "kenari"),
            ("Local SDXL \u2014 Free (10 steps)",             "stabilityai/stable-diffusion-xl-base-1.0__10", "local"),
            ("Local SDXL Turbo \u2014 Free",               "stabilityai/sdxl-turbo",              "local"),
        ]
        _custom_raw = st.session_state.get("_settings_img_models", "")
        if _custom_raw:
            for _line in _custom_raw.strip().split("\n"):
                _line = _line.strip()
                if _line:
                    _img_models.append((_line, _line, "openrouter"))
        img_model = st.selectbox("Image Model", _img_models, format_func=lambda x: x[0], index=0)

    style_preset = st.selectbox("Style", ["Indie Dark Comic", "Custom"], label_visibility="collapsed")
    if style_preset == "Custom":
        style = st.text_input("Style tag", value="", label_visibility="collapsed",
                              placeholder="Describe your custom style...", key="style_custom")
    else:
        style = DEFAULT_STYLE

    llm_provider = st.radio(
        "LLM Provider",
        ["Kenari (API)", "DeepSeek (API)", "Ollama (local)", "OpenRouter (API)"],
        horizontal=True,
    )

    # fingerprint of current inputs to detect changes
    current_fp = {"source": source, "voice": voice_id, "img_model": img_model[1], "llm": llm_provider, "preset": style_preset, "custom_style": style if style_preset == "Custom" else "", "settings_fp": st.session_state.get("_settings_img_models", "") + st.session_state.get("_settings_ollama", "") + st.session_state.get("_settings_deepseek", "") + st.session_state.get("_settings_openrouter_llm", "") + st.session_state.get("_settings_kenari", "")}
    inputs_changed = st.session_state.get("last_fp") != current_fp
    has_estimate = st.session_state.get("script_result") is not None

    if inputs_changed or not has_estimate:
        if st.button("Estimate Cost", use_container_width=True) and source:
            with st.spinner("Running LLM..."):
                if "Ollama" in llm_provider:
                    provider, llm_model = "ollama", st.session_state.get("_settings_ollama") or "llama3"
                elif "DeepSeek" in llm_provider:
                    provider, llm_model = "deepseek", st.session_state.get("_settings_deepseek") or "deepseek-v4-flash"
                elif "Kenari" in llm_provider:
                    provider, llm_model = "kenari", st.session_state.get("_settings_kenari") or "agnes-3-0-flash:free"
                else:
                    provider, llm_model = "openrouter", st.session_state.get("_settings_openrouter_llm") or "openai/gpt-4o-mini"
                result = generate_script(source, provider=provider, model=llm_model)
                st.session_state["script_result"] = result
                st.session_state["last_fp"] = current_fp
                st.rerun()
    else:
        result = st.session_state["script_result"]
        imgs = len(result["image_prompts"])
        label = img_model[0]
        _default_ids = {"google/gemini-3.1-flash-lite-image", "x-ai/grok-imagine-image-quality", "google/gemini-3.1-flash-image", "google/gemini-3-pro-image", "stabilityai/sdxl-turbo"}
        if img_model[2] == "kenari":
            img_cost = round(imgs * KENARI_IMAGE_IDR.get(img_model[1], KENARI_IMAGE_IDR_DEFAULT) / IDR_PER_USD, 4)
            cost_note = " (IDR)"
        elif img_model[1] not in _default_ids:
            img_cost = round(imgs * PRICES["openrouter_gemini_flash"], 4)
            cost_note = " (est.)"
        else:
            model_key = {"Lite": "openrouter_gemini_lite", "Pro": "openrouter_gemini_pro",
                         "Grok": "openrouter_grok_imagine"}.get(
                next((k for k in ["Lite", "Pro", "Grok"] if k in label), ""), "openrouter_gemini_flash"
            )
            img_cost = 0 if img_model[2] == "local" else round(imgs * PRICES[model_key], 4)
            cost_note = ""
        tokens = result.get("_token_count", 0)
        if "Ollama" in llm_provider:
            llm_cost = 0
        elif "Kenari" in llm_provider:
            llm_cost = 0  # Indie plan covers chat; :free models cost nothing
        else:
            llm_key = "deepseek_llm" if "DeepSeek" in llm_provider else "openrouter_llm"
            llm_cost = round(tokens * PRICES[llm_key], 4)
        st.session_state["_llm_cost"] = llm_cost
        total = img_cost + llm_cost
        lines = [f"Images .......................... ${img_cost:.2f}{cost_note}"]
        if llm_cost:
            lines.append(f"LLM ............................. ${llm_cost:.2f}")
        lines.append("-" * 40)
        lines.append(f"Total ........................... ${total:.2f}")
        st.code("\n".join(lines), language="text")
        with st.expander("View generated script"):
            st.write("**Narration:**")
            st.write(result["narration"])
            st.write("**Image prompts:**")
            for i, p in enumerate(result["image_prompts"]):
                st.write(f"{i+1}. {p}")

        if st.button("Generate Video", type="primary", use_container_width=True):
            st.toast("Generating video...")
            project_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            out_dir = f"{PROJECTS_DIR}/{project_id}"
            os.makedirs(out_dir, exist_ok=True)
            result = st.session_state["script_result"]

            prompts = result["image_prompts"]
            if style and "[INSERT" in style:
                prompts = [style.replace("[INSERT YOUR SCENE / CHARACTER HERE]", p) for p in prompts]
            elif style:
                prompts = [f"{p}, {style} style" for p in prompts]

            project = {
                "id": project_id, "status": "in_progress", "created": datetime.now().isoformat(),
                "source_text": source, "narration": result["narration"],
                "image_prompts": prompts, "voice_id": voice_id,
                "image_model": img_model[1], "image_model_label": img_model[0],
                "image_provider": img_model[2],
                "image_style": style, "llm_provider": llm_provider,
                "total_cost": st.session_state.get("_llm_cost", 0),
                "steps": {s: {"status": "pending"} for s in ["script", "audio", "transcribe", "images", "assemble"]},
            }
            project["steps"]["script"] = {"status": "done"}
            _save_project(out_dir, project)
            with open(f"{out_dir}/script.txt", "w") as f:
                f.write(result["narration"] + "\n\n")
                for i, p in enumerate(prompts, 1):
                    f.write(f"{i}. {p}\n")

            st.session_state["_clear_source"] = True
            st.session_state["_clear_style"] = True
            st.session_state["_clear_estimate"] = True

            status = st.empty()
            bar = st.progress(0)

            try:
                status.info("Generating audio...")
                synthesize_edge(result["narration"], voice_id, f"{out_dir}/audio.mp3")
                project["steps"]["audio"] = {"status": "done"}
                _save_project(out_dir, project)
                bar.progress(20)

                status.info("Transcribing audio...")
                words = transcribe(f"{out_dir}/audio.mp3")
                json.dump(words, open(f"{out_dir}/transcript.json", "w"))
                project["steps"]["transcribe"] = {"status": "done"}
                _save_project(out_dir, project)
                bar.progress(35)

                def img_progress(i, n, action):
                    pct = 35 + 45 * (i + 1) // n
                    bar.progress(pct)
                    status.info(f"Image {i+1}/{n}")

                images, img_cost = generate_images(prompts, out_dir, img_model[1], provider=img_model[2], progress_cb=img_progress)
                if len(images) != len(prompts):
                    raise RuntimeError(f"Only {len(images)}/{len(prompts)} images generated. Check the image model.")
                project["total_cost"] += img_cost
                project["steps"]["images"] = {"status": "done"}
                _save_project(out_dir, project)
                bar.progress(80)

                status.info("Assembling video...")
                assemble(images, f"{out_dir}/audio.mp3", words, f"{out_dir}/final.mp4")
                project["steps"]["assemble"] = {"status": "done"}
                project["status"] = "completed"
                _save_project(out_dir, project)
                bar.progress(100)
                st.toast("Video ready!")
                status.success("Done!")
                st.video(f"{out_dir}/final.mp4")
            except Exception as e:
                project["status"] = "failed"
                _save_project(out_dir, project)
                st.error(f"Pipeline failed: {e}")
                st.info("Go to the Projects tab to resume.")

with tab_projects:
    st.subheader("Projects")

    dirs = sorted([
        d for d in os.listdir(PROJECTS_DIR)
        if os.path.isdir(f"{PROJECTS_DIR}/{d}")
        and (os.path.exists(f"{PROJECTS_DIR}/{d}/project.json")
             or len(os.listdir(f"{PROJECTS_DIR}/{d}")) > 0)
    ], reverse=True)

    if not dirs:
        st.info("No projects yet. Go to Generate tab to create one.")

    for pid in dirs:
        proj = _load_or_backfill(f"{PROJECTS_DIR}/{pid}")
        steps = proj.get("steps", {})
        done = sum(1 for s in STEP_NAMES if steps.get(s, {}).get("status") == "done")
        pct = int(done / len(STEP_NAMES) * 100)
        icon = {"completed": "\u2705", "failed": "\u274c", "in_progress": "\u23f3"}.get(proj.get("status"), "\u2753")
        cost = proj.get("total_cost", 0)

        with st.expander(f"**{icon} {pid}**  \u2014  {pct}%  \u2022  ${cost:.2f}"):
            col1, col2 = st.columns(2)
            with col1:
                audio_path = f"{PROJECTS_DIR}/{pid}/audio.mp3"
                if os.path.exists(audio_path):
                    st.audio(audio_path)
                imgs = sorted(glob.glob(f"{PROJECTS_DIR}/{pid}/img_*.png"))
                if imgs:
                    _cols = st.columns(3)
                    for _i, _img in enumerate(imgs):
                        with _cols[_i % 3]:
                            st.image(_img)
            with col2:
                video_path = f"{PROJECTS_DIR}/{pid}/final.mp4"
                if os.path.exists(video_path):
                    st.video(video_path)

            if proj.get("status") in ("failed", "in_progress"):
                if st.button("Resume", key=f"resume_{pid}"):
                    out_dir = f"{PROJECTS_DIR}/{pid}"
                    proj = _load_or_backfill(out_dir)
                    steps = proj["steps"]

                    status = st.empty()
                    bar = st.progress(0)
                    words = None

                    try:
                        if steps.get("audio", {}).get("status") != "done":
                            status.info("Generating audio...")
                            synthesize_edge(proj["narration"], proj["voice_id"], f"{out_dir}/audio.mp3")
                            steps["audio"] = {"status": "done"}
                            _save_project(out_dir, proj)
                        bar.progress(20)

                        if steps.get("transcribe", {}).get("status") != "done":
                            status.info("Transcribing audio...")
                            words = transcribe(f"{out_dir}/audio.mp3")
                            json.dump(words, open(f"{out_dir}/transcript.json", "w"))
                            steps["transcribe"] = {"status": "done"}
                            _save_project(out_dir, proj)
                        else:
                            words = json.load(open(f"{out_dir}/transcript.json"))
                        bar.progress(35)

                        if steps.get("images", {}).get("status") != "done":
                            prompts = proj["image_prompts"]

                            def img_progress(i, n, action):
                                pct = 35 + 45 * (i + 1) // n
                                bar.progress(pct)
                                status.info(f"Image {i+1}/{n}")

                            _prov = proj.get("image_provider") or "openrouter"
                            if _prov == "local":
                                raise RuntimeError(
                                    "This project was generated with a local image model, which "
                                    "cannot be resumed from the UI yet. Re-run it from the Generate tab."
                                )
                            images, img_cost = generate_images(prompts, out_dir, proj["image_model"], provider=_prov, progress_cb=img_progress)
                            if len(images) != len(prompts):
                                raise RuntimeError(f"Only {len(images)}/{len(prompts)} images generated. Check the image model.")
                            proj["total_cost"] = proj.get("total_cost", 0) + img_cost
                            steps["images"] = {"status": "done"}
                            _save_project(out_dir, proj)
                        else:
                            images = sorted(glob.glob(f"{out_dir}/img_*.png"))
                            expected = len(proj.get("image_prompts", []))
                            images = [p for p in images if os.path.getsize(p) > 0]
                            if len(images) != expected:
                                steps["images"] = {"status": "pending"}
                                _save_project(out_dir, proj)
                                st.rerun()
                        bar.progress(80)

                        if steps.get("assemble", {}).get("status") != "done":
                            status.info("Assembling video...")
                            if not images:
                                raise RuntimeError("No valid images to assemble")
                            assemble(images, f"{out_dir}/audio.mp3", words, f"{out_dir}/final.mp4")
                            steps["assemble"] = {"status": "done"}
                            proj["status"] = "completed"
                            _save_project(out_dir, proj)
                        bar.progress(100)
                        st.toast("Video ready!")
                        st.success("Resumed! Video saved.")
                    except Exception as e:
                        proj["status"] = "failed"
                        _save_project(out_dir, proj)
                        import traceback
                        st.error(f"Resume failed: {e}")
                        st.code(traceback.format_exc())

            if st.button("Delete project", key=f"del_{pid}"):
                shutil.rmtree(f"{PROJECTS_DIR}/{pid}")
                st.rerun()

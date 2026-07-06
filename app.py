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
from pipeline.cost import PRICES  # noqa: E402

PROJECTS_DIR = "output"
STEP_NAMES = ["audio", "transcribe", "images", "assemble"]
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
    if os.path.exists(f"{path}/transcript.json") or (os.path.exists(f"{path}/audio.mp3") and os.path.exists(f"{path}/final.mp4")):
        steps["transcribe"] = {"status": "done"}
    if sorted(glob.glob(f"{path}/img_*.png")):
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
        "image_style": "", "llm_provider": "",
        "steps": steps,
    }
    _save_project(path, proj)
    return proj


st.set_page_config(page_title="Faceless", layout="centered")
st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)

st.markdown(
    '<h1 style="margin-bottom:0">FACELESS</h1>'
    '<p style="font-family:Inter,sans-serif;font-size:0.75rem;color:#555;'
    'letter-spacing:0.15em;text-transform:uppercase;margin-top:-0.25rem">'
    'vertical video generator</p>',
    unsafe_allow_html=True,
)

if not os.getenv("OPENROUTER_API_KEY"):
    st.warning("OPENROUTER_API_KEY not set in .env")

tab_gen, tab_projects = st.tabs(["Generate", "Projects"])

with tab_gen:
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
        img_model = st.selectbox(
            "Image Model",
            [("Gemini 3.1 Flash Lite \u2014 ~$0.035/img", "google/gemini-3.1-flash-lite-image", "openrouter"),
             ("Grok Imagine 1K \u2014 ~$0.05/img",          "x-ai/grok-imagine-image-quality",     "openrouter"),
             ("Gemini 3.1 Flash \u2014 ~$0.07/img",         "google/gemini-3.1-flash-image",       "openrouter"),
             ("Gemini 3 Pro \u2014 ~$0.14/img",             "google/gemini-3-pro-image",           "openrouter"),
             ("Local SDXL Turbo \u2014 Free",               "stabilityai/sdxl-turbo",              "local")],
            format_func=lambda x: x[0], index=0,
        )

    style_preset = st.selectbox("Style", ["Indie Dark Comic", "Custom"], label_visibility="collapsed")
    if style_preset == "Custom":
        style = st.text_input("Style tag", value="", label_visibility="collapsed",
                              placeholder="Describe your custom style...", key="style_custom")
    else:
        style = DEFAULT_STYLE

    llm_provider = st.radio(
        "LLM Provider",
        ["Ollama (local)", "DeepSeek (API)", "OpenRouter (API)"],
        horizontal=True,
    )

    # fingerprint of current inputs to detect changes
    current_fp = {"source": source, "voice": voice_id, "img_model": img_model[1], "llm": llm_provider, "preset": style_preset, "custom_style": style if style_preset == "Custom" else ""}
    inputs_changed = st.session_state.get("last_fp") != current_fp
    has_estimate = st.session_state.get("script_result") is not None

    if inputs_changed or not has_estimate:
        if st.button("Estimate Cost", use_container_width=True) and source:
            with st.spinner("Running LLM..."):
                if "Ollama" in llm_provider:
                    provider, llm_model = "ollama", "llama3"
                elif "DeepSeek" in llm_provider:
                    provider, llm_model = "deepseek", "deepseek-chat"
                else:
                    provider, llm_model = "openrouter", "openai/gpt-4o-mini"
                result = generate_script(source, provider=provider, model=llm_model)
                st.session_state["script_result"] = result
                st.session_state["last_fp"] = current_fp
                st.rerun()
    else:
        result = st.session_state["script_result"]
        imgs = len(result["image_prompts"])
        label = img_model[0]
        model_key = {"Lite": "openrouter_gemini_lite", "Pro": "openrouter_gemini_pro",
                     "Grok": "openrouter_grok_imagine"}.get(
            next((k for k in ["Lite", "Pro", "Grok"] if k in label), ""), "openrouter_gemini_flash"
        )
        img_cost = 0 if img_model[2] == "local" else round(imgs * PRICES[model_key], 4)
        tokens = result.get("_token_count", 0)
        if "Ollama" in llm_provider:
            llm_cost = 0
        else:
            llm_key = "deepseek_llm" if "DeepSeek" in llm_provider else "openrouter_llm"
            llm_cost = round(tokens * PRICES[llm_key], 4)
        st.session_state["_llm_cost"] = llm_cost
        total = img_cost + llm_cost
        lines = [f"Images .......................... ${img_cost:.2f}"]
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
                "image_style": style, "llm_provider": llm_provider,
                "total_cost": st.session_state.get("_llm_cost", 0),
                "steps": {s: {"status": "pending"} for s in ["script", "audio", "transcribe", "images", "assemble"]},
            }
            project["steps"]["script"] = {"status": "done"}
            _save_project(out_dir, project)
            with open(f"{out_dir}/script.txt", "w") as f:
                f.write(result["narration"] + "\n\n")
                for p in prompts:
                    f.write(p + "\n")

            st.session_state["source_text"] = ""
            st.session_state["style_custom"] = ""
            st.session_state["cost_estimated"] = False
            st.session_state.pop("script_result", None)

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
                    st.image(imgs, width=120)
            with col2:
                video_path = f"{PROJECTS_DIR}/{pid}/final.mp4"
                if os.path.exists(video_path):
                    st.video(video_path)

            if proj.get("status") in ("failed", "in_progress"):
                if st.button("Resume", key=f"resume_{pid}"):
                    out_dir = f"{PROJECTS_DIR}/{pid}"
                    proj = json.load(open(f"{out_dir}/project.json"))
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

                            images, img_cost = generate_images(prompts, out_dir, proj["image_model"], provider="openrouter", progress_cb=img_progress)
                            proj["total_cost"] = proj.get("total_cost", 0) + img_cost
                            steps["images"] = {"status": "done"}
                            _save_project(out_dir, proj)
                        else:
                            images = sorted(glob.glob(f"{out_dir}/img_*.png"))
                        bar.progress(80)

                        if steps.get("assemble", {}).get("status") != "done":
                            status.info("Assembling video...")
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
                        st.error(f"Resume failed: {e}")

            if st.button("Delete project", key=f"del_{pid}"):
                shutil.rmtree(f"{PROJECTS_DIR}/{pid}")
                st.rerun()

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
from pipeline.cost import estimate_image_cost, estimate_llm_cost, format_estimate  # noqa: E402

PROJECTS_DIR = "output"
STEP_NAMES = ["audio", "transcribe", "images", "assemble"]

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@400;500;700&display=swap');

.stApp {
    background: #0c0c14;
}

h1, h2, h3 {
    font-family: 'Playfair Display', serif;
    font-weight: 700;
    letter-spacing: -0.02em;
}

h1 {
    background: linear-gradient(135deg, #e0dcd0 0%, #d4a574 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-style: italic;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid rgba(212, 165, 116, 0.15);
}

.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.75rem;
    color: #6b6b7a;
    transition: color 0.2s;
}

.stTabs [aria-selected="true"] {
    color: #d4a574 !important;
}

div[data-testid="stButton"] > button {
    font-family: 'DM Sans', sans-serif;
    font-weight: 700;
    letter-spacing: 0.05em;
    border-radius: 0;
    transition: all 0.2s;
}

div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #d4a574 0%, #b8864e 100%);
    border: none;
    color: #0c0c14;
    font-size: 0.85rem;
    text-transform: uppercase;
}

div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #e0b88a 0%, #c4935a 100%);
    box-shadow: 0 4px 20px rgba(212, 165, 116, 0.25);
}

div[data-testid="stButton"] > button:not([kind="primary"]) {
    background: transparent;
    border: 1px solid rgba(212, 165, 116, 0.25);
    color: #d4a574;
    font-size: 0.75rem;
}

div[data-testid="stButton"] > button:not([kind="primary"]):hover {
    border-color: #d4a574;
    background: rgba(212, 165, 116, 0.08);
}

div[data-testid="stExpander"] {
    border: 1px solid rgba(212, 165, 116, 0.1);
    border-radius: 0;
    background: rgba(24, 24, 34, 0.6);
    margin-bottom: 0.75rem;
}

div[data-testid="stExpander"] summary {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
}

div[data-testid="stExpander"] summary span {
    font-size: 0.85rem;
}

.stTextArea textarea, div[data-testid="stTextInput"] input {
    background: rgba(24, 24, 34, 0.8);
    border: 1px solid rgba(212, 165, 116, 0.12);
    border-radius: 0;
    font-family: 'DM Sans', sans-serif;
}

.stTextArea textarea:focus, div[data-testid="stTextInput"] input:focus {
    border-color: #d4a574;
    box-shadow: 0 0 0 1px rgba(212, 165, 116, 0.2);
}

div[data-testid="stRadio"] label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.8rem;
}

div[role="progressbar"] > div {
    background: linear-gradient(90deg, #d4a574, #b8864e) !important;
}

div[role="alert"] {
    border-left: 3px solid #d4a574;
    border-radius: 0;
}

section[data-testid="stSidebar"] {
    background: #0c0c14;
}
"""


def _save_project(path, data):
    with open(f"{path}/project.json", "w") as f:
        json.dump(data, f, indent=2)


def _load_project(path):
    with open(f"{path}/project.json") as f:
        return json.load(f)


st.set_page_config(page_title="Faceless", layout="centered")
st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)

if not os.getenv("OPENROUTER_API_KEY"):
    st.warning("\u26a0\ufe0f OPENROUTER_API_KEY not set in .env")

tab_gen, tab_projects = st.tabs(["Generate", "Projects"])

with tab_gen:
    st.markdown(
        '<p style="font-family: DM Sans, sans-serif; font-size: 0.85rem; '
        'color: #6b6b7a; margin-top: -0.5rem;">Source text &rarr; narration '
        '&rarr; images &rarr; video</p>',
        unsafe_allow_html=True,
    )

    source = st.text_area("Source script", height=160, label_visibility="collapsed",
                          placeholder="Paste your story or script here...")

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
            [("Gemini 3.1 Flash Lite Image \u2014 ~$0.035/img", "google/gemini-3.1-flash-lite-image", "openrouter"),
             ("Grok Imagine 1K \u2014 ~$0.05/img",               "x-ai/grok-imagine-image-quality",         "openrouter"),
             ("Gemini 3.1 Flash Image \u2014 ~$0.07/img",       "google/gemini-3.1-flash-image",            "openrouter"),
             ("Gemini 3 Pro Image \u2014 ~$0.14/img",           "google/gemini-3-pro-image",                "openrouter"),
             ("Local SDXL Turbo \u2014 Free",                   "stabilityai/sdxl-turbo",                   "local")],
            format_func=lambda x: x[0], index=0,
        )

    style = st.text_input(
        "Image style",
        value="A dark graphic novel illustration of [INSERT YOUR SCENE / CHARACTER HERE]. Gritty indie comic book art style, thick clean black ink outlines, digital cel-shading. Dramatic cinematic lighting with deep shadows and high contrast. The characters must have large, wide-open anxious eyes with tiny pinpoint pupils, expressing shock. Suspenseful true-crime storybook aesthetic, high quality, 9:16 vertical aspect ratio.",
        label_visibility="collapsed",
    )

    llm_provider = st.radio(
        "LLM Provider",
        ["Ollama (local, free)", "DeepSeek (API, paid)", "OpenRouter (API, paid)"],
        horizontal=True,
    )

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
        imgs = len(result["image_prompts"])
        label = img_model[0]
        model_key = {"Lite": "openrouter_gemini_lite", "Pro": "openrouter_gemini_pro",
                     "Grok": "openrouter_grok_imagine"}.get(
            next((k for k in ["Lite", "Pro", "Grok"] if k in label), ""), "openrouter_gemini_flash"
        )
        img_cost = 0 if img_model[2] == "local" else estimate_image_cost(imgs, model_key)
        llm_cost = 0 if "Ollama" in llm_provider else estimate_llm_cost(result.get("_token_count", 0), "deepseek" if "DeepSeek" in llm_provider else "openrouter")
        st.code(format_estimate(img_cost, llm=llm_cost), language="text")
        with st.expander("View script"):
            st.write("**Narration:**")
            st.write(result["narration"])
            st.write("**Image prompts:**")
            for i, p in enumerate(result["image_prompts"]):
                st.write(f"{i+1}. {p}")
        st.session_state["cost_estimated"] = True

    if st.session_state.get("cost_estimated") and st.button("Generate Video", type="primary", use_container_width=True):
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
            "steps": {s: {"status": "pending"} for s in ["script", "audio", "transcribe", "images", "assemble"]},
        }
        project["steps"]["script"] = {"status": "done"}
        _save_project(out_dir, project)

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

            images = generate_images(prompts, out_dir, img_model[1], provider=img_model[2], progress_cb=img_progress)
            project["steps"]["images"] = {"status": "done"}
            _save_project(out_dir, project)
            bar.progress(80)

            status.info("Assembling video...")
            assemble(images, f"{out_dir}/audio.mp3", words, f"{out_dir}/final.mp4")
            project["steps"]["assemble"] = {"status": "done"}
            project["status"] = "completed"
            _save_project(out_dir, project)
            bar.progress(100)
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
        and os.path.exists(f"{PROJECTS_DIR}/{d}/project.json")
    ], reverse=True)

    if not dirs:
        st.info("No projects yet. Go to Generate tab to create one.")

    for pid in dirs:
        proj = _load_project(f"{PROJECTS_DIR}/{pid}")
        steps = proj.get("steps", {})
        done = sum(1 for s in STEP_NAMES if steps.get(s, {}).get("status") == "done")
        pct = int(done / len(STEP_NAMES) * 100)
        icon = {"completed": "\u2705", "failed": "\u274c", "in_progress": "\u23f3"}.get(proj.get("status"), "\u2753")

        with st.expander(f"**{icon} {pid}**  \u2014  {pct}% complete"):
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
                    proj = _load_project(out_dir)
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

                            images = generate_images(prompts, out_dir, proj["image_model"], provider="openrouter", progress_cb=img_progress)
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
                        st.success("Resumed! Video saved.")
                    except Exception as e:
                        proj["status"] = "failed"
                        _save_project(out_dir, proj)
                        st.error(f"Resume failed: {e}")

            if st.button("Delete project", key=f"del_{pid}"):
                shutil.rmtree(f"{PROJECTS_DIR}/{pid}")
                st.rerun()

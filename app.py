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


def _save_project(path, data):
    with open(f"{path}/project.json", "w") as f:
        json.dump(data, f, indent=2)


def _load_project(path):
    with open(f"{path}/project.json") as f:
        return json.load(f)


st.set_page_config(page_title="Faceless", layout="centered")
st.title("Faceless — Vertical Video Generator")

if not os.getenv("OPENROUTER_API_KEY"):
    st.warning("\u26a0\ufe0f OPENROUTER_API_KEY not set in .env")

tab_gen, tab_projects = st.tabs(["Generate", "Projects"])

with tab_gen:
    source = st.text_area("Source script", height=200)

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
             ("Gemini 3.1 Flash Image \u2014 ~$0.07/img",       "google/gemini-3.1-flash-image",       "openrouter"),
             ("Gemini 3 Pro Image \u2014 ~$0.14/img",           "google/gemini-3-pro-image",           "openrouter"),
             ("Local SDXL Turbo \u2014 Free",                   "stabilityai/sdxl-turbo",               "local")],
            format_func=lambda x: x[0], index=0,
        )

    style = st.text_input(
        "Image style",
        value="A dark graphic novel illustration of [INSERT YOUR SCENE / CHARACTER HERE]. Gritty indie comic book art style, thick clean black ink outlines, digital cel-shading. Dramatic cinematic lighting with deep shadows and high contrast. The characters must have large, wide-open anxious eyes with tiny pinpoint pupils, expressing shock. Suspenseful true-crime storybook aesthetic, high quality, 9:16 vertical aspect ratio.",
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
        model_key = {"Lite": "openrouter_gemini_lite", "Pro": "openrouter_gemini_pro"}.get(
            next((k for k in ["Lite", "Pro"] if k in label), ""), "openrouter_gemini_flash"
        )
        img_cost = 0 if img_model[2] == "local" else estimate_image_cost(imgs, model_key)
        llm_cost = 0 if "Ollama" in llm_provider else estimate_llm_cost(result.get("_token_count", 0), "deepseek" if "DeepSeek" in llm_provider else "openrouter")
        st.code(format_estimate(img_cost, llm=llm_cost), language="text")
        with st.expander("View generated script"):
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
            status.text("Generating audio...")
            synthesize_edge(result["narration"], voice_id, f"{out_dir}/audio.mp3")
            project["steps"]["audio"] = {"status": "done"}
            _save_project(out_dir, project)
            bar.progress(20)

            status.text("Transcribing audio...")
            words = transcribe(f"{out_dir}/audio.mp3")
            json.dump(words, open(f"{out_dir}/transcript.json", "w"))
            project["steps"]["transcribe"] = {"status": "done"}
            _save_project(out_dir, project)
            bar.progress(35)

            def img_progress(i, n, action):
                pct = 35 + 45 * (i + 1) // n
                bar.progress(pct)
                status.text(f"Image {i+1}/{n} ({prompts[i][:60]}...)")

            images = generate_images(prompts, out_dir, img_model[1], provider=img_model[2], progress_cb=img_progress)
            project["steps"]["images"] = {"status": "done"}
            _save_project(out_dir, project)
            bar.progress(80)

            status.text("Assembling video...")
            assemble(images, f"{out_dir}/audio.mp3", words, f"{out_dir}/final.mp4")
            project["steps"]["assemble"] = {"status": "done"}
            project["status"] = "completed"
            _save_project(out_dir, project)
            bar.progress(100)
            status.text("Done!")
            st.success(f"Video saved: `{out_dir}/final.mp4`")
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

        with st.expander(f"{icon} {pid} \u2014 {pct}% complete"):
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
                            status.text("Generating audio...")
                            synthesize_edge(proj["narration"], proj["voice_id"], f"{out_dir}/audio.mp3")
                            steps["audio"] = {"status": "done"}
                            _save_project(out_dir, proj)
                        bar.progress(20)

                        if steps.get("transcribe", {}).get("status") != "done":
                            status.text("Transcribing audio...")
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
                                status.text(f"Image {i+1}/{n} ({prompts[i][:60]}...)")

                            images = generate_images(prompts, out_dir, proj["image_model"], provider="openrouter", progress_cb=img_progress)
                            steps["images"] = {"status": "done"}
                            _save_project(out_dir, proj)
                        else:
                            images = sorted(glob.glob(f"{out_dir}/img_*.png"))
                        bar.progress(80)

                        if steps.get("assemble", {}).get("status") != "done":
                            status.text("Assembling video...")
                            assemble(images, f"{out_dir}/audio.mp3", words, f"{out_dir}/final.mp4")
                            steps["assemble"] = {"status": "done"}
                            proj["status"] = "completed"
                            _save_project(out_dir, proj)
                        bar.progress(100)
                        st.success(f"Resumed! Video saved to `{out_dir}/final.mp4`")
                    except Exception as e:
                        proj["status"] = "failed"
                        _save_project(out_dir, proj)
                        st.error(f"Resume failed: {e}")

            if st.button("Delete project", key=f"del_{pid}"):
                shutil.rmtree(f"{PROJECTS_DIR}/{pid}")
                st.rerun()

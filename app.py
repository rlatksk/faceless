import os
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

st.set_page_config(page_title="Faceless", layout="centered")
st.title("Faceless — Vertical Video Generator")

if not os.getenv("OPENROUTER_API_KEY"):
    st.warning("\u26a0\ufe0f OPENROUTER_API_KEY not set in .env")

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
        format_func=lambda x: x[0], index=1,
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
    out_dir = f"output/{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)
    result = st.session_state["script_result"]

    status = st.empty()
    bar = st.progress(0)

    status.text("Generating audio...")
    synthesize_edge(result["narration"], voice_id, f"{out_dir}/audio.mp3")
    bar.progress(20)

    status.text("Transcribing audio...")
    words = transcribe(f"{out_dir}/audio.mp3")
    bar.progress(35)

    prompts = result["image_prompts"]
    if style and "[INSERT" in style:
        prompts = [style.replace("[INSERT YOUR SCENE / CHARACTER HERE]", p) for p in prompts]
    elif style:
        prompts = [f"{p}, {style} style" for p in prompts]

    img_start = 35
    img_end = 80
    img_range = img_end - img_start
    total_imgs = len(prompts)

    def img_progress(i, n, action):
        pct = img_start + img_range * (i + 1) // n
        bar.progress(pct)
        status.text(f"Image {i+1}/{n} ({prompts[i][:60]}...)")

    images = generate_images(prompts, out_dir, img_model[1], provider=img_model[2], progress_cb=img_progress)
    bar.progress(img_end)

    status.text("Assembling video...")
    assemble(images, f"{out_dir}/audio.mp3", words, f"{out_dir}/final.mp4")
    bar.progress(100)
    status.text("Done!")

    st.success(f"Video saved: `{out_dir}/final.mp4`")
    st.video(f"{out_dir}/final.mp4")

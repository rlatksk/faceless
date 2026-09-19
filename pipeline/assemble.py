import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import AudioFileClip, ImageClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.fx import Resize

_FONT = "C:/Windows/Fonts/Arial.ttf"
_FONT_SZ = 95
_WORD_GAP = 14
_SUB_Y_RATIO = 0.72
_HIGHLIGHT_DUR = 0.35
_HIGHLIGHT_COLOR = "#ffcc00"


def assemble(image_paths, audio_path, timestamps, output_path, fps=24):
    if not image_paths:
        raise ValueError("No images to assemble — image step may need to be rerun")
    timestamps = [t for t in timestamps if t["word"].strip()]
    audio = AudioFileClip(audio_path)
    seg_starts = _segment_boundaries(timestamps, len(image_paths))

    clips = []
    for i, img_path in enumerate(image_paths):
        start = seg_starts[i]
        end = seg_starts[i + 1] if i + 1 < len(seg_starts) else audio.duration
        if end - start <= 0.01:
            continue
        clip = ImageClip(img_path)
        clip = clip.with_start(start).with_duration(end - start)
        clip = Resize(height=1920).apply(clip)
        clip = clip.with_position(("center", "center"))
        clips.append(clip)

    if not clips:
        raise ValueError("No image clips with positive duration")

    video = CompositeVideoClip(clips, size=(1080, 1920))
    video = video.with_audio(audio)

    subs = _make_subtitle_clips(timestamps, video.size)
    try:
        video = CompositeVideoClip([video] + subs, size=video.size)
    except Exception:
        import sys as _sys
        print("--- subs clip list ---", file=_sys.stderr)
        for si, c in enumerate(subs):
            print(f"  sub[{si}]: start={c.start:.3f} dur={c.duration:.3f}", file=_sys.stderr)
        print("--- end ---", file=_sys.stderr)
        raise

    try:
        video.write_videofile(output_path, fps=fps, codec="libx264", audio_codec="aac")
    except Exception as e:
        import sys as _sys
        print(f"--- write_videofile failed: {e} ---", file=_sys.stderr)
        for ci, c in enumerate(clips):
            print(f"  img_clip[{ci}]: start={c.start:.3f} dur={c.duration:.3f} size={c.size if hasattr(c, 'size') else '?'}", file=_sys.stderr)
        for ci, c in enumerate(subs):
            print(f"  sub_clip[{ci}]: start={c.start:.3f} dur={c.duration:.3f} size={c.size if hasattr(c, 'size') else '?'}", file=_sys.stderr)
        raise RuntimeError(f"Video assembly failed: {e}")


def _segment_boundaries(timestamps, n_segments):
    if n_segments <= 1 or not timestamps:
        return [0.0, timestamps[-1]["end"] if timestamps else 0.0]
    total_dur = timestamps[-1]["end"]
    if total_dur <= 0:
        return [0.0, 0.0]
    seg_dur = total_dur / n_segments
    sentence_ends = [i for i, t in enumerate(timestamps)
                     if t["word"].rstrip(".,!?\"';:)]}")[-1:] in (".", "!", "?")]
    starts = [timestamps[0]["start"]]
    for i in range(1, n_segments):
        target = seg_dur * i
        last = starts[-1]
        candidates = [j for j in range(len(timestamps)) if timestamps[j]["end"] > last + 0.01]
        if not candidates:
            break
        best_idx = min(candidates, key=lambda j: abs(timestamps[j]["end"] - target))
        if sentence_ends:
            valid_sent = [idx for idx in sentence_ends if timestamps[idx]["end"] > last + 0.01]
            if valid_sent:
                closest = min(valid_sent, key=lambda idx: abs(timestamps[idx]["end"] - target))
                if abs(timestamps[closest]["end"] - target) < abs(timestamps[best_idx]["end"] - target):
                    best_idx = closest
        starts.append(timestamps[best_idx]["end"])
    while len(starts) <= n_segments:
        starts.append(timestamps[-1]["end"])
    return starts


def _word_pop(t):
    if t < 0.08:
        return 0.3 + 0.8 * t / 0.08
    elif t < 0.18:
        return 1.1 - 0.1 * (t - 0.08) / 0.1
    return 1.0


def _pop_pos(cx, cy, w, h):
    return lambda t: (cx - w * _word_pop(t) / 2, cy - h * _word_pop(t) / 2)


def _render_text(text, font_path, font_size, fill_color, stroke_width=8, stroke_color="black"):
    font = ImageFont.truetype(font_path, font_size)
    bbox = font.getbbox(text)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = stroke_width + 4
    img_w = tw + pad * 2
    img_h = th + pad * 2
    img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    tx = pad - bbox[0]
    ty = pad - bbox[1]
    for dx in range(-stroke_width, stroke_width + 1):
        for dy in range(-stroke_width, stroke_width + 1):
            if math.sqrt(dx * dx + dy * dy) <= stroke_width:
                draw.text((tx + dx, ty + dy), text, font=font, fill=stroke_color)
    draw.text((tx, ty), text, font=font, fill=fill_color)
    return np.array(img)


def _make_subtitle_clips(timestamps, video_size):
    segments = _group_into_segments(timestamps)
    clips = []
    word_idx = 0

    for text, seg_start, seg_end in segments:
        words = text.split()

        font = ImageFont.truetype(_FONT, _FONT_SZ)
        space_w = _text_dim(" ", font)[0]
        pad = space_w * 2
        word_widths = [_text_dim(w.upper(), font)[0] + pad for w in words]
        total_w = sum(word_widths) + _WORD_GAP * (len(words) - 1)
        font_sz = _FONT_SZ

        if total_w > video_size[0] - 20:
            scale = (video_size[0] - 20) / total_w
            font_sz = max(30, int(_FONT_SZ * scale))
            font = ImageFont.truetype(_FONT, font_sz)
            space_w = _text_dim(" ", font)[0]
            pad = space_w * 2
            word_widths = [_text_dim(w.upper(), font)[0] + pad for w in words]
            total_w = sum(word_widths) + _WORD_GAP * (len(words) - 1)

        ref_h = _text_dim("Test", font)[1]
        y_center = int(video_size[1] * _SUB_Y_RATIO)
        x_start = (video_size[0] - total_w) // 2

        x = x_start
        for j, w in enumerate(words):
            w_w = word_widths[j]
            ts = timestamps[word_idx]["start"] if word_idx < len(timestamps) else seg_start
            word_idx += 1
            if w_w <= 0:
                x += w_w + _WORD_GAP
                continue
            wdur = max(seg_end - ts, 0.2)
            cx = x + w_w // 2

            hl_dur = min(_HIGHLIGHT_DUR, wdur)
            hl_text = " " + w.upper() + " "
            hl_arr = _render_text(hl_text, _FONT, font_sz, _HIGHLIGHT_COLOR)
            hl = (ImageClip(hl_arr)
                  .with_start(ts).with_duration(hl_dur)
                  .with_position(_pop_pos(cx, y_center, w_w, ref_h))
                  .with_effects([Resize(_word_pop)]))
            clips.append(hl)

            if wdur > hl_dur:
                rest = wdur - hl_dur
                wh_arr = _render_text(hl_text, _FONT, font_sz, "white")
                wh = (ImageClip(wh_arr)
                      .with_start(ts + hl_dur).with_duration(rest)
                      .with_position((cx - w_w // 2, y_center - ref_h // 2)))
                clips.append(wh)

            x += w_w + _WORD_GAP

    return clips


def _text_dim(text, font):
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _group_into_segments(timestamps, max_words=5, min_duration=2.0):
    segments = []
    seg = []
    seg_start = None
    for t in timestamps:
        if seg_start is None:
            seg_start = t["start"]
        seg.append(t)
        dur = t["end"] - seg_start
        if dur >= min_duration or len(seg) >= max_words:
            text = " ".join(s["word"] for s in seg)
            segments.append((text, seg_start, max(s["end"] for s in seg)))
            seg = []
            seg_start = None
    if seg:
        text = " ".join(s["word"] for s in seg)
        segments.append((text, seg_start, max(s["end"] for s in seg)))
    return segments

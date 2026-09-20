import os
import subprocess

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import AudioFileClip
from moviepy.audio.AudioClip import CompositeAudioClip
from moviepy.audio.fx import AudioFadeOut, AudioLoop

_FONT = "C:/Windows/Fonts/Arial.ttf"
_FONT_SZ = 95
_WORD_GAP = 14
_SUB_Y_RATIO = 0.72
_HIGHLIGHT_COLOR = "#ffcc00"

FRAME_W = 1080
FRAME_H = 1920

# 30fps is what TikTok and Reels prefer; YouTube accepts it too. Subtitle state
# changes are quantised to this, so raising it also makes the word highlight
# land closer to the spoken word.
FPS = 30

MUSIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "music")

# Ducking envelope. Speech pulls the bed down by _DUCK_DB; the ramps are what
# stop that from being audible as pumping.
_DUCK_DB = -18.0
_DUCK_ATTACK = 0.05
_DUCK_RELEASE = 0.35
_MUSIC_FADE_OUT = 2.0
_MUSIC_ENV_FPS = 100.0


def assemble(image_paths, audio_path, timestamps, output_path, fps=FPS,
             music_path=None, music_volume=0.15, progress_cb=None):
    """Composite the images and subtitles, then encode.

    Frames are built directly in numpy and piped to ffmpeg rather than going
    through MoviePy's composite chain. MoviePy converts every clip to a Pillow
    image per frame, alpha-composites each one, and converts back — for a
    93-second video with ~420 subtitle clips that was 415s of rendering, of which
    only ~9s was actual encoding. Profiling put 80 of 105 seconds in that
    per-frame composite path. Rendering the same output directly takes ~50s.

    The subtitle layer is pre-rendered once per distinct state (a few hundred
    small RGBA strips) and blended with numpy slicing, which is the part that
    made the difference.
    """
    if not image_paths:
        raise ValueError("No images to assemble — image step may need to be rerun")
    timestamps = [t for t in timestamps if t["word"].strip()]
    audio = AudioFileClip(audio_path)
    if music_path:
        audio = _mix_music(audio, music_path, timestamps, music_volume)
    duration = audio.duration
    seg_starts = _segment_boundaries(timestamps, len(image_paths))

    frames = [_fit_frame(p, FRAME_W, FRAME_H) for p in image_paths]
    if not frames:
        raise ValueError("No image clips with positive duration")

    audio_tmp = _export_audio(audio, output_path)
    try:
        _encode(frames, seg_starts, duration, timestamps, audio_tmp, output_path, fps,
                progress_cb=progress_cb)
    finally:
        if os.path.exists(audio_tmp):
            os.remove(audio_tmp)


def _fit_frame(path, width, height):
    """Scale an image to cover the frame, preserving aspect, then centre-crop.

    Scaling to width alone would stretch a square image into 9:16. Scaling to
    cover means one axis overflows and gets cropped, which is what a vertical
    frame should do with a wider source.
    """
    im = Image.open(path).convert("RGB")
    scale = max(width / im.width, height / im.height)
    new_w = max(width, int(round(im.width * scale)))
    new_h = max(height, int(round(im.height * scale)))
    resized = im.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - width) // 2
    top = (new_h - height) // 2
    return np.array(resized.crop((left, top, left + width, top + height)))


def _export_audio(audio, output_path):
    """Write the mixed audio beside the output so ffmpeg can mux it."""
    path = os.path.join(os.path.dirname(os.path.abspath(output_path)),
                        "_audio_tmp.m4a")
    audio.write_audiofile(path, codec="aac", logger=None)
    return path


def _frame_owner(seg_starts, duration, fps):
    """For each frame, which image is on screen."""
    n = max(1, int(round(duration * fps)))
    times = np.arange(n) / fps
    owner = np.searchsorted(np.asarray(seg_starts[1:], dtype=float), times, side="right")
    return np.clip(owner, 0, len(seg_starts) - 2)


def _subtitle_states(timestamps, duration, fps):
    """Pre-render each distinct subtitle state, and map frames to them.

    Subtitles only change at word boundaries, so this renders a few hundred small
    strips instead of drawing text 24 times a second.
    """
    segments = _group_into_segments(timestamps)
    states = [(text, j) for text, _, _ in segments for j in range(len(text.split()))]

    n = max(1, int(round(duration * fps)))
    frame_state = np.full(n, -1, dtype=np.int32)
    for k, ts in enumerate(timestamps):
        if k >= len(states):
            break
        s = max(0, int(round(ts["start"] * fps)))
        e = max(0, int(round(ts["end"] * fps)))
        frame_state[s:e] = k

    rendered = []
    for text, highlight in states:
        strip, y = _render_line(text, highlight)
        rendered.append((strip, y))
    return rendered, frame_state


def _render_line(text, highlight_idx):
    """Draw one subtitle line as an RGBA strip, with one word highlighted."""
    words = text.split()
    font = ImageFont.truetype(_FONT, _FONT_SZ)
    space_w = _text_dim(" ", font)[0]
    pad = space_w * 2
    widths = [_text_dim(w.upper(), font)[0] + pad for w in words]
    total = sum(widths) + _WORD_GAP * (len(words) - 1)
    font_sz = _FONT_SZ

    if total > FRAME_W - 20:
        scale = (FRAME_W - 20) / total
        font_sz = max(30, int(_FONT_SZ * scale))
        font = ImageFont.truetype(_FONT, font_sz)
        space_w = _text_dim(" ", font)[0]
        pad = space_w * 2
        widths = [_text_dim(w.upper(), font)[0] + pad for w in words]
        total = sum(widths) + _WORD_GAP * (len(words) - 1)

    ref_h = _text_dim("Test", font)[1]
    y = int(FRAME_H * _SUB_Y_RATIO) - ref_h // 2
    # _render_text pads its ink inside the array, so the strip height comes from
    # the same call rather than from ref_h, or every line lands offset.
    strip_h = _render_text("Ag", _FONT, font_sz, "white").shape[0]
    strip = Image.new("RGBA", (FRAME_W, strip_h), (0, 0, 0, 0))
    x = (FRAME_W - total) // 2
    for j, (w, w_w) in enumerate(zip(words, widths)):
        colour = _HIGHLIGHT_COLOR if j == highlight_idx else "white"
        arr = _render_text(" " + w.upper() + " ", _FONT, font_sz, colour)
        strip.alpha_composite(Image.fromarray(arr), (max(0, x), 0))
        x += w_w + _WORD_GAP
    return np.array(strip), y


def _encode(frames, seg_starts, duration, timestamps, audio_path, output_path, fps,
            progress_cb=None):
    """Stream raw frames to ffmpeg, blending the subtitle layer per frame.

    `progress_cb(done, total)` fires as frames are written so the UI can show
    real progress during the longest stage rather than only at its boundaries.
    """
    owner = _frame_owner(seg_starts, duration, fps)
    subs, frame_state = _subtitle_states(timestamps, duration, fps)
    total = len(owner)

    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{FRAME_W}x{FRAME_H}",
        "-pix_fmt", "rgb24", "-r", str(fps), "-i", "-",
        "-i", audio_path,
        "-vcodec", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p",
        "-c:a", "copy", "-shortest", output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for i, img_idx in enumerate(owner):
            frame = frames[img_idx].copy()
            state = frame_state[i]
            if 0 <= state < len(subs):
                strip, y = subs[state]
                h = strip.shape[0]
                y0 = max(0, min(FRAME_H - h, y))
                region = frame[y0:y0 + h]
                alpha = strip[:, :, 3:4].astype(np.float32) / 255.0
                region[:] = (strip[:, :, :3] * alpha + region * (1 - alpha)).astype(np.uint8)
            proc.stdin.write(frame.tobytes())
            if progress_cb and (i % 12 == 0 or i == total - 1):
                progress_cb(i + 1, total)
    except BrokenPipeError as e:
        raise RuntimeError(f"Video assembly failed: ffmpeg stopped early ({e})")
    finally:
        proc.stdin.close()
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Video assembly failed: ffmpeg exited {proc.returncode}")


def _mix_music(narration, music_path, timestamps, volume):
    """Return the narration with a ducked music bed mixed under it.

    The bed is looped to the narration length, faded out, then multiplied by a
    gain envelope built from the word timestamps: `volume` during pauses, down
    by `_DUCK_DB` while any word is being spoken. Envelope ramps are applied
    before mixing so the result cannot clip at a ramp edge.
    """
    if not os.path.exists(music_path):
        raise FileNotFoundError(f"Music track not found: {music_path}")
    bed = AudioFileClip(music_path).with_effects([
        AudioLoop(duration=narration.duration),
        AudioFadeOut(_MUSIC_FADE_OUT),
    ])
    env = _duck_envelope(timestamps, narration.duration)

    def ducked(get_frame, t):
        frame = get_frame(t)
        idx = np.minimum((np.asarray(t) * _MUSIC_ENV_FPS).astype(int), len(env) - 1)
        gain = env[idx]
        return frame * (gain[:, None] if frame.ndim > 1 else gain)

    bed = bed.transform(ducked, keep_duration=True)
    return CompositeAudioClip([narration, bed.with_volume_scaled(volume)])


def _duck_envelope(timestamps, duration):
    """Per-sample gain multiplier: 1.0 in pauses, `_DUCK_DB` under speech."""
    n = max(int(duration * _MUSIC_ENV_FPS) + 1, 2)
    target = np.ones(n)
    for t in timestamps:
        s = max(int(t["start"] * _MUSIC_ENV_FPS), 0)
        e = min(int(t["end"] * _MUSIC_ENV_FPS) + 1, n)
        if e > s:
            target[s:e] = 10 ** (_DUCK_DB / 20)
    return _ramp(target, _DUCK_ATTACK, _DUCK_RELEASE)


def _ramp(target, attack, release):
    """Smooth a step envelope into an asymmetric ducking envelope.

    One pass: the gain follows the target down quickly (`attack`, so speech
    ducks immediately) and returns up slowly (`release`, so the bed does not
    pump in the gaps between words).
    """
    a = max(1.0 - 1.0 / (attack * _MUSIC_ENV_FPS), 0.0)
    r = max(1.0 - 1.0 / (release * _MUSIC_ENV_FPS), 0.0)
    out = np.empty_like(target)
    acc = target[0]
    for i, v in enumerate(target):
        coeff = a if v < acc else r
        acc = coeff * acc + (1 - coeff) * v
        out[i] = acc
    return out


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


def _render_text(text, font_path, font_size, fill_color, stroke_width=8, stroke_color="black"):
    """Draw text with a stroked outline as an RGBA array.

    Uses Pillow's native stroke rather than hand-plotting a disc of offsets. The
    old version looped 17x17 positions per call — 289 draws of the same glyph —
    which measured 38ms per call and 52s of a 100s render for 230 subtitle words.
    The native path is ~40x faster and lands ~2% more ink, so the outline reads
    slightly more even.
    """
    font = ImageFont.truetype(font_path, font_size)
    bbox = font.getbbox(text)
    pad = stroke_width + 4
    img = Image.new("RGBA", (bbox[2] - bbox[0] + pad * 2, bbox[3] - bbox[1] + pad * 2),
                    (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((pad - bbox[0], pad - bbox[1]), text, font=font, fill=fill_color,
              stroke_width=stroke_width, stroke_fill=stroke_color)
    return np.array(img)


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

from moviepy import AudioFileClip, ImageClip, TextClip, ColorClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.fx import Resize


def assemble(image_paths, audio_path, timestamps, output_path, fps=24):
    audio = AudioFileClip(audio_path)
    seg_starts = _segment_boundaries(timestamps, len(image_paths))

    clips = []
    for i, img_path in enumerate(image_paths):
        start = seg_starts[i]
        end = seg_starts[i + 1] if i + 1 < len(seg_starts) else audio.duration
        clip = ImageClip(img_path)
        clip = clip.with_start(start).with_duration(end - start)
        clip = Resize(height=1920).apply(clip)
        clip = clip.with_position(("center", "center"))
        clips.append(clip)

    video = CompositeVideoClip(clips, size=(1080, 1920))
    video = video.with_audio(audio)

    subs = _make_subtitle_clips(timestamps, video.size)
    video = CompositeVideoClip([video] + subs, size=video.size)

    video.write_videofile(output_path, fps=fps, codec="libx264", audio_codec="aac")


def _segment_boundaries(timestamps, n_segments):
    if n_segments <= 1 or not timestamps:
        return [0.0, timestamps[-1]["end"] if timestamps else 0.0]
    words_per = len(timestamps) / n_segments
    starts = [timestamps[0]["start"]]
    for i in range(1, n_segments):
        idx = int(i * words_per)
        if idx < len(timestamps):
            starts.append(timestamps[idx]["start"])
    return starts


def _make_subtitle_clips(timestamps, video_size):
    segments = _group_into_segments(timestamps)
    clips = []
    bar_w = int(video_size[0] * 0.92)
    bar_y = video_size[1] * 0.88
    for text, start, end in segments:
        dur = max(end - start, 0.3)
        txt = TextClip(
            text=text.upper(), font="Arial", font_size=80, color="white",
            stroke_color="black", stroke_width=10,
            method="caption", size=(bar_w, None),
        )
        bar = ColorClip(size=(bar_w, txt.h + 24), color=(0, 0, 0)).with_opacity(0.6)
        bar = bar.with_position(("center", bar_y)).with_start(start).with_duration(dur)

        def bounce(t):
            if t < 0.1:
                return 1.1 * t / 0.1
            if t < 0.15:
                return 1.1 - 0.1 * (t - 0.1) / 0.05
            return 1.0

        txt = txt.with_position(("center", bar_y)).with_start(start).with_duration(dur)
        txt = txt.resize(lambda t: bounce(t))
        clips.extend([bar, txt])
    return clips


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

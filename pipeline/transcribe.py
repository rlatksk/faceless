import os

from faster_whisper import WhisperModel

model = WhisperModel("base", device="cpu", compute_type="int8")


def transcribe(audio_path):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    try:
        words = []
        segments, _ = model.transcribe(audio_path, word_timestamps=True)
        for seg in segments:
            for word in seg.words:
                words.append({
                    "word": word.word.strip(),
                    "start": round(word.start, 2),
                    "end": round(word.end, 2),
                })
        return words
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}")

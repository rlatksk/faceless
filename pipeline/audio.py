import asyncio

import edge_tts


def synthesize_edge(narration, voice, output_path):
    try:
        async def _run():
            tts = edge_tts.Communicate(narration, voice)
            await tts.save(output_path)
        asyncio.run(_run())
    except Exception as e:
        raise RuntimeError(f"Edge TTS error: {e}")


def list_edge_voices():
    voices = asyncio.run(edge_tts.list_voices())
    return sorted(
        [{"id": v["ShortName"], "name": v["FriendlyName"] or v["ShortName"]}
         for v in voices if v["Locale"].startswith("en")],
        key=lambda x: x["name"],
    )

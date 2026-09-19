"""Checks for the music ducking envelope and segment grouping.

The envelope is easy to get subtly wrong — never reaching its target, or
symmetric so it pumps between words — so it gets a check. The rest of the
assembly module is I/O-bound and shows up in the rendered video.
"""

import numpy as np
import pytest

from pipeline.assemble import _DUCK_DB, _duck_envelope, _group_into_segments


class TestDuckEnvelope:
    def test_silence_is_full_volume(self):
        env = _duck_envelope([], 2.0)
        assert len(env) > 1
        assert np.allclose(env, 1.0)

    def test_reaches_target_under_speech(self):
        env = _duck_envelope([{"word": "a", "start": 1.0, "end": 3.0}], 4.0)
        assert env[200] == pytest.approx(10 ** (_DUCK_DB / 20), abs=1e-6)

    def test_duck_depth_is_18_db(self):
        """Sampled well after the release has settled, so this measures the
        target, not how far the ramp has travelled."""
        env = _duck_envelope([{"word": "a", "start": 1.0, "end": 3.0}], 6.0)
        depth_db = 20 * np.log10(env[200] / env[550])
        assert depth_db == pytest.approx(_DUCK_DB, abs=0.5)

    def test_attack_is_faster_than_release(self):
        """Equal elapsed time either side: the drop must travel further than the
        recovery, or the envelope is symmetric and will pump on short words."""
        target = 10 ** (_DUCK_DB / 20)
        env = _duck_envelope([{"word": "a", "start": 1.0, "end": 1.4}], 3.0)
        attack = (1.0 - env[105]) / (1.0 - target)      # 50 ms into speech
        release = (env[145] - target) / (1.0 - target)  # 50 ms after it ends
        assert attack > 0.5 > release

    def test_envelope_covers_the_whole_duration(self):
        env = _duck_envelope([{"word": "a", "start": 0.0, "end": 0.1}], 5.0)
        assert len(env) >= int(5.0 * 100)


class TestSegmentGrouping:
    def test_caps_words_per_segment(self):
        ts = [{"word": f"w{i}", "start": i * 0.1, "end": i * 0.1 + 0.1} for i in range(12)]
        segments = _group_into_segments(ts, max_words=5, min_duration=99.0)
        assert all(len(text.split()) <= 5 for text, _, _ in segments)

    def test_keeps_trailing_partial_segment(self):
        ts = [{"word": "a", "start": 0.0, "end": 0.1}, {"word": "b", "start": 0.1, "end": 0.2}]
        segments = _group_into_segments(ts, max_words=5, min_duration=99.0)
        assert len(segments) == 1
        assert segments[0][0] == "a b"

    def test_segments_are_contiguous_and_ordered(self):
        ts = [{"word": f"w{i}", "start": i * 0.5, "end": i * 0.5 + 0.4} for i in range(10)]
        segments = _group_into_segments(ts, max_words=3, min_duration=1.0)
        for (_, _, end), (_, start, _) in zip(segments, segments[1:]):
            assert start >= end

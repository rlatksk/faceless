"""Checks for the weighted progress percentage.

Pure arithmetic that is easy to get wrong — the first version returned a fraction
while the template formatted it as a percentage, so the bar sat at 0-1% and then
jumped to 100.
"""

import pytest

from app import _STAGE_WEIGHTS, _percent


def _project(*done):
    steps = {k: {"status": "pending"} for k in _STAGE_WEIGHTS}
    for k in done:
        steps[k] = {"status": "done"}
    return {"steps": steps}


class TestPercent:
    def test_nothing_done_is_zero(self):
        assert _percent(_project()) == 0.0

    def test_everything_done_is_one_hundred(self):
        assert _percent(_project(*_STAGE_WEIGHTS)) == 100.0

    def test_result_is_a_percentage_not_a_fraction(self):
        """The bug this guards: a fraction rendered as a percentage."""
        pct = _percent(_project("audio"))
        assert pct == pytest.approx(6.0), f"expected 6.0, got {pct}"

    def test_weights_sum_to_one(self):
        assert sum(_STAGE_WEIGHTS.values()) == pytest.approx(1.0)

    def test_partial_stage_contributes_proportionally(self):
        half = _percent(_project(), active="assemble", fraction=0.5)
        assert half == pytest.approx(_STAGE_WEIGHTS["assemble"] * 50)

    def test_partial_fraction_is_clamped(self):
        assert _percent(_project(), active="assemble", fraction=5.0) <= 100.0
        assert _percent(_project(), active="assemble", fraction=-1.0) == 0.0

    def test_never_exceeds_one_hundred(self):
        done = _project(*_STAGE_WEIGHTS)
        assert _percent(done, active="assemble", fraction=1.0) == 100.0

    def test_progress_is_monotonic_as_stages_complete(self):
        order = ["audio", "transcribe", "images", "assemble"]
        values = []
        done = []
        for stage in order:
            done.append(stage)
            values.append(_percent(_project(*done)))
        assert values == sorted(values)
        assert values[-1] == pytest.approx(100.0)

    def test_render_weighted_heavier_than_voice(self):
        """An unweighted bar would hit 40% after the two fast stages and then
        look stalled through the slow ones."""
        assert _STAGE_WEIGHTS["assemble"] > _STAGE_WEIGHTS["images"] > _STAGE_WEIGHTS["audio"]

"""Checks for the concurrent image runner.

The runner is deterministic and easy to get subtly wrong — progress that moves
backwards, a re-run that pays for images it already has — so it gets a check.
Everything else in the module is I/O-bound and shows up in the video.
"""

import os
import tempfile
import time

import pytest

from pipeline.images import _run_parallel


class TestParallelRunner:
    def _write(self, i, prompt, path):
        with open(path, "wb") as f:
            f.write(b"x")
        return 0.5

    def test_returns_paths_in_prompt_order(self):
        with tempfile.TemporaryDirectory() as d:
            paths, cost = _run_parallel([f"p{i}" for i in range(6)], d, self._write)
        assert [os.path.basename(p) for p in paths] == [f"img_{i:03d}.png" for i in range(6)]
        assert cost == pytest.approx(3.0)

    def test_runs_concurrently(self):
        def slow(i, prompt, path):
            time.sleep(0.3)
            with open(path, "wb") as f:
                f.write(b"x")
            return 0.0

        with tempfile.TemporaryDirectory() as d:
            start = time.time()
            _run_parallel([f"p{i}" for i in range(4)], d, slow, workers=4)
            elapsed = time.time() - start
        assert elapsed < 1.0, "four 0.3 s tasks should overlap, not sum to 1.2 s"

    def test_existing_images_are_never_regenerated(self):
        """This short-circuit is what makes resume reuse paid images."""
        calls = []

        def track(i, prompt, path):
            calls.append(i)
            with open(path, "wb") as f:
                f.write(b"x")
            return 0.0

        with tempfile.TemporaryDirectory() as d:
            _run_parallel(["a", "b"], d, track)
            calls.clear()
            _run_parallel(["a", "b"], d, track)
        assert calls == []

    def test_progress_counts_completions_never_goes_backwards(self):
        seen = []

        def report(done, total, action):
            seen.append((done, total))

        with tempfile.TemporaryDirectory() as d:
            _run_parallel([f"p{i}" for i in range(8)], d, self._write,
                          progress_cb=report, workers=4)
        assert [d for d, _ in seen] == list(range(1, 9))
        assert {total for _, total in seen} == {8}

    def test_failure_propagates_but_keeps_finished_images(self):
        def flaky(i, prompt, path):
            if i == 1:
                raise RuntimeError("provider said no")
            with open(path, "wb") as f:
                f.write(b"x")
            return 0.0

        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(RuntimeError, match="provider said no"):
                _run_parallel([f"p{i}" for i in range(3)], d, flaky, workers=1)
            survivors = sorted(os.listdir(d))
        assert "img_000.png" in survivors

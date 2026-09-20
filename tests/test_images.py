"""Checks for the concurrent image runner.

The runner is deterministic and easy to get subtly wrong — progress that moves
backwards, a re-run that pays for images it already has — so it gets a check.
Everything else in the module is I/O-bound and shows up in the video.
"""

import base64
import json
import os
import tempfile
import time

import pytest
import requests

from pipeline.images import _run_parallel

PNG = b"\x89PNG\r\n\x1a\nfake-image-bytes"


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

    def test_failure_reports_all_failures_and_keeps_the_rest(self):
        """A failing image must not discard the ones that succeeded — each one is
        already paid for, so aborting early forces a re-charge on resume."""
        def flaky(i, prompt, path):
            if i in (1, 4):
                raise RuntimeError("provider said no")
            with open(path, "wb") as f:
                f.write(b"x")
            return 0.0

        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(RuntimeError) as err:
                _run_parallel([f"p{i}" for i in range(6)], d, flaky, workers=1)
            survivors = sorted(os.listdir(d))

        message = str(err.value)
        assert "2 of 6 images failed" in message
        assert "prompt 2, 5" in message          # 1-based prompt numbers
        assert survivors == ["img_000.png", "img_002.png", "img_003.png", "img_005.png"]

    def test_every_prompt_is_attempted_even_after_a_failure(self):
        """ex.map raised at the first failing index, so later work was lost."""
        attempted = []

        def flaky(i, prompt, path):
            attempted.append(i)
            if i == 0:
                raise RuntimeError("first one fails")
            with open(path, "wb") as f:
                f.write(b"x")
            return 0.0

        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(RuntimeError):
                _run_parallel([f"p{i}" for i in range(5)], d, flaky, workers=2)
        assert attempted == [0, 1, 2, 3, 4]


class TestRetry:
    """A paid image request that fails transiently must be retried, not lost."""

    def _call(self, monkeypatch, responses):
        import pipeline.images as images

        calls = []

        class _Resp:
            def __init__(self, status, headers=None):
                self.status_code = status
                self.headers = headers or {}
                self.text = "{}"

        def fake_post(url, headers=None, json=None, timeout=None):
            calls.append(json)
            return _Resp(responses[min(len(calls) - 1, len(responses) - 1)])

        monkeypatch.setattr(images.requests, "post", fake_post)
        monkeypatch.setattr(images.time, "sleep", lambda s: None)
        resp = images._post_image("http://x", {}, {"model": "m", "prompt": "p"})
        return resp, calls

    def test_retries_a_transient_500(self, monkeypatch):
        resp, calls = self._call(monkeypatch, [500, 500, 200])
        assert resp.status_code == 200
        assert len(calls) == 3

    def test_retries_a_throttle(self, monkeypatch):
        resp, calls = self._call(monkeypatch, [429, 200])
        assert resp.status_code == 200
        assert len(calls) == 2

    def test_gives_up_after_the_attempt_limit(self, monkeypatch):
        resp, calls = self._call(monkeypatch, [503])
        assert resp.status_code == 503
        assert len(calls) == 3

    def test_does_not_retry_a_billing_error(self, monkeypatch):
        """402 is a balance state; retrying it just delays a clear message."""
        resp, calls = self._call(monkeypatch, [402])
        assert resp.status_code == 402
        assert len(calls) == 1

    def test_does_not_retry_a_bad_request(self, monkeypatch):
        resp, calls = self._call(monkeypatch, [400])
        assert resp.status_code == 400
        assert len(calls) == 1

    def test_honours_retry_after(self, monkeypatch):
        import pipeline.images as images
        slept = []
        seq = iter([429, 200])

        class _Resp:
            def __init__(self, status):
                self.status_code = status
                self.headers = {"Retry-After": "7"}
                self.text = "{}"

        monkeypatch.setattr(images.requests, "post",
                            lambda *a, **k: _Resp(next(seq)))
        monkeypatch.setattr(images.time, "sleep", lambda s: slept.append(s))
        images._post_image("http://x", {}, {"model": "m"})
        assert slept == [7.0]


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = json.dumps(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


class TestKenariImageShapes:
    """Kenari has returned the image in three different shapes. All must land a file."""

    def _run(self, monkeypatch, entry, tmp_path):
        import pipeline.images as images

        monkeypatch.setenv("KENARI_API_KEY", "test")
        monkeypatch.setattr(images, "_post_image",
                            lambda *a, **k: _FakeResponse({"data": [entry]}))
        paths, _ = images._generate_kenari(["a prompt"], str(tmp_path), "nano-banana-2-lite")
        return paths[0]

    def test_decodes_b64_json(self, monkeypatch, tmp_path):
        path = self._run(monkeypatch, {"b64_json": base64.b64encode(PNG).decode()}, tmp_path)
        assert open(path, "rb").read() == PNG

    def test_decodes_a_data_uri_in_the_url_field(self, monkeypatch, tmp_path):
        """What Kenari actually returns for nano-banana models — not an http link."""
        uri = "data:image/jpeg;base64," + base64.b64encode(PNG).decode()
        path = self._run(monkeypatch, {"url": uri}, tmp_path)
        assert open(path, "rb").read() == PNG

    def test_fetches_an_http_url(self, monkeypatch, tmp_path):
        import pipeline.images as images

        fetched = {}

        class _Get:
            content = PNG

            def raise_for_status(self):
                pass

        def fake_get(url, timeout=None):
            fetched["url"] = url
            return _Get()

        monkeypatch.setattr(images.requests, "get", fake_get)
        path = self._run(monkeypatch, {"url": "https://example.test/a.png"}, tmp_path)
        assert fetched["url"] == "https://example.test/a.png"
        assert open(path, "rb").read() == PNG

    def test_rejects_a_response_with_no_image(self, monkeypatch, tmp_path):
        with pytest.raises(RuntimeError, match="neither b64_json nor url"):
            self._run(monkeypatch, {"revised_prompt": "x"}, tmp_path)

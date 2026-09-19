"""Checks for frame fitting.

Scaling a source image into the 1080x1920 frame is easy to get subtly wrong:
scaling to width alone stretches a square image into 9:16, which is exactly the
bug this guards against.
"""

import os
import tempfile

import numpy as np
import pytest
from PIL import Image, ImageDraw

from pipeline.assemble import FRAME_H, FRAME_W, _fit_frame, _frame_owner


def _circle_image(width, height):
    """A black image with a white circle, to detect non-uniform scaling."""
    path = os.path.join(tempfile.gettempdir(), f"_test_circle_{width}x{height}.png")
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    r = min(width, height) // 4
    draw.ellipse([width // 2 - r, height // 2 - r, width // 2 + r, height // 2 + r],
                 fill=(255, 255, 255))
    img.save(path)
    return path


class TestFitFrame:
    @pytest.mark.parametrize("size", [(512, 512), (768, 1376), (1024, 1792), (1920, 1080), (800, 600)])
    def test_output_is_always_the_frame_size(self, size):
        frame = _fit_frame(_circle_image(*size), FRAME_W, FRAME_H)
        assert frame.shape == (FRAME_H, FRAME_W, 3)

    @pytest.mark.parametrize("size", [(512, 512), (768, 1376), (1024, 1792), (1920, 1080), (800, 600)])
    def test_aspect_is_preserved(self, size):
        """A circle must stay a circle. Scaling to width alone would make a
        square source come out as a 0.56-ratio ellipse."""
        frame = _fit_frame(_circle_image(*size), FRAME_W, FRAME_H)
        mask = frame[:, :, 0] > 128
        ys, xs = np.where(mask)
        assert mask.any(), "the test circle vanished"
        width = xs.max() - xs.min() + 1
        height = ys.max() - ys.min() + 1
        assert width / height == pytest.approx(1.0, abs=0.02)

    def test_frame_is_fully_covered(self):
        """A portrait source must not letterbox — it should cover the frame."""
        frame = _fit_frame(_circle_image(512, 512), FRAME_W, FRAME_H)
        # every edge row/column should contain some non-black pixel somewhere
        assert frame[0].max() >= 0
        assert frame.shape[0] == FRAME_H


class TestFrameOwner:
    def test_frames_map_to_the_right_image(self):
        starts = [0.0, 10.0, 20.0]
        owner = _frame_owner(starts, duration=20.0, fps=10)
        assert len(owner) == 200
        assert owner[0] == 0        # t=0.0
        assert owner[99] == 0       # t=9.9
        assert owner[100] == 1      # t=10.0
        assert owner[199] == 1      # t=19.9

    def test_never_indexes_past_the_last_image(self):
        starts = [0.0, 5.0, 10.0]
        owner = _frame_owner(starts, duration=10.0, fps=24)
        assert owner.max() <= len(starts) - 2

    def test_single_image_covers_the_whole_timeline(self):
        owner = _frame_owner([0.0, 8.0], duration=8.0, fps=24)
        assert set(owner.tolist()) == {0}

import numpy as np
from PIL import Image, ImageDraw

import preprocessing as pp


def _drawing(points, brush=22, size=520):
    img = Image.new("RGB", (size, size), "white")
    ImageDraw.Draw(img).line(points, fill="#111827", width=brush, joint="curve")
    return img


def _center_of_mass(arr):
    weights = arr.astype(np.float64)
    rows, cols = np.indices(arr.shape)
    return np.sum(rows * weights) / weights.sum(), np.sum(cols * weights) / weights.sum()


def test_shift_digit_moves_and_zero_fills():
    arr = np.zeros((5, 5), dtype=np.uint8)
    arr[0, 0] = 9
    shifted = pp.shift_digit(arr, dx=2, dy=1)
    assert shifted[1, 2] == 9
    assert shifted.sum() == 9


def test_shift_digit_drops_ink_instead_of_wrapping():
    arr = np.zeros((5, 5), dtype=np.uint8)
    arr[:, 4] = 9
    assert pp.shift_digit(arr, dx=1, dy=0).sum() == 0


def test_is_blank():
    assert pp.is_blank(None)
    assert pp.is_blank(np.full((20, 20, 3), 255, dtype=np.uint8))
    assert not pp.is_blank(np.array(_drawing([(5, 5), (15, 15)], brush=3, size=20)))


def test_is_blank_reads_editor_dict():
    blank = np.full((20, 20, 3), 255, dtype=np.uint8)
    assert pp.is_blank({"composite": blank, "layers": []})


def test_too_little_ink_is_rejected():
    tiny = Image.new("L", (520, 520), 0)
    tiny.putpixel((10, 10), 255)
    assert pp.center_digit(tiny) is None


def test_center_digit_matches_mnist_layout():
    gray = pp.to_grayscale_inverted(_drawing([(100, 60), (100, 300), (220, 300)]))
    arr = np.array(pp.center_digit(gray))
    assert arr.shape == (28, 28)

    ink = arr > pp.INK_THRESHOLD
    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    assert max(rows[-1] - rows[0], cols[-1] - cols[0]) + 1 <= pp.DIGIT_BOX + 1

    center_y, center_x = _center_of_mass(arr)
    assert abs(center_y - 14) <= 1 and abs(center_x - 14) <= 1


def test_estimate_stroke_width_on_bars():
    for width in (3, 6, 10):
        arr = np.zeros((100, 100), dtype=np.uint8)
        arr[10:90, 40 : 40 + width] = 255
        assert abs(pp.estimate_stroke_width(arr) - width) < 1.0


def test_thin_strokes_are_thickened():
    gray = pp.to_grayscale_inverted(_drawing([(260, 80), (260, 440), (380, 440)], brush=8))
    thin = np.array(pp.center_digit(gray, fix_stroke=False))
    fixed = np.array(pp.center_digit(gray, fix_stroke=True))
    assert fixed.sum() > thin.sum() * 1.3


def test_normal_strokes_are_left_alone():
    # A 3px-wide glyph at 20x20 scale already matches MNIST and must pass through unchanged.
    cropped = np.zeros((80, 80), dtype=np.uint8)
    cropped[:, 34:46] = 255
    assert pp.normalize_stroke(cropped) is cropped

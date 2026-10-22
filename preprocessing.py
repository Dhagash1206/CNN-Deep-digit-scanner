"""Turns canvas drawings into MNIST-style 28x28 inputs. Pure NumPy/PIL, no TensorFlow."""

import numpy as np
from PIL import Image, ImageFilter, ImageOps

INK_THRESHOLD = 15
MIN_INK_PIXELS = 40
DIGIT_BOX = 20
FRAME_SIZE = 28
# Stroke widths are in pixels of the final 28x28 frame. MNIST's median is about 2.6.
MIN_STROKE_WIDTH = 2.0
TARGET_STROKE_WIDTH = 2.6
# The glyph is thickened at 4x its final size so the dilation step is sub-pixel at 28x28.
WORK_SCALE = 4


def extract_image(image) -> Image.Image | None:
    if image is None:
        return None

    if isinstance(image, dict):
        data = image
        image = data.get("composite")
        if image is None:
            layers = data.get("layers", [])
            image = layers[0] if layers else None

    if image is None:
        return None

    return Image.fromarray(image.astype("uint8"))


def is_blank(image) -> bool:
    img = extract_image(image)
    if img is None:
        return True
    arr = np.array(img.convert("RGB"))
    return bool(np.all(arr > 245))


def to_grayscale_inverted(img: Image.Image) -> Image.Image:
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background

    img = img.convert("L")
    return ImageOps.autocontrast(ImageOps.invert(img), cutoff=1)


def estimate_stroke_width(arr: np.ndarray, threshold: int = 127) -> float:
    # For a stroke of width w and length L: area ~ w*L and boundary ~ 2L, so w ~ 2*area/boundary.
    ink = arr > threshold
    area = int(ink.sum())
    if area == 0:
        return 0.0
    padded = np.pad(ink, 1)
    interior = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    boundary = area - int(interior.sum())
    return 2.0 * area / max(boundary, 1)


def _crop_to_ink(arr: np.ndarray, threshold: int) -> np.ndarray | None:
    ink = arr > threshold
    if not ink.any():
        return None
    rows = np.where(np.any(ink, axis=1))[0]
    cols = np.where(np.any(ink, axis=0))[0]
    return arr[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _fit_to_box(cropped: np.ndarray, box: int) -> Image.Image:
    height, width = cropped.shape
    scale = box / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return Image.fromarray(cropped).resize(size, Image.LANCZOS)


def normalize_stroke(cropped: np.ndarray) -> np.ndarray:
    """Thickens strokes that would end up thinner than MNIST's after scaling to DIGIT_BOX."""
    work = np.array(_fit_to_box(cropped, DIGIT_BOX * WORK_SCALE))
    width = estimate_stroke_width(work) / WORK_SCALE
    if width >= MIN_STROKE_WIDTH:
        return cropped

    # A MaxFilter of size k widens a stroke by k - 1 pixels.
    grow = round((TARGET_STROKE_WIDTH - width) * WORK_SCALE)
    kernel = grow + 1 if grow % 2 == 0 else grow + 2
    padded = np.pad(work, kernel)
    thickened = np.array(Image.fromarray(padded).filter(ImageFilter.MaxFilter(kernel)))
    return _crop_to_ink(thickened, INK_THRESHOLD)


def shift_digit(arr: np.ndarray, dx: int, dy: int) -> np.ndarray:
    shifted = np.roll(arr, shift=dy, axis=0)
    shifted = np.roll(shifted, shift=dx, axis=1)
    if dy > 0:
        shifted[:dy, :] = 0
    elif dy < 0:
        shifted[dy:, :] = 0
    if dx > 0:
        shifted[:, :dx] = 0
    elif dx < 0:
        shifted[:, dx:] = 0
    return shifted


def rotate_digit(arr: np.ndarray, degrees: float) -> np.ndarray:
    img = Image.fromarray(arr.astype("float32"), mode="F")
    return np.array(img.rotate(degrees, resample=Image.BILINEAR, fillcolor=0.0))


def center_digit(gray: Image.Image, fix_stroke: bool = True) -> Image.Image | None:
    arr = np.array(gray, dtype=np.uint8)
    if int(np.sum(arr > INK_THRESHOLD)) < MIN_INK_PIXELS:
        return None

    cropped = _crop_to_ink(arr, INK_THRESHOLD)
    if fix_stroke:
        cropped = normalize_stroke(cropped)

    # MNIST convention: scale the longest side to 20px (keeping aspect ratio),
    # then shift so the center of mass sits at the middle of the 28x28 frame.
    glyph = _fit_to_box(cropped, DIGIT_BOX)
    new_width, new_height = glyph.size

    frame = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
    y_off = (FRAME_SIZE - new_height) // 2
    x_off = (FRAME_SIZE - new_width) // 2
    frame[y_off : y_off + new_height, x_off : x_off + new_width] = np.array(glyph)

    weights = frame.astype(np.float32)
    rows_idx, cols_idx = np.indices(frame.shape)
    center_y = float(np.sum(rows_idx * weights) / np.sum(weights))
    center_x = float(np.sum(cols_idx * weights) / np.sum(weights))
    half = FRAME_SIZE // 2
    # Clamp so the shift never pushes ink outside the frame.
    shift_y = int(np.clip(round(half - center_y), -y_off, FRAME_SIZE - new_height - y_off))
    shift_x = int(np.clip(round(half - center_x), -x_off, FRAME_SIZE - new_width - x_off))
    return Image.fromarray(shift_digit(frame, shift_x, shift_y))

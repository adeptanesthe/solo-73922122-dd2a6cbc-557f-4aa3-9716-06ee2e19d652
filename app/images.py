"""PNG decoding/encoding and colour parsing with strict input checks."""

from __future__ import annotations

import io
import re
import struct
from typing import Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

MAX_SIDE = 1024
MAX_UPLOAD_BYTES = 16 * 1024 * 1024  # PNG for a 1024x1024 RGB image is far smaller
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{6})$")
_TRIPLE_RE = re.compile(
    r"^rgba?\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)$"
)
_LIST_RE = re.compile(r"^\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*$")


class ImageValidationError(ValueError):
    """Raised for input that is not an 8-bit RGB PNG of an allowed size."""


def parse_color(value: str) -> Tuple[int, int, int]:
    """Parse ``#RRGGBB``, ``RRGGBB``, ``r,g,b`` or ``rgb(r,g,b)``."""
    if not isinstance(value, str):
        raise ValueError("background colour must be a string")
    text = value.strip()
    match = _HEX_RE.match(text)
    if match:
        raw = match.group(1)
        return (int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16))
    match = _TRIPLE_RE.match(text) or _LIST_RE.match(text)
    if match:
        channels = tuple(int(part) for part in match.groups())
        if all(0 <= channel <= 255 for channel in channels):
            return channels  # type: ignore[return-value]
    raise ValueError(
        "background must be #RRGGBB, RRGGBB, 'r,g,b' or 'rgb(r,g,b)' "
        "with channels in 0..255"
    )


def colour_hex(colour: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*colour)


def _check_png_header(data: bytes) -> Tuple[int, int]:
    """Validate the IHDR of an 8-bit truecolour PNG before decoding it."""
    if len(data) < 33 or not data.startswith(PNG_SIGNATURE):
        raise ImageValidationError("file is not a PNG")
    length = struct.unpack(">I", data[8:12])[0]
    chunk_type = data[12:16]
    if chunk_type != b"IHDR" or length != 13:
        raise ImageValidationError("PNG does not start with a valid IHDR chunk")
    (width, height, bit_depth, colour_type, compression, filter_method,
     interlace) = struct.unpack(">IIBBBBB", data[16:29])
    if bit_depth != 8 or colour_type != 2:
        raise ImageValidationError(
            "PNG must be 8-bit truecolour RGB (colour type 2, bit depth 8); "
            f"got colour type {colour_type}, bit depth {bit_depth}"
        )
    if compression != 0 or filter_method != 0 or interlace not in (0, 1):
        raise ImageValidationError("PNG uses unsupported compression/filter/interlace")
    if not (1 <= width <= MAX_SIDE and 1 <= height <= MAX_SIDE):
        raise ImageValidationError(
            f"image must be at most {MAX_SIDE} pixels per side; got {width}x{height}"
        )
    return width, height


def decode_rgb_png(data: bytes) -> np.ndarray:
    """Decode a PNG into an (H, W, 3) uint8 array."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise ImageValidationError(
            f"file is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB limit"
        )
    expected = _check_png_header(data)
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "PNG":
                raise ImageValidationError("file is not a PNG")
            if image.size != expected:
                raise ImageValidationError("PNG size does not match its header")
            image.load()
            if image.mode != "RGB":
                raise ImageValidationError(
                    f"image mode is {image.mode!r}; exactly RGB is required"
                )
            array = np.array(image, dtype=np.uint8)
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError(f"cannot decode PNG: {exc}") from exc
    if array.ndim != 3 or array.shape[2] != 3:
        raise ImageValidationError("decoded image is not RGB")
    return np.ascontiguousarray(array)


def encode_png(array: np.ndarray) -> bytes:
    """Encode an (H, W, 3) RGB or (H, W, 4) RGBA uint8 array as PNG."""
    contiguous = np.ascontiguousarray(array, dtype=np.uint8)
    if contiguous.ndim != 3 or contiguous.shape[2] not in (3, 4):
        raise ValueError("encode_png expects an HxWx3 or HxWx4 uint8 array")
    mode = "RGB" if contiguous.shape[2] == 3 else "RGBA"
    buffer = io.BytesIO()
    Image.fromarray(contiguous, mode=mode).save(buffer, format="PNG", compress_level=6)
    return buffer.getvalue()

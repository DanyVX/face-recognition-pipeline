"""Safe image decoding: bytes in, upright RGB ``uint8`` array out.

Handles format sniffing by magic bytes (never by file name), decompression-bomb protection,
EXIF orientation, alpha/palette/CMYK/16-bit conversion, animated images, and bounded downscaling
with a mapping back to original coordinates.

Coordinate convention: "original image space" is the EXIF-upright pixel grid at the original
resolution. :class:`LoadedImage` records how the (possibly downscaled) array maps onto it.
"""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

from frp.errors import (
    AnimatedImageError,
    CorruptImageError,
    EmptyImageError,
    ImageTooLargeError,
    UnsupportedImageFormatError,
)

DEFAULT_MAX_SIDE: Final = 4096
# 12000 x 9000 (a 108 MP photo) must load via bounded downscale; far beyond that is rejected.
DEFAULT_MAX_PIXELS: Final = 128_000_000

# Pillow's own bomb check is a second line of defense behind our explicit header check.
Image.MAX_IMAGE_PIXELS = DEFAULT_MAX_PIXELS

_EXIF_ORIENTATION_TAG: Final = 0x0112
_TRANSPOSE: Final = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}
AnimatedPolicy = Literal["first_frame", "reject"]


@dataclass(frozen=True, eq=False)
class LoadedImage:
    """A decoded image plus the information needed to map coordinates back to the original."""

    array: NDArray[np.uint8]  # RGB, shape (H, W, 3), possibly downscaled
    original_size: tuple[int, int]  # (width, height), EXIF-upright, original resolution
    format: str
    warnings: tuple[str, ...] = ()

    @property
    def scale(self) -> tuple[float, float]:
        """(sx, sy): array pixels per original pixel."""
        height, width = self.array.shape[:2]
        return width / self.original_size[0], height / self.original_size[1]

    def to_original(self, points: NDArray[np.floating]) -> NDArray[np.float32]:
        """Map (..., 2) x/y points from array coordinates to original image space."""
        sx, sy = self.scale
        scaled = np.asarray(points, dtype=np.float64) / np.array([sx, sy])
        return np.asarray(scaled, dtype=np.float32)


def sniff_format(data: bytes) -> str | None:
    """Identify the container by magic bytes. Returns ``None`` for anything unsupported."""
    if data.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "GIF"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    if data.startswith(b"BM"):
        return "BMP"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "TIFF"
    return None


def _to_rgb_array(img: Image.Image) -> NDArray[np.uint8]:
    mode = img.mode
    if mode in ("I;16", "I;16L", "I;16B", "I;16N", "I"):
        # Treated as the full 16-bit range: 65535 -> 255. Documented in docs/LIMITATIONS.md.
        raw = np.asarray(img).astype(np.float64)
        gray = np.clip(np.round(raw / 257.0), 0, 255).astype(np.uint8)
        return np.ascontiguousarray(np.stack([gray] * 3, axis=-1))
    if mode in ("RGBA", "LA", "PA") or (mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, rgba)
    return np.ascontiguousarray(np.asarray(img.convert("RGB"), dtype=np.uint8))


def load_image(
    data: bytes,
    *,
    max_side: int = DEFAULT_MAX_SIDE,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    animated: AnimatedPolicy = "first_frame",
) -> LoadedImage:
    """Decode ``data`` into an upright RGB array, downscaling so the long side is <= ``max_side``.

    Raises a subclass of :class:`frp.errors.InvalidImageError` for anything that is not a usable
    image. Never trusts file names or extensions.
    """
    if not data:
        raise EmptyImageError("Image is empty (zero bytes).")
    fmt = sniff_format(data)
    if fmt is None:
        raise UnsupportedImageFormatError(
            "Not a supported image (expected JPEG, PNG, GIF, WebP, BMP or TIFF)."
        )

    notes: list[str] = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            img = Image.open(io.BytesIO(data), formats=[fmt])
            width0, height0 = img.size
            if width0 * height0 > max_pixels:
                raise ImageTooLargeError(
                    f"Image has {width0 * height0:,} pixels; the limit is {max_pixels:,}."
                )
            if getattr(img, "n_frames", 1) > 1:
                if animated == "reject":
                    raise AnimatedImageError("Animated images are not accepted.")
                notes.append("animated image: using the first frame")
            orientation = int(img.getexif().get(_EXIF_ORIENTATION_TAG, 1))
            if max(width0, height0) > max_side:
                img.thumbnail((max_side, max_side), Image.Resampling.BILINEAR, reducing_gap=2.0)
            img.load()
            rgb_image = Image.fromarray(_to_rgb_array(img))
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageTooLargeError("Image is too large to decode safely.") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, EOFError) as exc:
        raise CorruptImageError("Image is corrupt or truncated.") from exc

    if orientation in _TRANSPOSE:
        rgb_image = rgb_image.transpose(_TRANSPOSE[orientation])
    swaps_axes = orientation in (5, 6, 7, 8)
    original_size = (height0, width0) if swaps_axes else (width0, height0)
    return LoadedImage(
        array=np.ascontiguousarray(np.asarray(rgb_image, dtype=np.uint8)),
        original_size=original_size,
        format=fmt,
        warnings=tuple(notes),
    )

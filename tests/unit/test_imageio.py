"""Image loading edge cases from the M1 spec: every case maps to at least one test here."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from frp.errors import (
    AnimatedImageError,
    CorruptImageError,
    EmptyImageError,
    ImageTooLargeError,
    UnsupportedImageFormatError,
)
from frp.imageio import load_image, sniff_format


def encode(img: Image.Image, fmt: str, **kwargs: object) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


def corner_marked(width: int = 40, height: int = 20) -> Image.Image:
    """Asymmetric RGB image: red top-left pixel block, everything else mid-gray."""
    arr = np.full((height, width, 3), 128, dtype=np.uint8)
    arr[:4, :4] = (255, 0, 0)
    return Image.fromarray(arr)


# --- trivial rejects ----------------------------------------------------------------------------


def test_zero_byte_file() -> None:
    with pytest.raises(EmptyImageError):
        load_image(b"")


@pytest.mark.parametrize(
    "payload",
    [b"MZ\x90\x00 this is an exe", b"%PDF-1.7 ...", b"<svg xmlns='...'/>", b"just some text"],
)
def test_non_image_bytes_are_rejected_regardless_of_claimed_extension(payload: bytes) -> None:
    # The loader takes bytes only, so a ".jpg" name cannot influence the result.
    with pytest.raises(UnsupportedImageFormatError):
        load_image(payload)


def test_valid_header_but_garbage_body_is_corrupt() -> None:
    with pytest.raises(CorruptImageError):
        load_image(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)


def test_truncated_jpeg_is_corrupt() -> None:
    data = encode(
        Image.fromarray(np.random.default_rng(0).integers(0, 255, (64, 64, 3), np.uint8)), "JPEG"
    )
    with pytest.raises(CorruptImageError):
        load_image(data[: len(data) // 2])


def test_decompression_bomb_is_rejected_before_decoding() -> None:
    bomb = encode(Image.new("1", (20000, 20000)), "PNG")  # 400 MP, tiny on disk
    assert len(bomb) < 2_000_000
    with pytest.raises(ImageTooLargeError):
        load_image(bomb)


def test_sniffing_uses_magic_bytes() -> None:
    assert sniff_format(encode(corner_marked(), "PNG")) == "PNG"
    assert sniff_format(encode(corner_marked(), "JPEG")) == "JPEG"
    assert sniff_format(encode(corner_marked(), "BMP")) == "BMP"
    assert sniff_format(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "WEBP"
    assert sniff_format(b"GIF89a") == "GIF"
    assert sniff_format(b"") is None


# --- channel order -------------------------------------------------------------------------------


def test_output_is_rgb_not_bgr() -> None:
    red = Image.new("RGB", (8, 8), (255, 0, 0))
    loaded = load_image(encode(red, "PNG"))
    assert tuple(loaded.array[0, 0]) == (255, 0, 0)  # channel 0 is red
    blue = load_image(encode(Image.new("RGB", (8, 8), (0, 0, 255)), "PNG"))
    assert tuple(blue.array[0, 0]) == (0, 0, 255)


# --- modes ---------------------------------------------------------------------------------------


def test_grayscale_is_expanded_to_three_channels() -> None:
    gray = Image.fromarray(np.full((10, 12), 77, dtype=np.uint8), mode="L")
    loaded = load_image(encode(gray, "PNG"))
    assert loaded.array.shape == (10, 12, 3)
    assert (loaded.array == 77).all()


def test_rgba_is_composited_on_white() -> None:
    rgba = Image.new("RGBA", (6, 6), (255, 0, 0, 0))  # fully transparent red
    loaded = load_image(encode(rgba, "PNG"))
    assert (loaded.array == 255).all()
    opaque = load_image(encode(Image.new("RGBA", (6, 6), (10, 20, 30, 255)), "PNG"))
    assert tuple(opaque.array[0, 0]) == (10, 20, 30)


def test_palette_image() -> None:
    pal = Image.new("P", (8, 8))
    pal.putpalette([255, 0, 0] + [0, 0, 0] * 255)
    loaded = load_image(encode(pal, "PNG"))
    assert loaded.array.shape == (8, 8, 3)
    assert tuple(loaded.array[0, 0]) == (255, 0, 0)


def test_palette_with_transparency_is_composited() -> None:
    pal = Image.new("P", (4, 4), 0)
    pal.putpalette([255, 0, 0] + [0, 0, 0] * 255)
    loaded = load_image(encode(pal, "PNG", transparency=0))
    assert (loaded.array == 255).all()


def test_cmyk_jpeg_is_converted() -> None:
    cmyk = Image.new("CMYK", (8, 8), (0, 0, 0, 0))  # no ink = white
    loaded = load_image(encode(cmyk, "JPEG"))
    assert loaded.array.shape == (8, 8, 3)
    assert loaded.array.min() >= 250


def test_sixteen_bit_grayscale_is_scaled_to_eight_bit() -> None:
    arr = np.full((8, 8), 65535, dtype="<u2")
    arr[0, 0] = 0
    loaded = load_image(encode(Image.fromarray(arr), "PNG"))
    assert loaded.array[0, 0, 0] == 0
    assert loaded.array[4, 4, 0] == 255
    half = np.full((4, 4), 32768, dtype="<u2")
    mid = load_image(encode(Image.fromarray(half), "PNG"))
    assert mid.array[0, 0, 0] == 128


# --- orientation ---------------------------------------------------------------------------------


@pytest.mark.parametrize("orientation", range(1, 9))
def test_exif_orientation_matches_pillow_reference(orientation: int) -> None:
    from PIL import ImageOps

    base = corner_marked(40, 20)
    exif = Image.Exif()
    exif[0x0112] = orientation
    data = encode(base, "PNG", exif=exif.tobytes())
    reference = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
    assert reference is not None
    loaded = load_image(data)
    np.testing.assert_array_equal(loaded.array, np.asarray(reference.convert("RGB")))
    swaps = orientation in (5, 6, 7, 8)
    assert loaded.original_size == ((20, 40) if swaps else (40, 20))


# --- animated ------------------------------------------------------------------------------------


def _animated_gif() -> bytes:
    frames = [Image.new("RGB", (8, 8), c) for c in ((255, 0, 0), (0, 0, 255))]
    return encode(frames[0], "GIF", save_all=True, append_images=frames[1:], duration=50, loop=0)


def test_animated_gif_uses_first_frame_by_default() -> None:
    loaded = load_image(_animated_gif())
    assert tuple(loaded.array[0, 0]) == (255, 0, 0)
    assert any("first frame" in w for w in loaded.warnings)


def test_animated_image_can_be_rejected() -> None:
    with pytest.raises(AnimatedImageError):
        load_image(_animated_gif(), animated="reject")


def test_animated_webp_uses_first_frame() -> None:
    frames = [Image.new("RGB", (8, 8), c) for c in ((255, 0, 0), (0, 255, 0))]
    data = encode(frames[0], "WEBP", save_all=True, append_images=frames[1:], lossless=True)
    loaded = load_image(data)
    assert tuple(loaded.array[0, 0]) == (255, 0, 0)
    assert loaded.warnings


# --- size handling -------------------------------------------------------------------------------


def test_huge_image_is_downscaled_with_coordinate_mapping() -> None:
    big = Image.new("RGB", (12000, 900), (200, 100, 50))
    loaded = load_image(encode(big, "JPEG", quality=50))
    height, width = loaded.array.shape[:2]
    assert max(height, width) <= 4096
    assert loaded.original_size == (12000, 900)
    sx, _ = loaded.scale
    assert sx == pytest.approx(width / 12000)
    # A point at the original bottom-right maps back from the array's bottom-right.
    back = loaded.to_original(np.array([[width, height]], dtype=np.float32))
    np.testing.assert_allclose(back, [[12000, 900]], atol=1e-3)


def test_small_image_is_not_resized() -> None:
    loaded = load_image(encode(corner_marked(), "PNG"))
    assert loaded.array.shape == (20, 40, 3)
    assert loaded.scale == (1.0, 1.0)


def test_max_side_is_configurable() -> None:
    loaded = load_image(encode(corner_marked(400, 200), "PNG"), max_side=100)
    assert max(loaded.array.shape[:2]) <= 100
    assert loaded.original_size == (400, 200)


@pytest.mark.parametrize("size", [(4000, 8), (8, 4000), (1, 1)])
def test_extreme_aspect_ratios_decode(size: tuple[int, int]) -> None:
    loaded = load_image(encode(Image.new("RGB", size, (1, 2, 3)), "PNG"))
    assert loaded.array.shape[:2] == (size[1], size[0])

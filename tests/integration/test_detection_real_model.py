"""Detection edge cases on the real SCRFD model and a public-domain portrait.

Skipped automatically when the model weights are not downloaded. Several tests record how the
model *actually* behaves (for example that it does not find a face in a 64x64 image); those are
observations on one image, not guarantees, and are described that way in docs/LIMITATIONS.md.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from frp.detect import (
    Face,
    FacePolicy,
    ScrfdDetector,
    detect_in_original_space,
    select_faces,
)
from frp.errors import MultipleFacesError
from frp.imageio import load_image
from frp.models import MODELS

pytestmark = pytest.mark.needs_models


@pytest.fixture(scope="module")
def detector(models_dir: Path) -> ScrfdDetector:
    spec = MODELS["scrfd_10g"]
    return ScrfdDetector.from_file(models_dir / spec.filename, expected_sha256=spec.sha256)


@pytest.fixture(scope="module")
def portrait(astronaut_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(astronaut_bytes)).convert("RGB")


def encode(img: Image.Image, fmt: str, **kwargs: object) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


def faces_in(detector: ScrfdDetector, data: bytes) -> list[Face]:
    return detect_in_original_space(detector, load_image(data))


def iou(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(x2 - x1, 0) * max(y2 - y1, 0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union)


@pytest.fixture(scope="module")
def reference(detector: ScrfdDetector, portrait: Image.Image) -> Face:
    (face,) = faces_in(detector, encode(portrait, "PNG"))
    return face


# --- the basics ----------------------------------------------------------------------------------


def test_finds_one_face_with_plausible_landmarks(reference: Face) -> None:
    assert reference.score > 0.5
    left_eye, right_eye, nose, left_mouth, right_mouth = reference.landmarks
    assert left_eye[0] < right_eye[0]  # subject's left eye appears on the image's left
    assert left_mouth[0] < right_mouth[0]
    assert max(left_eye[1], right_eye[1]) < nose[1] < min(left_mouth[1], right_mouth[1])
    x1, y1, x2, y2 = reference.box
    assert (reference.landmarks[:, 0] > x1).all() and (reference.landmarks[:, 0] < x2).all()
    assert (reference.landmarks[:, 1] > y1).all() and (reference.landmarks[:, 1] < y2).all()
    assert not reference.truncated


def test_inference_is_deterministic(detector: ScrfdDetector, portrait: Image.Image) -> None:
    arr = np.asarray(portrait)
    first, second = detector.detect(arr), detector.detect(arr)
    np.testing.assert_array_equal(first[0].box, second[0].box)
    np.testing.assert_array_equal(first[0].landmarks, second[0].landmarks)


def test_no_face_in_blank_and_noise_images(detector: ScrfdDetector) -> None:
    blank = encode(Image.new("RGB", (400, 400), (120, 120, 120)), "PNG")
    noise = np.random.default_rng(0).integers(0, 255, (400, 400, 3), dtype=np.uint8)
    assert faces_in(detector, blank) == []
    assert faces_in(detector, encode(Image.fromarray(noise), "PNG")) == []


def test_tiny_images_yield_no_face(detector: ScrfdDetector, portrait: Image.Image) -> None:
    for side in (32, 64):
        assert faces_in(detector, encode(portrait.resize((side, side)), "PNG")) == []


# --- pixel formats and orientation give the same face --------------------------------------------


def _variants(portrait: Image.Image) -> dict[str, bytes]:
    gray16 = np.asarray(portrait.convert("L")).astype("<u2") * 257
    return {
        "grayscale": encode(portrait.convert("L"), "PNG"),
        "rgba": encode(portrait.convert("RGBA"), "PNG"),
        "palette": encode(portrait.convert("P", palette=Image.Palette.ADAPTIVE), "PNG"),
        "cmyk_jpeg": encode(portrait.convert("CMYK"), "JPEG"),
        "gray16": encode(Image.fromarray(gray16), "PNG"),
        "bmp": encode(portrait, "BMP"),
        "tiff": encode(portrait, "TIFF"),
    }


def test_all_pixel_formats_find_the_same_face(
    detector: ScrfdDetector, portrait: Image.Image, reference: Face
) -> None:
    for name, data in _variants(portrait).items():
        (face,) = faces_in(detector, data)
        assert iou(face.box, reference.box) > 0.85, name


@pytest.mark.parametrize("orientation", [3, 6, 8])
def test_exif_rotated_phone_photo_gives_upright_coordinates(
    detector: ScrfdDetector, portrait: Image.Image, reference: Face, orientation: int
) -> None:
    # Store the pixels rotated the way a camera would, tagged with the inverse EXIF orientation.
    stored_rotation = {3: 180, 6: 90, 8: 270}[orientation]
    exif = Image.Exif()
    exif[0x0112] = orientation
    data = encode(
        portrait.rotate(stored_rotation, expand=True), "JPEG", quality=95, exif=exif.tobytes()
    )
    (face,) = faces_in(detector, data)
    assert iou(face.box, reference.box) > 0.85


def test_horizontal_flip_mirrors_the_landmark_order_consistently(
    detector: ScrfdDetector, portrait: Image.Image
) -> None:
    flipped = portrait.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    (face,) = faces_in(detector, encode(flipped, "PNG"))
    # A mirrored face is still a normal face: the model keeps "left eye" on the image's left.
    assert face.landmarks[0, 0] < face.landmarks[1, 0]


# --- coordinates stay in original image space ----------------------------------------------------


def test_upscaled_image_scales_the_box(
    detector: ScrfdDetector, portrait: Image.Image, reference: Face
) -> None:
    big = portrait.resize((1024, 1024), Image.Resampling.LANCZOS)
    (face,) = faces_in(detector, encode(big, "PNG"))
    assert iou(face.box, reference.box * 2) > 0.85


def test_downscaled_by_loader_still_reports_original_coordinates(
    detector: ScrfdDetector, portrait: Image.Image, reference: Face
) -> None:
    data = encode(portrait, "PNG")
    loaded = load_image(data, max_side=256)
    assert loaded.array.shape[:2] == (256, 256)  # the detector really saw a smaller image
    (face,) = detect_in_original_space(detector, loaded)
    assert iou(face.box, reference.box) > 0.8


@pytest.mark.slow
def test_huge_image_is_downscaled_and_face_located_in_original_space(
    detector: ScrfdDetector, portrait: Image.Image
) -> None:
    canvas = Image.new("RGB", (12000, 6000), (40, 40, 40))
    scale, offset = 6.0, (3000, 1500)  # paste the portrait at 6x: 3072 x 3072 pixels
    canvas.paste(portrait.resize((3072, 3072), Image.Resampling.LANCZOS), offset)
    loaded = load_image(encode(canvas, "JPEG", quality=70))
    assert loaded.original_size == (12000, 6000)
    assert max(loaded.array.shape[:2]) <= 4096
    (face,) = detect_in_original_space(detector, loaded)
    expected = np.array([181, 58, 271, 179], dtype=np.float32) * scale
    expected += np.array([offset[0], offset[1], offset[0], offset[1]], dtype=np.float32)
    assert iou(face.box, expected) > 0.8


# --- framing -------------------------------------------------------------------------------------


def test_face_cut_off_by_the_frame_is_flagged_truncated(
    detector: ScrfdDetector, portrait: Image.Image
) -> None:
    for cut in (225, 235):  # slices through the face at x of about 181..271
        (face,) = faces_in(detector, encode(portrait.crop((0, 0, cut, 512)), "PNG"))
        assert face.truncated, cut


def test_face_cut_off_at_the_top_has_landmarks_outside_the_frame(
    detector: ScrfdDetector, portrait: Image.Image
) -> None:
    (face,) = faces_in(detector, encode(portrait.crop((0, 100, 512, 512)), "PNG"))
    assert face.truncated
    assert not face.landmarks_inside


def test_extreme_aspect_ratio_does_not_crash(
    detector: ScrfdDetector, portrait: Image.Image
) -> None:
    strip = Image.new("RGB", (4000, 160), (30, 30, 30))
    strip.paste(portrait.resize((160, 160)), (2000, 0))
    # Observation, not a guarantee: the face is shrunk to ~25 px by the letterbox and is missed.
    assert isinstance(faces_in(detector, encode(strip, "PNG")), list)


# --- multiple faces and policy -------------------------------------------------------------------


@pytest.fixture(scope="module")
def two_faces(portrait: Image.Image) -> bytes:
    canvas = Image.new("RGB", (1000, 500), (60, 60, 60))
    canvas.paste(portrait.resize((400, 400)), (20, 50))  # big, on the left
    canvas.paste(portrait.resize((200, 200)), (600, 150))  # small, on the right
    return encode(canvas, "PNG")


def test_two_faces_are_both_found(detector: ScrfdDetector, two_faces: bytes) -> None:
    assert len(faces_in(detector, two_faces)) == 2


def test_largest_policy_picks_the_larger_face_even_with_lower_score(
    detector: ScrfdDetector, two_faces: bytes
) -> None:
    found = faces_in(detector, two_faces)
    (chosen,) = select_faces(found, FacePolicy.LARGEST)
    assert chosen.box[0] < 400  # the big face on the left
    assert chosen.area == max(f.area for f in found)


def test_reject_policy_raises_on_two_faces(detector: ScrfdDetector, two_faces: bytes) -> None:
    with pytest.raises(MultipleFacesError):
        select_faces(faces_in(detector, two_faces), FacePolicy.REJECT)


def test_all_policy_returns_both_largest_first(detector: ScrfdDetector, two_faces: bytes) -> None:
    both = select_faces(faces_in(detector, two_faces), FacePolicy.ALL)
    assert len(both) == 2
    assert both[0].area > both[1].area


# --- rotation (edge-case matrix: head tilt 90/180 degrees) --------------------------------------


@pytest.mark.parametrize("degrees", [90, 180, 270])
def test_rotated_faces_are_detected_on_this_image(
    detector: ScrfdDetector, portrait: Image.Image, degrees: int
) -> None:
    # Observation on one image: SCRFD found the rotated face every time. Not a guarantee, and the
    # downstream embedding of such a face is a separate question (measured in M6 robustness).
    assert len(faces_in(detector, encode(portrait.rotate(degrees, expand=True), "PNG"))) == 1

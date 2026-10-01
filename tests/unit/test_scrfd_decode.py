"""SCRFD pre/post-processing, tested with a fake ONNX session (no model weights needed)."""

from __future__ import annotations

import numpy as np
import pytest

from frp.detect import Face, ScrfdDetector
from frp.errors import ModelFileError
from tests.unit.fake_scrfd import FakeScrfdSession, PlacedFace, default_landmarks

CANVAS = 640


def placed(box: tuple[float, float, float, float], score: float = 0.9, **kw: int) -> PlacedFace:
    return PlacedFace(box=box, landmarks=default_landmarks(box), score=score, **kw)


def blank(width: int, height: int) -> np.ndarray:
    return np.full((height, width, 3), 100, dtype=np.uint8)


def make(faces: list[PlacedFace], **params: float) -> tuple[ScrfdDetector, FakeScrfdSession]:
    session = FakeScrfdSession(faces)
    return ScrfdDetector(session, **params), session  # type: ignore[arg-type]


# --- coordinates ---------------------------------------------------------------------------------


@pytest.mark.parametrize("size", [(1000, 500), (500, 1000), (640, 640), (200, 100)])
def test_detections_are_mapped_back_to_original_image_space(size: tuple[int, int]) -> None:
    width, height = size
    scale = CANVAS / max(width, height)  # letterbox scale, computed independently of the code
    # A face occupying a known region of the original image.
    box_orig = (0.4 * width, 0.3 * height, 0.6 * width, 0.6 * height)
    box_canvas = tuple(v * scale for v in box_orig)
    detector, _ = make([placed(box_canvas)])  # type: ignore[arg-type]
    (face,) = detector.detect(blank(width, height))
    np.testing.assert_allclose(face.box, box_orig, atol=0.05)
    np.testing.assert_allclose(face.landmarks, default_landmarks(box_orig), atol=0.05)
    assert face.score == pytest.approx(0.9, abs=1e-6)


def test_blob_is_rgb_normalized_and_nchw() -> None:
    red = np.zeros((64, 64, 3), dtype=np.uint8)
    red[..., 0] = 255
    detector, session = make([])
    detector.detect(red)
    blob = session.last_blob
    assert blob is not None
    assert blob.shape == (1, 3, CANVAS, CANVAS)
    assert blob.dtype == np.float32
    # Inside the pasted image (top-left 640x640 here since the image is square):
    assert blob[0, 0, 10, 10] == pytest.approx((255 - 127.5) / 128)  # channel 0 is RED
    assert blob[0, 1, 10, 10] == pytest.approx((0 - 127.5) / 128)
    assert blob[0, 2, 10, 10] == pytest.approx((0 - 127.5) / 128)


def test_letterbox_pads_bottom_right_not_center() -> None:
    img = np.full((100, 200, 3), 255, dtype=np.uint8)  # wide image -> padded at the bottom
    detector, session = make([])
    detector.detect(img)
    assert session.last_blob is not None
    assert session.last_blob[0, 0, 0, 0] == pytest.approx((255 - 127.5) / 128)  # top-left is image
    assert session.last_blob[0, 0, CANVAS - 1, CANVAS - 1] == pytest.approx(-127.5 / 128)  # padding


# --- filtering -----------------------------------------------------------------------------------


def test_no_face_returns_empty_list() -> None:
    detector, _ = make([])
    assert detector.detect(blank(300, 300)) == []


def test_min_confidence_filters_low_scores() -> None:
    faces = [placed((100, 100, 200, 220), 0.9), placed((300, 300, 400, 420), 0.4)]
    detector, _ = make(faces, min_confidence=0.5)
    assert len(detector.detect(blank(640, 640))) == 1
    detector_lo, _ = make(faces, min_confidence=0.3)
    assert len(detector_lo.detect(blank(640, 640))) == 2


def test_tiny_faces_are_dropped() -> None:
    faces = [placed((100, 100, 112, 114), 0.95), placed((300, 300, 400, 420), 0.9)]
    detector, _ = make(faces, min_face_px=20)
    found = detector.detect(blank(640, 640))
    assert len(found) == 1
    assert found[0].size_px >= 20


def test_max_faces_keeps_highest_scores_first() -> None:
    faces = [placed((10 + 110 * i, 100, 100 + 110 * i, 220), 0.6 + 0.05 * i) for i in range(5)]
    detector, _ = make(faces, max_faces=3)
    found = detector.detect(blank(640, 640))
    assert [round(f.score, 2) for f in found] == [0.8, 0.75, 0.7]


def test_nms_suppresses_duplicate_anchors_for_one_face() -> None:
    box = (200.0, 200.0, 320.0, 340.0)
    dupes = [placed(box, 0.9, anchor=0), placed(box, 0.8, anchor=1)]
    detector, _ = make(dupes)
    found = detector.detect(blank(640, 640))
    assert len(found) == 1
    assert found[0].score == pytest.approx(0.9, abs=1e-6)


def test_multiple_scales_are_all_decoded() -> None:
    faces = [
        placed((50, 50, 120, 130), 0.9, stride=8),
        placed((300, 100, 500, 330), 0.85, stride=32),
    ]
    detector, _ = make(faces)
    assert len(detector.detect(blank(640, 640))) == 2


# --- partially out of frame ----------------------------------------------------------------------


def test_partially_out_of_frame_face_is_flagged_and_box_clipped() -> None:
    detector, _ = make([placed((-40, 100, 80, 240), 0.9)])
    (face,) = detector.detect(blank(640, 640))
    assert face.box[0] == 0.0  # clipped to the image
    assert face.visible_fraction == pytest.approx(80 / 120, abs=1e-3)
    assert face.truncated
    assert not face.landmarks_inside or face.visible_fraction < 0.9


def test_fully_inside_face_is_not_truncated() -> None:
    detector, _ = make([placed((200, 200, 320, 340), 0.9)])
    (face,) = detector.detect(blank(640, 640))
    assert not face.truncated
    assert face.visible_fraction == 1.0


def test_face_flush_with_the_border_is_flagged_even_when_fully_visible() -> None:
    # A frame that slices through a face makes the model predict a box ending at the edge, so
    # visible_fraction stays 1.0; the border flag is what catches it.
    detector, _ = make([placed((520, 200, 640, 340), 0.9)])
    (face,) = detector.detect(blank(640, 640))
    assert face.visible_fraction == pytest.approx(1.0, abs=0.01)
    assert face.touches_border
    assert face.truncated


def test_face_entirely_outside_the_image_is_discarded() -> None:
    # 100x50 image: canvas rows below the padded image area carry no real pixels.
    detector, _ = make([placed((100, 500, 220, 620), 0.95)])
    assert detector.detect(blank(100, 50)) == []


# --- extreme shapes and bad input ----------------------------------------------------------------


@pytest.mark.parametrize("size", [(4000, 8), (8, 4000), (1, 1), (3, 5000)])
def test_extreme_aspect_ratios_do_not_crash(size: tuple[int, int]) -> None:
    detector, session = make([])
    assert detector.detect(blank(*size)) == []
    assert session.calls == 1


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((10, 10), dtype=np.uint8),  # no channel axis
        np.zeros((10, 10, 4), dtype=np.uint8),  # RGBA
        np.zeros((10, 10, 3), dtype=np.float32),  # wrong dtype
    ],
)
def test_invalid_arrays_are_rejected(bad: np.ndarray) -> None:
    detector, _ = make([])
    with pytest.raises(ValueError, match="uint8 RGB"):
        detector.detect(bad)


def test_wrong_model_is_reported_clearly() -> None:
    class EightOutputs(FakeScrfdSession):
        def run(self, output_names, input_feed):  # type: ignore[no-untyped-def]
            return super().run(output_names, input_feed)[:8]

    detector = ScrfdDetector(EightOutputs())
    with pytest.raises(ModelFileError, match="9 outputs"):
        detector.detect(blank(64, 64))


@pytest.mark.parametrize(
    "params",
    [{"min_confidence": 1.5}, {"min_face_px": 0}, {"max_faces": 0}, {"input_size": 650}],
)
def test_invalid_parameters_are_rejected(params: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        ScrfdDetector(FakeScrfdSession(), **params)  # type: ignore[arg-type]


def test_face_scaled_is_consistent() -> None:
    face = Face(
        box=np.array([10, 20, 30, 60], dtype=np.float32),
        landmarks=default_landmarks((10, 20, 30, 60)),
        score=0.5,
    )
    doubled = face.scaled(2.0, 3.0)
    np.testing.assert_allclose(doubled.box, [20, 60, 60, 180])
    np.testing.assert_allclose(doubled.landmarks, face.landmarks * [2.0, 3.0], rtol=1e-6)

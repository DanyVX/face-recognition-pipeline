from __future__ import annotations

import numpy as np
import pytest

from frp.detect import Face, FacePolicy, detect_in_original_space, select_faces
from frp.errors import MultipleFacesError, NoFaceError
from frp.imageio import LoadedImage
from tests.unit.fake_scrfd import default_landmarks


def face(box: tuple[float, float, float, float], score: float = 0.9) -> Face:
    return Face(box=np.array(box, dtype=np.float32), landmarks=default_landmarks(box), score=score)


SMALL = face((10, 10, 50, 50), score=0.99)  # highest score, smaller
LARGE = face((100, 100, 300, 320), score=0.80)  # lower score, larger


def test_no_faces_raises_no_face() -> None:
    for policy in FacePolicy:
        with pytest.raises(NoFaceError):
            select_faces([], policy)


def test_largest_policy_prefers_area_over_score() -> None:
    (chosen,) = select_faces([SMALL, LARGE], FacePolicy.LARGEST)
    assert chosen is LARGE


def test_largest_is_the_default() -> None:
    assert select_faces([SMALL, LARGE]) == [LARGE]


def test_all_policy_returns_every_face_largest_first() -> None:
    assert select_faces([SMALL, LARGE], FacePolicy.ALL) == [LARGE, SMALL]


def test_reject_policy_raises_for_multiple_faces() -> None:
    with pytest.raises(MultipleFacesError):
        select_faces([SMALL, LARGE], FacePolicy.REJECT)


def test_reject_policy_accepts_a_single_face() -> None:
    assert select_faces([LARGE], FacePolicy.REJECT) == [LARGE]


def test_detect_in_original_space_undoes_downscaling() -> None:
    class Fixed:
        def detect(self, image: np.ndarray) -> list[Face]:
            return [face((10, 20, 50, 60))]

    loaded = LoadedImage(
        array=np.zeros((50, 100, 3), dtype=np.uint8), original_size=(400, 200), format="PNG"
    )
    (found,) = detect_in_original_space(Fixed(), loaded)
    np.testing.assert_allclose(found.box, [40, 80, 200, 240])

"""RecognitionService behavior with fakes (no model weights)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from frp.errors import (
    IdentityExistsError,
    IdentityNotFoundError,
    InvalidImageError,
    MultipleFacesError,
    NoFaceError,
    UnsupportedImageFormatError,
)
from frp.gallery import OnExisting
from frp.quality import QualityThresholds
from tests.unit.service_fixtures import (
    FACE_BOX,
    NoFaceDetector,
    SceneDetector,
    make_service,
    person_png,
)


def test_enroll_verify_identify_delete_round_trip(tmp_path: Path) -> None:
    svc = make_service(tmp_path)
    out = svc.enroll([person_png(1), person_png(1, 4, 1)], name="Alice", identity_id="alice")
    assert out.result.created and len(out.result.embedding_ids) == 2 and len(out.faces) == 2
    svc.enroll([person_png(2)], name="Bob", identity_id="bob")

    same = svc.verify_identity(person_png(1, 4, 2), "alice")
    other = svc.verify_identity(person_png(2, 4, 3), "alice")
    assert same.result.score > other.result.score
    found = svc.identify(person_png(2, 4, 4), k=2)
    assert found.result.candidates[0].identity_id == "bob"
    assert svc.delete("alice") == 2
    with pytest.raises(IdentityNotFoundError):
        svc.verify_identity(person_png(1), "alice")


def test_face_boxes_are_reported_in_original_image_coordinates(tmp_path: Path) -> None:
    out = make_service(tmp_path).enroll([person_png(1)], name="A")
    x1, y1, x2, y2 = out.faces[0].box
    assert (x1, y1, x2, y2) == pytest.approx(FACE_BOX)
    assert out.faces[0].n_faces == 1


def test_enrolling_with_no_detectable_face_is_an_error_naming_the_image(tmp_path: Path) -> None:
    svc = make_service(tmp_path, detector=NoFaceDetector())
    with pytest.raises(NoFaceError) as info:
        svc.enroll([person_png(1)], name="A")
    assert info.value.details == {"image_index": 0}
    assert svc.gallery.stats()["identities"] == 0


def test_enrollment_is_all_or_nothing_and_reports_which_photo_failed(tmp_path: Path) -> None:
    svc = make_service(tmp_path)
    with pytest.raises(UnsupportedImageFormatError) as info:
        svc.enroll([person_png(1), b"not an image"], name="A", identity_id="a")
    assert info.value.details == {"image_index": 1}
    assert svc.gallery.stats()["identities"] == 0  # the good first photo was not enrolled either


def test_enroll_requires_at_least_one_image(tmp_path: Path) -> None:
    with pytest.raises(InvalidImageError):
        make_service(tmp_path).enroll([], name="A")


def test_multiple_faces_are_rejected_at_enrollment_but_largest_is_used_for_queries(
    tmp_path: Path,
) -> None:
    svc = make_service(tmp_path, detector=SceneDetector(n_faces=2))
    with pytest.raises(MultipleFacesError):
        svc.enroll([person_png(1)], name="A")
    permissive = make_service(
        tmp_path / "p", detector=SceneDetector(n_faces=2), enroll_face_policy="largest"
    )
    permissive.enroll([person_png(1)], name="A", identity_id="a")
    outcome = permissive.identify(person_png(1))  # query policy defaults to "largest"
    assert outcome.face.n_faces == 2


def test_strict_query_policy_rejects_multiple_faces(tmp_path: Path) -> None:
    svc = make_service(
        tmp_path,
        detector=SceneDetector(n_faces=2),
        enroll_face_policy="largest",
        query_face_policy="reject",
    )
    svc.enroll([person_png(1)], name="A")
    with pytest.raises(MultipleFacesError):
        svc.identify(person_png(1))


def test_verify_images_same_person_scores_higher_than_different_people(tmp_path: Path) -> None:
    svc = make_service(tmp_path)
    same = svc.verify_images(person_png(1), person_png(1, 4, 1))
    different = svc.verify_images(person_png(1), person_png(2))
    assert same.score > different.score
    assert same.accepted and not different.accepted  # thresholds are 0.9 in the fixtures
    assert svc.verify_images(person_png(1), person_png(2), threshold=-1.0).accepted
    with pytest.raises(UnsupportedImageFormatError) as info:
        svc.verify_images(person_png(1), b"junk")
    assert info.value.details == {"image_index": 1}


def test_threshold_boundary_uses_the_configured_value(tmp_path: Path) -> None:
    svc = make_service(tmp_path)
    score = svc.verify_images(person_png(1), person_png(2), threshold=0.0).score
    assert svc.verify_images(person_png(1), person_png(2), threshold=score).accepted
    assert not svc.verify_images(person_png(1), person_png(2), threshold=score + 1e-4).accepted


def test_identity_collision_and_append(tmp_path: Path) -> None:
    svc = make_service(tmp_path)
    svc.enroll([person_png(1)], name="A", identity_id="a")
    with pytest.raises(IdentityExistsError):
        svc.enroll([person_png(1, 4, 1)], name="A", identity_id="a")
    appended = svc.enroll(
        [person_png(1, 4, 2)], name="A", identity_id="a", on_existing=OnExisting.APPEND
    )
    assert not appended.result.created and svc.gallery.get("a").n_embeddings == 2


def test_stage_timer_sees_every_stage(tmp_path: Path) -> None:
    seen: dict[str, int] = {}

    def timer(stage: str, seconds: float) -> None:
        assert seconds >= 0
        seen[stage] = seen.get(stage, 0) + 1

    svc = make_service(tmp_path, stage_timer=timer)
    svc.enroll([person_png(1)], name="A")
    svc.identify(person_png(1))
    assert set(seen) == {"decode", "detect_align", "embed", "match"}


def test_oversized_images_are_rejected_by_the_configured_limit(tmp_path: Path) -> None:
    svc = make_service(tmp_path, max_image_pixels=2000)
    with pytest.raises(InvalidImageError):
        svc.enroll([person_png(1)], name="A")  # 240 x 200 = 48 000 pixels > 2000


def test_quality_thresholds_are_loaded_from_a_file(tmp_path: Path) -> None:
    path = tmp_path / "q.json"
    path.write_text(
        json.dumps({"result": {"thresholds": QualityThresholds(min_face_px=500.0).to_dict()}}),
        encoding="utf-8",
    )
    svc = make_service(tmp_path, quality_thresholds_file=path)
    from frp.errors import LowQualityError

    with pytest.raises(LowQualityError) as info:
        svc.enroll([person_png(1)], name="A")  # the synthetic face box is 100 px
    assert "FACE_TOO_SMALL" in info.value.reasons


def test_no_embedding_content_is_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    import logging

    with caplog.at_level(logging.DEBUG):
        svc = make_service(tmp_path)
        out = svc.enroll([person_png(1)], name="A")
        svc.identify(person_png(1))
    text = caplog.text
    first_values = [f"{v:.4f}" for v in out.embeddings.vectors[0][:5]]
    assert not any(v in text for v in first_values)
    assert np.isfinite(out.embeddings.vectors).all()

"""Compose detection, face selection, alignment and quality gates into one reusable step."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from frp.align import AlignedFace, align_face
from frp.detect import Face, FaceDetector, FacePolicy, select_faces
from frp.errors import LowQualityError
from frp.imageio import LoadedImage
from frp.quality import Gate, GateContext, GateResult


@dataclass(frozen=True, eq=False)
class PreparedFace:
    """A selected, aligned face ready for embedding."""

    face: Face  # coordinates of the (possibly downscaled) loaded array
    aligned: AlignedFace
    n_faces: int  # how many faces the detector found in the image
    image_shape: tuple[int, int]  # (height, width) of the array the face was found in
    gate_results: tuple[GateResult, ...] = ()


class FacePipeline:
    """detect -> select (policy) -> align -> quality gates.

    Raises the specific ``FrpError`` subclasses (``NoFaceError``, ``MultipleFacesError``,
    ``InvalidLandmarksError``, ``LowQualityError``) rather than returning a best-effort result.
    """

    def __init__(
        self,
        detector: FaceDetector,
        *,
        policy: FacePolicy = FacePolicy.LARGEST,
        gates: Sequence[Gate] = (),
        allow_outside_landmarks: bool = False,
    ) -> None:
        self.detector = detector
        self.policy = policy
        self.gates = tuple(gates)
        self.allow_outside_landmarks = allow_outside_landmarks

    def prepare(self, loaded: LoadedImage) -> PreparedFace:
        image = loaded.array
        shape = (int(image.shape[0]), int(image.shape[1]))
        faces = self.detector.detect(image)
        (chosen,) = select_faces(faces, self.policy, image_shape=shape)
        aligned = align_face(image, chosen.landmarks, allow_outside=self.allow_outside_landmarks)
        ctx = GateContext(face=chosen, aligned=aligned, image_shape=shape)
        results = tuple(gate.check(ctx) for gate in self.gates)
        failed = [r for r in results if not r.passed]
        if failed:
            reasons = tuple(r.reason_code for r in failed if r.reason_code)
            raise LowQualityError(
                f"Face rejected by quality gate(s): {', '.join(reasons)}", reasons
            )
        return PreparedFace(
            face=chosen,
            aligned=aligned,
            n_faces=len(faces),
            image_shape=shape,
            gate_results=results,
        )

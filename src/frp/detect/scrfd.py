"""SCRFD face detector (InsightFace ``det_10g``) running on ONNX Runtime.

Pre/post-processing follows the public InsightFace reference implementation: letterbox into a
square canvas (top-left anchored), normalize ``(x - 127.5) / 128`` in RGB order, decode
anchor-free distance predictions at strides 8/16/32 (2 anchors per cell), then NMS.

License note: the pretrained weights are for non-commercial research use only; see MODEL_CARD.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from frp.detect.base import Face
from frp.errors import ModelFileError
from frp.onnx import create_session

_STRIDES = (8, 16, 32)
_NUM_ANCHORS = 2
_MEAN = 127.5
_STD = 128.0
_BORDER_PX = 1.0  # a box edge this close to the image edge counts as touching the border


class InferenceSessionLike(Protocol):
    """The subset of ``onnxruntime.InferenceSession`` used here (lets tests inject a fake)."""

    def run(self, output_names: Any, input_feed: dict[str, Any]) -> list[Any]: ...

    def get_inputs(self) -> list[Any]: ...


def _nms(boxes: NDArray[np.float32], scores: NDArray[np.float32], thresh: float) -> list[int]:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(xx2 - xx1, 0) * np.maximum(yy2 - yy1, 0)
        union = areas[i] + areas[order[1:]] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        order = order[1:][iou <= thresh]
    return keep


class ScrfdDetector:
    """``FaceDetector`` backend for SCRFD.

    Args:
        session: an ONNX session (or compatible object) for the detector model.
        min_confidence: discard detections scoring below this.
        min_face_px: discard faces whose shorter box side is below this many pixels (in the
            coordinate frame of the image passed to :meth:`detect`).
        max_faces: keep at most this many faces (highest scores).
        nms_threshold: IoU threshold for non-maximum suppression.
        input_size: side of the square detection canvas.
    """

    def __init__(
        self,
        session: InferenceSessionLike,
        *,
        min_confidence: float = 0.5,
        min_face_px: int = 20,
        max_faces: int = 10,
        nms_threshold: float = 0.4,
        input_size: int = 640,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if min_face_px < 1 or max_faces < 1 or input_size % 32 != 0:
            raise ValueError("min_face_px/max_faces must be >= 1 and input_size a multiple of 32")
        self._session = session
        self._input_name: str = session.get_inputs()[0].name
        self.min_confidence = min_confidence
        self.min_face_px = min_face_px
        self.max_faces = max_faces
        self.nms_threshold = nms_threshold
        self.input_size = input_size

    @classmethod
    def from_file(
        cls,
        model_path: Path,
        *,
        providers: list[str] | None = None,
        intra_op_threads: int | None = None,
        expected_sha256: str | None = None,
        **params: Any,
    ) -> ScrfdDetector:
        session = create_session(
            model_path,
            providers=providers,
            intra_op_threads=intra_op_threads,
            expected_sha256=expected_sha256,
        )
        return cls(session, **params)

    # -- pipeline ---------------------------------------------------------------------------

    def _letterbox(self, image: NDArray[np.uint8]) -> tuple[NDArray[np.uint8], float]:
        size = self.input_size
        height, width = image.shape[:2]
        if height / width > 1.0:
            new_h, new_w = size, max(1, int(size / (height / width)))
        else:
            new_w, new_h = size, max(1, int(size * (height / width)))
        new_h, new_w = max(1, new_h), max(1, new_w)
        shrinking = new_h < height or new_w < width
        interp = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
        resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized
        return canvas, new_h / height

    def _blob(self, canvas: NDArray[np.uint8]) -> NDArray[np.float32]:
        # The model expects RGB; `image` is already RGB so no channel swap is needed here.
        blob = (canvas.astype(np.float32) - _MEAN) / _STD
        return np.ascontiguousarray(blob.transpose(2, 0, 1)[None])

    def _decode(
        self, outputs: list[NDArray[np.float32]]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        if len(outputs) != 9:
            msg = (
                "Unexpected detector model: expected 9 outputs (SCRFD with keypoints), "
                f"got {len(outputs)}"
            )
            raise ModelFileError(msg)
        all_boxes: list[NDArray[np.float32]] = []
        all_kps: list[NDArray[np.float32]] = []
        all_scores: list[NDArray[np.float32]] = []
        for idx, stride in enumerate(_STRIDES):
            scores = outputs[idx].reshape(-1)
            bbox_preds = outputs[idx + 3].reshape(-1, 4) * stride
            kps_preds = outputs[idx + 6].reshape(-1, 10) * stride
            cells = self.input_size // stride
            ys, xs = np.mgrid[:cells, :cells]
            centers = np.stack([xs, ys], axis=-1).astype(np.float32).reshape(-1, 2) * stride
            centers = np.repeat(centers, _NUM_ANCHORS, axis=0)
            if centers.shape[0] != scores.shape[0]:
                msg = "Detector output shape does not match input_size; wrong model or size."
                raise ModelFileError(msg)
            keep = np.flatnonzero(scores >= self.min_confidence)
            if keep.size == 0:
                continue
            c = centers[keep]
            d = bbox_preds[keep]
            boxes = np.stack(
                [c[:, 0] - d[:, 0], c[:, 1] - d[:, 1], c[:, 0] + d[:, 2], c[:, 1] + d[:, 3]], 1
            )
            kps = kps_preds[keep].reshape(-1, 5, 2) + c[:, None, :]
            all_boxes.append(boxes.astype(np.float32))
            all_kps.append(kps.astype(np.float32))
            all_scores.append(scores[keep].astype(np.float32))
        if not all_boxes:
            empty = np.zeros((0,), dtype=np.float32)
            return np.zeros((0, 4), np.float32), np.zeros((0, 5, 2), np.float32), empty
        return np.concatenate(all_boxes), np.concatenate(all_kps), np.concatenate(all_scores)

    def detect(self, image: NDArray[np.uint8]) -> list[Face]:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be a uint8 RGB array of shape (H, W, 3)")
        height, width = image.shape[:2]
        canvas, det_scale = self._letterbox(image)
        outputs = self._session.run(None, {self._input_name: self._blob(canvas)})
        boxes, kps, scores = self._decode([np.asarray(o, dtype=np.float32) for o in outputs])
        if scores.size == 0:
            return []
        keep = _nms(boxes, scores, self.nms_threshold)
        boxes, kps, scores = boxes[keep] / det_scale, kps[keep] / det_scale, scores[keep]

        faces: list[Face] = []
        for box, lm, score in zip(boxes, kps, scores, strict=True):
            raw_area = max(float(box[2] - box[0]), 0.0) * max(float(box[3] - box[1]), 0.0)
            clipped = np.array(
                [max(box[0], 0), max(box[1], 0), min(box[2], width), min(box[3], height)],
                dtype=np.float32,
            )
            clipped_area = max(float(clipped[2] - clipped[0]), 0.0) * max(
                float(clipped[3] - clipped[1]), 0.0
            )
            if raw_area <= 0 or clipped_area <= 0:
                continue
            face = Face(
                box=clipped,
                landmarks=lm.astype(np.float32),
                score=float(score),
                visible_fraction=clipped_area / raw_area,
                touches_border=bool(
                    box[0] <= _BORDER_PX
                    or box[1] <= _BORDER_PX
                    or box[2] >= width - _BORDER_PX
                    or box[3] >= height - _BORDER_PX
                ),
                landmarks_inside=bool(
                    (lm[:, 0] >= 0).all()
                    and (lm[:, 0] <= width).all()
                    and (lm[:, 1] >= 0).all()
                    and (lm[:, 1] <= height).all()
                ),
            )
            if face.size_px >= self.min_face_px:
                faces.append(face)
        faces.sort(key=lambda f: f.score, reverse=True)
        return faces[: self.max_faces]

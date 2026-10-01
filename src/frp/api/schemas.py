"""Request/response schemas. They drive the OpenAPI documentation, examples included."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ErrorBody(BaseModel):
    code: str = Field(description="Stable machine-readable error code.")
    message: str = Field(description="Human-readable explanation. Not a stable contract.")
    request_id: str = Field(description="Correlates the response with server logs.")
    details: dict[str, Any] | None = Field(
        default=None, description="Optional context, e.g. which uploaded image failed."
    )


class ErrorResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "error": {
                        "code": "NO_FACE",
                        "message": "No face detected.",
                        "request_id": "3f2c9b1e7a5d4c60a1b2c3d4e5f60718",
                        "details": {"image_index": 0},
                    }
                }
            ]
        }
    )
    error: ErrorBody


class FaceOut(BaseModel):
    box: list[float] = Field(
        description="Face box [x1, y1, x2, y2] in pixels of the uploaded image (EXIF-upright, "
        "original resolution).",
        min_length=4,
        max_length=4,
    )
    score: float = Field(description="Detector confidence, 0-1.")
    n_faces: int = Field(description="How many faces the detector found in this image.")


class DuplicateOut(BaseModel):
    identity_id: str
    name: str
    score: float = Field(description="Cosine similarity to the closest stored embedding.")


class EnrollResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "identity_id": "a1b2c3d4e5f60718293a4b5c6d7e8f90",
                    "created": True,
                    "embeddings_added": 2,
                    "faces": [{"box": [40.0, 20.0, 200.0, 180.0], "score": 0.97, "n_faces": 1}],
                    "duplicates": [],
                }
            ]
        }
    )
    identity_id: str
    created: bool = Field(description="False when photos were appended to an existing identity.")
    embeddings_added: int
    faces: list[FaceOut]
    duplicates: list[DuplicateOut] = Field(
        description="Other identities this person already resembles (duplicate policy warn)."
    )


class VerifyResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "accepted": True,
                    "score": 0.71,
                    "threshold": 0.4,
                    "identity_id": None,
                    "faces": [{"box": [40.0, 20.0, 200.0, 180.0], "score": 0.97, "n_faces": 1}],
                }
            ]
        }
    )
    accepted: bool = Field(description="True when score >= threshold (inclusive).")
    score: float = Field(description="Cosine similarity.")
    threshold: float
    identity_id: str | None = Field(description="Set when the probe was compared with an identity.")
    faces: list[FaceOut]
    embedding_b64: list[str] | None = Field(
        default=None,
        description="Probe embedding(s) as base64 float32, only when the server allows it.",
    )


class CandidateOut(BaseModel):
    identity_id: str
    name: str
    score: float


class IdentifyResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "match": {
                        "identity_id": "a1b2c3d4e5f60718293a4b5c6d7e8f90",
                        "name": "Alice",
                        "score": 0.74,
                    },
                    "candidates": [
                        {
                            "identity_id": "a1b2c3d4e5f60718293a4b5c6d7e8f90",
                            "name": "Alice",
                            "score": 0.74,
                        }
                    ],
                    "threshold": 0.4,
                    "face": {"box": [40.0, 20.0, 200.0, 180.0], "score": 0.97, "n_faces": 1},
                }
            ]
        }
    )
    match: CandidateOut | None = Field(
        description="The best candidate if its score reaches the threshold; otherwise null "
        "(open set: the person may not be enrolled)."
    )
    candidates: list[CandidateOut]
    threshold: float
    face: FaceOut
    embedding_b64: str | None = None


class DeleteResponse(BaseModel):
    identity_id: str
    embeddings_removed: int


class HealthResponse(BaseModel):
    status: str = "ok"


class ReadyResponse(BaseModel):
    ready: bool
    reasons: list[str] = Field(default_factory=list)

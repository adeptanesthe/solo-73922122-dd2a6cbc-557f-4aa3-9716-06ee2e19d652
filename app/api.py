"""FastAPI application: register samples, then replace their background."""

from __future__ import annotations

import base64
import os
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from .compose import ComposeResult, compose
from .images import (
    MAX_UPLOAD_BYTES,
    ImageValidationError,
    colour_hex,
    decode_rgb_png,
    encode_png,
    parse_color,
)
from .registry import Sample, SampleRegistry

# Fixed preview colours for the human-readable mask PNG.
CONTRADICTION_RGB = (255, 64, 64)
AMBIGUOUS_RGB = (64, 128, 255)


class SampleMetadata(BaseModel):
    slot: int
    background: str
    colour: list[int]
    width: int
    height: int
    version: int
    filename: str
    updated_at: str
    pair_size_match: bool = Field(
        default=True,
        description=(
            "false when the other slot is registered with a different size; "
            "replacing the background will then fail until one is corrected"
        ),
    )


class ReplaceResponse(BaseModel):
    status: str
    reason: Optional[str]
    width: int
    height: int
    image_png_base64: Optional[str]
    mask_png_base64: Optional[str]
    mask_regions_png_base64: Optional[str] = None
    mask_regions: Optional[dict[str, str]] = None
    counts: dict[str, int]
    versions: list[Optional[int]]


class RegistryState(BaseModel):
    revision: int
    samples: dict[str, Optional[SampleMetadata]]
    ready: bool


class HealthState(BaseModel):
    status: str = "ok"


def _sample_metadata(sample: Sample, pair_size_match: bool = True) -> SampleMetadata:
    height, width, _ = sample.image.shape
    return SampleMetadata(
        slot=sample.slot,
        background=colour_hex(sample.background),
        colour=list(sample.background),
        width=width,
        height=height,
        version=sample.version,
        filename=sample.filename,
        updated_at=sample.updated_at,
        pair_size_match=pair_size_match,
    )


def _bad_request(code: str, message: str, status: int = 422, **extra) -> HTTPException:
    detail = {"code": code, "message": message}
    detail.update(extra)
    return HTTPException(status_code=status, detail=detail)


def _read_upload(upload: UploadFile, field: str) -> bytes:
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise _bad_request(
            "too_large",
            f"{field} is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB",
            status=413,
        )
    if not data:
        raise _bad_request("empty_file", f"{field} is empty")
    return data


def _decode(upload: UploadFile, field: str) -> tuple[np.ndarray, str]:
    data = _read_upload(upload, field)
    try:
        array = decode_rgb_png(data)
    except ImageValidationError as exc:
        raise _bad_request("invalid_image", f"{field}: {exc}") from exc
    filename = os.path.basename(upload.filename or "upload.png")[:128]
    return array, filename


def _parse_background(value: str, field: str) -> tuple[int, int, int]:
    try:
        return parse_color(value)
    except ValueError as exc:
        raise _bad_request("invalid_background", f"{field}: {exc}") from exc


def _parse_slot(slot: int) -> int:
    if slot not in (1, 2):
        raise _bad_request("invalid_slot", "slot must be 1 or 2")
    return slot


def _mask_png(result: ComposeResult) -> bytes:
    """Human-readable mask: transparent where unique, red/blue elsewhere."""
    height, width = result.unique.shape
    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    canvas[result.contradiction, :3] = CONTRADICTION_RGB
    canvas[result.ambiguous, :3] = AMBIGUOUS_RGB
    canvas[..., 3] = np.where(result.unique, 0, 255)
    return encode_png(canvas)


def _region_masks_png(result: ComposeResult) -> bytes:
    """Lossless diagnostic: R = contradiction, G = ambiguous, B = unique."""
    height, width = result.unique.shape
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[..., 0] = np.where(result.contradiction, 255, 0)
    canvas[..., 1] = np.where(result.ambiguous, 255, 0)
    canvas[..., 2] = np.where(result.unique, 255, 0)
    return encode_png(canvas)


def _counts(result: ComposeResult) -> dict:
    return {
        "unique": int(result.unique.sum()),
        "contradiction": int(result.contradiction.sum()),
        "ambiguous": int(result.ambiguous.sum()),
        "total": int(result.unique.size),
    }


def create_app(registry: Optional[SampleRegistry] = None) -> FastAPI:
    registry = registry or SampleRegistry()
    app = FastAPI(
        title="Backdrop replacement service",
        version="1.0.0",
        description=(
            "Register two samples of one frame over two solid colours, then "
            "composite the underlying frame over a new background. Pixels whose "
            "new RGB is not provably unique are reported as masks."
        ),
    )
    app.state.registry = registry

    @app.get("/healthz", response_model=HealthState)
    def healthz() -> HealthState:
        return HealthState()

    @app.get("/v1/samples", response_model=RegistryState)
    def list_samples() -> RegistryState:
        snapshot = registry.snapshot()
        first, second = snapshot.first, snapshot.second
        pair_match = (
            first is None
            or second is None
            or first.image.shape == second.image.shape
        )
        items: dict[str, Optional[SampleMetadata]] = {}
        for slot, sample in ((1, first), (2, second)):
            items[str(slot)] = (
                _sample_metadata(sample, pair_size_match=pair_match)
                if sample
                else None
            )
        return RegistryState(
            revision=registry.revision(),
            samples=items,
            ready=snapshot.first is not None and snapshot.second is not None,
        )

    @app.put("/v1/samples/{slot}", response_model=SampleMetadata)
    def put_sample(
        slot: int,
        background: str = Form(..., description="#RRGGBB, RRGGBB or r,g,b"),
        image: UploadFile = File(..., description="8-bit RGB PNG, max 1024 per side"),
    ) -> SampleMetadata:
        slot = _parse_slot(slot)
        background_rgb = _parse_background(background, "background")
        array, filename = _decode(image, "image")
        sample = registry.put(slot, array, background_rgb, filename)
        other = registry.get(3 - slot)
        matches = other is None or other.image.shape == sample.image.shape
        return _sample_metadata(sample, pair_size_match=matches)

    @app.delete("/v1/samples/{slot}")
    def delete_sample(slot: int) -> dict:
        slot = _parse_slot(slot)
        if not registry.delete(slot):
            raise _bad_request("not_registered", f"slot {slot} is empty", status=404)
        return {"deleted": slot}

    @app.post("/v1/replace", response_model=ReplaceResponse)
    def replace(
        background: UploadFile = File(..., description="new background PNG"),
        format: str = Form("json"),
    ):
        if format not in ("json", "png"):
            raise _bad_request("invalid_format", "format must be 'json' or 'png'")

        snapshot = registry.snapshot()
        if snapshot.first is None or snapshot.second is None:
            missing = [
                str(slot)
                for slot, sample in ((1, snapshot.first), (2, snapshot.second))
                if sample is None
            ]
            raise _bad_request(
                "samples_missing",
                "both samples must be registered before composing",
                status=409,
                missing_slots=missing,
            )
        first, second = snapshot.first, snapshot.second
        if first.image.shape != second.image.shape:
            raise _bad_request(
                "sample_size_mismatch",
                (
                    f"samples differ in size: slot 1 is {first.image.shape[1]}x"
                    f"{first.image.shape[0]}, slot 2 is {second.image.shape[1]}x"
                    f"{second.image.shape[0]}"
                ),
                status=409,
            )

        new_array, _ = _decode(background, "background")
        if new_array.shape != first.image.shape:
            raise _bad_request(
                "background_size_mismatch",
                (
                    f"new background is {new_array.shape[1]}x{new_array.shape[0]} "
                    f"but the samples are {first.image.shape[1]}x{first.image.shape[0]}"
                ),
                status=409,
            )

        result = compose(
            first.image,
            second.image,
            first.background,
            second.background,
            new_array,
        )

        # Samples may be corrected while the computation runs.  Never answer
        # with a stale result: tell the client to retry instead.
        if not registry.is_current(snapshot.version_token):
            raise _bad_request(
                "samples_changed",
                "a sample changed during composition; rerun with the current samples",
                status=409,
                versions_at_start=list(snapshot.version_token),
            )

        if format == "png":
            if result.all_unique:
                return Response(
                    content=encode_png(result.image),
                    media_type="image/png",
                    headers={
                        "X-Compose-Status": "ok",
                        "X-Unique-Pixels": str(int(result.unique.sum())),
                    },
                )
            headers = {
                "X-Compose-Status": "mask",
                "X-Mask-Kind": "preview",
                "X-Preview-Colours": (
                    f"contradiction=#{CONTRADICTION_RGB[0]:02x}{CONTRADICTION_RGB[1]:02x}"
                    f"{CONTRADICTION_RGB[2]:02x},ambiguous=#{AMBIGUOUS_RGB[0]:02x}"
                    f"{AMBIGUOUS_RGB[1]:02x}{AMBIGUOUS_RGB[2]:02x}"
                ),
                "X-Unique-Pixels": str(int(result.unique.sum())),
                "X-Contradiction-Pixels": str(int(result.contradiction.sum())),
                "X-Ambiguous-Pixels": str(int(result.ambiguous.sum())),
            }
            return Response(
                content=_mask_png(result), media_type="image/png", headers=headers
            )

        if result.all_unique:
            return JSONResponse(
                {
                    "status": "ok",
                    "reason": None,
                    "width": result.image.shape[1],
                    "height": result.image.shape[0],
                    "image_png_base64": base64.b64encode(
                        encode_png(result.image)
                    ).decode("ascii"),
                    "mask_png_base64": None,
                    "counts": _counts(result),
                    "versions": list(snapshot.version_token),
                }
            )

        return JSONResponse(
            {
                "status": "mask",
                "reason": "not_unique",
                "width": result.unique.shape[1],
                "height": result.unique.shape[0],
                "image_png_base64": None,
                "mask_png_base64": base64.b64encode(_mask_png(result)).decode("ascii"),
                "mask_regions_png_base64": base64.b64encode(
                    _region_masks_png(result)
                ).decode("ascii"),
                "mask_regions": {
                    "red_channel": "contradiction",
                    "green_channel": "ambiguous",
                    "blue_channel": "unique",
                },
                "counts": _counts(result),
                "versions": list(snapshot.version_token),
            }
        )

    return app


app = create_app()

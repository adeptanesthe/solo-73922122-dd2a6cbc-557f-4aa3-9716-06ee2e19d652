"""API-level tests through FastAPI's TestClient."""

import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api import create_app
from app.registry import SampleRegistry


def png_bytes(array, mode="RGB"):
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(array, dtype=np.uint8), mode=mode).save(
        buffer, format="PNG"
    )
    return buffer.getvalue()


def decode_b64(value):
    data = base64.b64decode(value)
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        return np.array(image)


@pytest.fixture()
def client():
    registry = SampleRegistry()
    app = create_app(registry)
    with TestClient(app) as test_client:
        test_client.registry = registry
        yield test_client


def register(client, slot, array, background):
    return client.put(
        f"/v1/samples/{slot}",
        data={"background": background},
        files={"image": ("sample.png", png_bytes(array), "image/png")},
    )


def scene():
    """A 6x5 frame with known alpha and background replacement result."""
    h, w = 5, 6
    # These alphas are pinned well enough by the two samples that the new
    # background is uniquely determined (verified exhaustively elsewhere).
    alpha = np.tile(np.array([22, 63, 129, 200, 255], dtype=np.int64)[:, None], (1, w))
    alpha = alpha[:, :, None]
    fg = np.zeros((h, w, 3), dtype=np.int64)
    fg[..., 0] = 200
    fg[..., 1] = 100
    fg[..., 2] = 50
    bg1 = np.array([20, 40, 60], dtype=np.int64)
    bg2 = np.array([200, 180, 160], dtype=np.int64)
    bnew = np.array([5, 250, 15], dtype=np.int64)
    s1 = ((alpha * fg + (255 - alpha) * bg1 + 127) // 255).astype(np.uint8)
    s2 = ((alpha * fg + (255 - alpha) * bg2 + 127) // 255).astype(np.uint8)
    expected = ((alpha * fg + (255 - alpha) * bnew + 127) // 255).astype(np.uint8)
    return s1, s2, tuple(int(v) for v in bg1), tuple(int(v) for v in bg2), (
        np.broadcast_to(bnew, (h, w, 3)).astype(np.uint8)
    ), expected


def test_health(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_register_and_list(client):
    s1, s2, bg1, bg2, _, _ = scene()
    response = register(client, 1, s1, "#141428")
    assert response.status_code == 200, response.text
    assert response.json()["background"] == "#141428"
    assert response.json()["width"] == 6 and response.json()["height"] == 5
    assert response.json()["slot"] == 1

    register(client, 2, s2, "#c8b4a0")
    listing = client.get("/v1/samples").json()
    assert listing["ready"] is True
    assert listing["samples"]["1"]["background"] == "#141428"
    assert listing["samples"]["2"]["background"] == "#c8b4a0"
    assert "image" not in listing["samples"]["1"]


def test_replacement_unique_returns_image(client):
    s1, s2, bg1, bg2, bnew, expected = scene()
    register(client, 1, s1, "%d,%d,%d" % bg1)
    register(client, 2, s2, "%d,%d,%d" % bg2)
    response = client.post(
        "/v1/replace",
        files={"background": ("new.png", png_bytes(bnew), "image/png")},
        data={"format": "json"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["reason"] is None
    assert payload["mask_png_base64"] is None
    assert payload["counts"]["total"] == 30
    assert payload["counts"]["unique"] == 30
    assert payload["counts"]["contradiction"] == 0
    assert payload["counts"]["ambiguous"] == 0
    image = decode_b64(payload["image_png_base64"])
    np.testing.assert_array_equal(image, expected)


def test_replacement_png_format(client):
    s1, s2, bg1, bg2, bnew, expected = scene()
    register(client, 1, s1, "%d,%d,%d" % bg1)
    register(client, 2, s2, "%d,%d,%d" % bg2)
    response = client.post(
        "/v1/replace",
        files={"background": ("new.png", png_bytes(bnew), "image/png")},
        data={"format": "png"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-compose-status"] == "ok"
    with Image.open(io.BytesIO(response.content)) as image:
        image.load()
        np.testing.assert_array_equal(np.array(image), expected)


def test_contradictory_samples_return_mask(client):
    s1, s2, bg1, bg2, _, _ = scene()
    s1 = s1.copy()
    s1[0, 0] = [1, 2, 254]  # arbitrary, breaks the shared frame
    register(client, 1, s1, "%d,%d,%d" % bg1)
    register(client, 2, s2, "%d,%d,%d" % bg2)
    bnew = np.full_like(s1, 99)
    # The rest of the frame is still uniquely determined, so the mask must mix
    # contradiction and unique pixels rather than failing wholesale.
    response = client.post(
        "/v1/replace",
        files={"background": ("new.png", png_bytes(bnew), "image/png")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "mask"
    assert payload["reason"] == "not_unique"
    assert payload["counts"]["contradiction"] == 1
    assert payload["image_png_base64"] is None
    mask = decode_b64(payload["mask_png_base64"])
    assert mask.shape == (5, 6, 4)
    assert mask[0, 0, 3] == 255 and tuple(mask[0, 0, :3]) == (255, 64, 64)
    # row 0 alpha 22 with the new background 99 does not pin the RGB exactly,
    # but alpha 255 in row 4 does.
    assert mask[4, 4, 3] == 0

    regions = decode_b64(payload["mask_regions_png_base64"])
    assert regions[0, 0, 0] == 255  # contradiction in red
    assert regions[1, 1, 0] == 0


def test_ambiguous_samples_are_marked(client):
    # Alpha 0 region: both samples equal their backgrounds over a wide area,
    # but the frame underneath is unknowable there.
    h, w = 4, 4
    s1 = np.zeros((h, w, 3), dtype=np.uint8)
    s1[..., 0] = 10
    s2 = np.zeros((h, w, 3), dtype=np.uint8)
    s2[..., 0] = 250
    # one genuinely reconstructed pixel
    s1[0, 0] = [200, 100, 50]
    s2[0, 0] = [100, 50, 25]
    register(client, 1, s1, "10,0,0")
    register(client, 2, s2, "250,0,0")
    bnew = np.full((h, w, 3), 77, dtype=np.uint8)
    payload = client.post(
        "/v1/replace",
        files={"background": ("new.png", png_bytes(bnew), "image/png")},
    ).json()
    # The rest of the image is either contradictory or ambiguous, so no image.
    assert payload["status"] == "mask"
    assert payload["counts"]["unique"] + payload["counts"]["ambiguous"] + payload[
        "counts"
    ]["contradiction"] == h * w


def test_replace_requires_both_samples(client):
    s1, _, bg1, _, _, _ = scene()
    register(client, 1, s1, "%d,%d,%d" % bg1)
    response = client.post(
        "/v1/replace",
        files={"background": ("new.png", png_bytes(s1), "image/png")},
    )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "samples_missing"
    assert response.json()["detail"]["missing_slots"] == ["2"]


def test_sample_correction_replaces_old_slot(client):
    s1, s2, bg1, bg2, bnew, _ = scene()
    register(client, 1, s1, "%d,%d,%d" % bg1)
    first = client.get("/v1/samples").json()
    register(client, 1, s2, "%d,%d,%d" % bg2)
    second = client.get("/v1/samples").json()
    assert second["revision"] == first["revision"] + 1
    assert second["samples"]["1"]["background"] == "#%02x%02x%02x" % bg2
    assert second["samples"]["1"]["version"] > first["samples"]["1"]["version"]


def test_stale_snapshot_is_rejected_during_computation():
    """A sample edited mid-computation makes the endpoint ask for a rerun."""
    registry = SampleRegistry()
    s1, s2, bg1, bg2, bnew, _ = scene()
    registry.put(1, s1, bg1, "a.png")
    registry.put(2, s2, bg2, "b.png")

    app = create_app(registry)
    from app import api as api_module

    original = api_module.compose

    def compile_then_mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        # Simulate a correction arriving while the request was computing.
        registry.put(1, s1, (1, 2, 3), "corrected.png")
        return result

    api_module.compose = compile_then_mutate
    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/replace",
                files={"background": ("new.png", png_bytes(bnew), "image/png")},
            )
    finally:
        api_module.compose = original
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "samples_changed"
    # The corrected sample is kept.
    assert registry.get(1).background == (1, 2, 3)


def test_invalid_inputs(client):
    s1, _, bg1, _, _, _ = scene()
    # non-PNG upload
    response = client.put(
        "/v1/samples/1",
        data={"background": "%d,%d,%d" % bg1},
        files={"image": ("x.png", b"not a png", "image/png")},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_image"

    # RGBA PNG is rejected
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    response = client.put(
        "/v1/samples/1",
        data={"background": "0,0,0"},
        files={"image": ("x.png", png_bytes(rgba, mode="RGBA"), "image/png")},
    )
    assert response.status_code == 422
    assert "8-bit truecolour RGB" in response.json()["detail"]["message"]

    # palette PNG is rejected
    palette = Image.fromarray(np.zeros((4, 4), dtype=np.uint8), mode="P")
    buffer = io.BytesIO()
    palette.save(buffer, format="PNG")
    response = client.put(
        "/v1/samples/1",
        data={"background": "0,0,0"},
        files={"image": ("x.png", buffer.getvalue(), "image/png")},
    )
    assert response.status_code == 422

    # oversize (1025 px) image is rejected before decoding pixels
    wide = np.zeros((1, 1025, 3), dtype=np.uint8)
    response = client.put(
        "/v1/samples/1",
        data={"background": "0,0,0"},
        files={"image": ("x.png", png_bytes(wide), "image/png")},
    )
    assert response.status_code == 422
    assert "at most 1024" in response.json()["detail"]["message"]

    # bad background
    response = client.put(
        "/v1/samples/1",
        data={"background": "#12345"},
        files={"image": ("x.png", png_bytes(s1), "image/png")},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_background"

    # bad slot
    response = client.put(
        "/v1/samples/3",
        data={"background": "0,0,0"},
        files={"image": ("x.png", png_bytes(s1), "image/png")},
    )
    assert response.status_code == 422


def test_size_mismatch_between_samples_flagged_on_upload(client):
    s1, s2, bg1, bg2, _, _ = scene()
    first = register(client, 1, s1, "%d,%d,%d" % bg1)
    assert first.json()["pair_size_match"] is True
    response = register(client, 2, s2[..., :4, :, :], "%d,%d,%d" % bg2)
    assert response.status_code == 200
    assert response.json()["pair_size_match"] is False
    # The mismatching sample is kept (it can be corrected next), but the pair
    # is not usable yet and replacing is rejected explicitly.
    listing = client.get("/v1/samples").json()
    assert listing["ready"] is True
    replacement = client.post(
        "/v1/replace",
        files={"background": ("new.png", png_bytes(s1), "image/png")},
    )
    assert replacement.status_code == 409
    assert replacement.json()["detail"]["code"] == "sample_size_mismatch"


def test_size_mismatch_between_samples_and_background(client):
    s1, s2, bg1, bg2, _, _ = scene()
    register(client, 1, s1, "%d,%d,%d" % bg1)
    register(client, 2, s2, "%d,%d,%d" % bg2)
    response = client.post(
        "/v1/replace",
        files={"background": ("new.png", png_bytes(s1[..., :3, :, :]), "image/png")},
    )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "background_size_mismatch"


def test_background_must_match_sample_size(client):
    s1, s2, bg1, bg2, _, _ = scene()
    register(client, 1, s1, "%d,%d,%d" % bg1)
    register(client, 2, s2, "%d,%d,%d" % bg2)
    response = client.post(
        "/v1/replace",
        files={
            "background": (
                "new.png",
                png_bytes(np.zeros((2, 2, 3), dtype=np.uint8)),
                "image/png",
            )
        },
    )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "background_size_mismatch"


def test_delete_sample(client):
    s1, _, bg1, _, _, _ = scene()
    register(client, 1, s1, "%d,%d,%d" % bg1)
    assert client.delete("/v1/samples/1").status_code == 200
    assert client.delete("/v1/samples/1").status_code == 404
    assert client.get("/v1/samples").json()["ready"] is False

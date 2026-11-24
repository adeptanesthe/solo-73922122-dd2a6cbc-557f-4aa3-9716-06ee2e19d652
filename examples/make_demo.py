"""Generate demo inputs for the API requests (self-contained, no downloads).

Two scenes are produced under ``examples/demo``:

``cel``
    Hard-edged animation cel: alpha is exactly 0 or 255.  Samples over pure
    black and pure white pin the frame well enough that the replacement is
    unique everywhere, so ``/v1/replace`` returns an image.

``soft``
    Soft alpha with a few hand-corrupted pixels in sample 2.  The replacement
    is not unique at semi-transparent pixels and two pixels contradict the two
    samples, so ``/v1/replace`` returns a mask with both regions marked.

Each scene writes ``sample_1.png``, ``sample_2.png`` (the two registered
samples), ``background.png`` (the new backdrop), plus ``frame.png`` for
reference (not used by the API).

Usage::

    python examples/make_demo.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent / "demo"
SIZE = 192
CEL_BACKGROUNDS = ((0x00, 0x00, 0x00), (0xFF, 0xFF, 0xFF))
SOFT_BACKGROUNDS = ((0x13, 0x2A, 0x5E), (0xE8, 0xC1, 0x6B))


def _canvas() -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    frame = np.zeros((SIZE, SIZE, 3), dtype=np.int64)
    frame[..., 0] = 230
    frame[..., 1] = 120
    frame[..., 2] = 40
    centre = SIZE / 2
    radius = np.hypot(xx - centre, yy - centre)
    stripe = (np.abs(yy - centre) < SIZE * 0.05) & (radius < SIZE * 0.33)
    frame[stripe] = (30, 200, 220)
    return frame, radius


def cel_scene() -> tuple[np.ndarray, np.ndarray]:
    frame, radius = _canvas()
    alpha = np.where(radius < SIZE * 0.35, 255, 0).astype(np.int64)
    return frame, alpha


def soft_scene() -> tuple[np.ndarray, np.ndarray]:
    frame, radius = _canvas()
    rng = np.random.default_rng(7)
    frame = np.clip(frame + rng.integers(-12, 13, frame.shape), 0, 255)
    fade = np.clip((SIZE * 0.40 - radius) / (SIZE * 0.15), 0.0, 1.0)
    alpha = np.round(fade * 255).astype(np.int64)
    return frame, alpha


def composite(frame: np.ndarray, alpha: np.ndarray, background) -> np.ndarray:
    bg = np.asarray(background, dtype=np.int64)
    blended = alpha[..., None] * frame + (255 - alpha[..., None]) * bg + 127
    return (blended // 255).astype(np.uint8)


def new_background() -> np.ndarray:
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    bg = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    bg[..., 0] = np.clip(16 + xx / SIZE * 90, 0, 255)
    bg[..., 1] = np.clip(60 - yy / SIZE * 50, 0, 255)
    bg[..., 2] = np.clip(30 + (xx + yy) / (2 * SIZE) * 180, 0, 255)
    return bg


def write_scene(name: str, frame, alpha, backgrounds) -> None:
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame.astype(np.uint8), mode="RGB").save(out / "frame.png")
    for index, background in enumerate(backgrounds, start=1):
        Image.fromarray(composite(frame, alpha, background), mode="RGB").save(
            out / f"sample_{index}.png"
        )
    Image.fromarray(new_background(), mode="RGB").save(out / "background.png")
    (out / "backgrounds.txt").write_text(
        "\n".join("#{:02x}{:02x}{:02x}".format(*bg) for bg in backgrounds) + "\n"
    )


REQUEST = r"""#!/usr/bin/env sh
# End-to-end replacement request for both demo scenes.
# Requires: curl, a running service (docker compose up, or uvicorn app.api:app).
set -eu
BASE="${BASE:-http://localhost:8000}"
DIR="$(cd "$(dirname "$0")" && pwd)"

decode() {
  python3 - "$1" "$2" <<'PY'
import base64, json, sys
payload = json.load(open(sys.argv[1]))
print("status:", payload["status"], "counts:", payload["counts"])
encoded = payload["image_png_base64"] or payload["mask_png_base64"]
open(sys.argv[2], "wb").write(base64.b64decode(encoded))
print("wrote", sys.argv[2])
PY
}

run_scene() {
  scene="$1"
  echo "== $scene =="
  first=$(sed -n 1p "$DIR/$scene/backgrounds.txt")
  second=$(sed -n 2p "$DIR/$scene/backgrounds.txt")
  curl -fsS -X PUT "$BASE/v1/samples/1" \
    -F "background=$first" \
    -F "image=@$DIR/$scene/sample_1.png;type=image/png"
  echo
  curl -fsS -X PUT "$BASE/v1/samples/2" \
    -F "background=$second" \
    -F "image=@$DIR/$scene/sample_2.png;type=image/png"
  echo
  curl -fsS -X POST "$BASE/v1/replace" \
    -F "background=@$DIR/$scene/background.png;type=image/png" \
    -F "format=json" \
    -o "$DIR/$scene/result.json"
  decode "$DIR/$scene/result.json" "$DIR/$scene/result.png"
}

run_scene cel
run_scene soft
"""


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    cel_frame, cel_alpha = cel_scene()
    write_scene("cel", cel_frame, cel_alpha, CEL_BACKGROUNDS)
    soft_frame, soft_alpha = soft_scene()
    write_scene("soft", soft_frame, soft_alpha, SOFT_BACKGROUNDS)

    # Corrupt two pixels of sample 2 in the soft scene: the two samples can no
    # longer come from one frame there, which the service reports separately.
    soft_sample_2 = ROOT / "soft" / "sample_2.png"
    array = np.array(Image.open(soft_sample_2))
    array[10, 10] = (3, 250, 7)
    array[11, 10] = (250, 3, 200)
    Image.fromarray(array, mode="RGB").save(soft_sample_2)

    script = ROOT / "request.sh"
    script.write_text(REQUEST)
    script.chmod(0o755)
    print(f"wrote demo files under {ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

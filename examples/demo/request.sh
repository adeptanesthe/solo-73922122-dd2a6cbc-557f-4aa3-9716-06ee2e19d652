#!/usr/bin/env sh
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

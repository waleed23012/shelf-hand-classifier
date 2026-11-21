#!/usr/bin/env bash
# Smoke test against a running API: is it up, is the model loaded, does /predict answer?
# Usage: bash scripts/smoke_test.sh <base_url>     e.g. http://localhost:8000
set -euo pipefail

fail() {
  echo "FAIL: $*"
  exit 1
}

[[ $# -eq 1 ]] || fail "usage: bash scripts/smoke_test.sh <base_url>"
BASE_URL="${1%/}"

# Use python, not jq: python is always there in this project, jq often is not.
# Try each name and keep the first that really runs: on Windows "python3" can be a
# Microsoft Store stub that exists on PATH but only prints an install hint.
PYTHON=""
for candidate in python3 python; do
  if "$candidate" -c 'import sys' >/dev/null 2>&1; then
    PYTHON="$candidate"
    break
  fi
done
[[ -n "$PYTHON" ]] || fail "python not found"

TMP_DIR="$(mktemp -d)"
# Git Bash on Windows: mktemp gives an MSYS path (/tmp/...) that native python/curl
# resolve differently. cygpath -m turns it into C:/... which all of them read the same.
if command -v cygpath >/dev/null 2>&1; then
  TMP_DIR="$(cygpath -m "$TMP_DIR")"
fi
trap 'rm -rf "$TMP_DIR"' EXIT

# 1. Health: the service is up and the model is loaded
health="$(curl -fsS --max-time 10 "$BASE_URL/health")" || fail "GET $BASE_URL/health did not return 200"
echo "$health"
echo "$health" | "$PYTHON" -c '
import json, sys
body = json.load(sys.stdin)
sys.exit(0 if body.get("status") == "ok" and body.get("model_loaded") is True else 1)
' || fail "/health is not status=ok with model_loaded=true"

# 2. Predict: no test image is committed under tests/, so generate a 64x64 PNG
# with the standard library only (the machine running this may not have Pillow).
IMAGE="$TMP_DIR/smoke.png"
"$PYTHON" - "$IMAGE" <<'EOF'
import struct, sys, zlib

w = h = 64
# A gradient instead of a flat colour, so it looks a bit like a real image.
# Each PNG row starts with filter byte 0, then RGB triples.
rows = b"".join(
    b"\x00" + b"".join(bytes((x * 4, y * 4, 128)) for x in range(w)) for y in range(h)
)

def chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

png = (b"\x89PNG\r\n\x1a\n"
       + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
       + chunk(b"IDAT", zlib.compress(rows))
       + chunk(b"IEND", b""))
with open(sys.argv[1], "wb") as f:
    f.write(png)
EOF

predict="$(curl -fsS --max-time 60 -F "file=@$IMAGE;type=image/png" "$BASE_URL/predict")" \
  || fail "POST $BASE_URL/predict did not return 200"
echo "$predict"
echo "$predict" | "$PYTHON" -c '
import json, sys
sys.exit(0 if "decision" in json.load(sys.stdin) else 1)
' || fail "/predict response has no decision field"

echo "PASS: smoke test ok"

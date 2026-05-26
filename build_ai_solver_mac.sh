#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="$ROOT/ai_solver.py"
DIST="$ROOT/dist"
BUILD="$ROOT/build"
SPEC="$ROOT/AISolver.spec"
VENDOR_TESSERACT="$ROOT/vendor/tesseract"

rm -rf "$DIST" "$BUILD"
rm -f "$SPEC"

ARGS=(
  --noconfirm
  --clean
  --windowed
  --name AISolver
  --collect-all google.genai
  --exclude-module torch
  --exclude-module torchvision
  --exclude-module tensorflow
  --exclude-module keras
  --exclude-module pandas
  --exclude-module scipy
  --exclude-module matplotlib
  --exclude-module onnxruntime
  --exclude-module sqlalchemy
)

if [[ -d "$VENDOR_TESSERACT" ]]; then
  ARGS+=(--add-data "$VENDOR_TESSERACT:vendor/tesseract")
fi

pyinstaller "${ARGS[@]}" "$ENTRY"

echo "Build complete: $DIST/AISolver.app"

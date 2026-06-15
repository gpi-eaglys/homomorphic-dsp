#!/usr/bin/env bash
# Downloads ESC-50 dataset and extracts it into the asset dir
#  assets/
#  └── esc-50
#      └── ESC-50-master
#          ├── audio
#          └── meta


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSET_DIR="$SCRIPT_DIR/../assets/esc-50"
ZIP="$ASSET_DIR/esc50.zip"

mkdir -p "$ASSET_DIR"

echo "Downloading ESC-50..."
curl -L -o "$ZIP" https://github.com/karoldvl/ESC-50/archive/master.zip

echo "Extracting..."
unzip -q "$ZIP" -d "$ASSET_DIR"
mv "$ASSET_DIR/ESC-50-master"/* "$ASSET_DIR/"
rmdir "$ASSET_DIR/ESC-50-master"
rm "$ZIP"

echo "Done. Dataset at: $ASSET_DIR"

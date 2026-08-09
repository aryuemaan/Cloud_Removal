#!/usr/bin/env bash
# Usage: ./scripts/run_inference.sh <checkpoint> <input_tif> <output_tif> [sar_tif]
set -euo pipefail
CKPT=${1:?checkpoint path required}
INPUT=${2:?input GeoTIFF required}
OUTPUT=${3:?output path required}
SAR=${4:-}

if [ -n "$SAR" ]; then
  python -m src.inference.predict --config config/config.yaml --checkpoint "$CKPT" --input "$INPUT" --output "$OUTPUT" --sar "$SAR"
else
  python -m src.inference.predict --config config/config.yaml --checkpoint "$CKPT" --input "$INPUT" --output "$OUTPUT"
fi

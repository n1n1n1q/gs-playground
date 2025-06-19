#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 /path/to/images"
    exit 1
fi

INPUT_IMAGES=$(realpath "$1")
PROJECT_NAME=$(basename "$INPUT_IMAGES")_glomap
mkdir -p "$PROJECT_NAME"/{images,sparse}

cp "$INPUT_IMAGES"/* "$PROJECT_NAME/images/"

cd "$PROJECT_NAME"

SECONDS=0

echo "[INFO] Running COLMAP feature extraction..."
colmap feature_extractor \
    --image_path images \
    --database_path database.db

echo "[INFO] Running COLMAP exhaustive matcher..."
colmap exhaustive_matcher \
    --database_path database.db

echo "[INFO] Running GLOMAP mapper (sparse reconstruction)..."
glomap mapper \
    --database_path database.db \
    --image_path images \
    --output_path sparse \
    --GlobalPositioning.use_gpu 1 \
    --BundleAdjustment.use_gpu 1


echo "[DONE] GLOMAP sparse reconstruction complete. Output saved to $PROJECT_NAME/sparse/"
echo "[INFO] Total time taken: $SECONDS seconds"

#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 /path/to/images"
    exit 1
fi

INPUT_IMAGES=$(realpath "$1")
BASE_NAME=$(basename "$INPUT_IMAGES")_colmap
PROJECT_NAME="$BASE_NAME"
i=1

while [ -d "$PROJECT_NAME" ]; do
    PROJECT_NAME="${BASE_NAME}_$i"
    ((i++))
done

mkdir -p "$PROJECT_NAME"/{distorted,sparse}
mkdir -p "$PROJECT_NAME/sparse/0"
mkdir -p "$PROJECT_NAME/distorted/images"

cp "$INPUT_IMAGES"/* "$PROJECT_NAME/distorted/images/"

cd "$PROJECT_NAME"

SECONDS=0

echo "[INFO] Running COLMAP feature extraction..."
colmap feature_extractor \
    --database_path distorted/database.db \
    --image_path distorted/images \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model PINHOLE

echo "[INFO] Running COLMAP exhaustive matcher..."
colmap exhaustive_matcher \
    --database_path distorted/database.db

echo "[INFO] Running COLMAP mapper (sparse reconstruction)..."
colmap mapper \
    --database_path distorted/database.db \
    --image_path distorted/images \
    --output_path sparse \
    --Mapper.ba_global_function_tolerance=0.000001 \
    --Mapper.ba_use_gpu 1   

echo "[INFO] Undistorting images and converting to pinhole model..."
colmap image_undistorter \
    --image_path distorted/images \
    --input_path sparse/0 \
    --output_path . \
    --output_type COLMAP

# echo "[INFO] Converting undistorted model to PLY..."
# colmap model_converter \
#     --input_path sparse/0 \
#     --output_path points3D.ply \
#     --output_type PLY

echo "[DONE] COLMAP sparse reconstruction complete. Output saved to $PROJECT_NAME/points3D.ply"
echo "[INFO] Total time taken: $SECONDS seconds"
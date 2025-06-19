    #!/bin/bash

    if [ "$#" -ne 1 ]; then
        echo "Usage: $0 /path/to/images"
        exit 1
    fi

    INPUT_IMAGES=$(realpath "$1")
    PROJECT_NAME=$(basename "$INPUT_IMAGES")_colmap
    mkdir -p "$PROJECT_NAME"/{images,sparse}

    cp "$INPUT_IMAGES"/* "$PROJECT_NAME/images/"

    cd "$PROJECT_NAME"

    SECONDS=0

    echo "[INFO] Running COLMAP feature extraction..."
    colmap feature_extractor \
        --database_path database.db \
        --image_path images

    echo "[INFO] Running COLMAP exhaustive matcher..."
    colmap exhaustive_matcher \
        --database_path database.db

    echo "[INFO] Running COLMAP mapper (sparse reconstruction)..."
    colmap mapper \
        --database_path database.db \
        --image_path images \
        --output_path sparse \
        --Mapper.ba_global_function_tolerance=0.000001 \
        --Mapper.ba_use_gpu 1

    echo "[INFO] Converting model to PLY..."
    colmap model_converter \
        --input_path sparse/0 \
        --output_path sparse/0 \
        --output_type PLY

    echo "[DONE] Sparse reconstruction complete. Output saved to $PROJECT_NAME/sparse/0/points3D.ply"
    echo "[INFO] Total time taken: $SECONDS seconds"
"""
Convert script
"""
import os
from pipeline.extractors import colmap_sift
from pipeline.pipeline import Pipeline
from utils.resize import resize_image

if __name__ == "__main__":
    IMAGES_PATH = "data/images"
    OUTPUT_PATH = "dataset/"
    RESIZED_PATH = "dataset/images"
    os.makedirs(RESIZED_PATH, exist_ok=True)
    resize_image(IMAGES_PATH, RESIZED_PATH)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    p = Pipeline(
        colmap_sift.ColmapSift(IMAGES_PATH, OUTPUT_PATH+"database.db"),
    )
    p()

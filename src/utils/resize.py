import os
from PIL import Image

MAX_SIZE   = 1000

def resize_image(image_path, output_path):
    """
    Resize images in the specified directory to a maximum size of 1000 pixels.
    """
    os.makedirs(output_path, exist_ok=True)

    for fname in os.listdir(image_path):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            continue

        in_path  = os.path.join(image_path,  fname)
        out_path = os.path.join(output_path, fname)

        with Image.open(in_path) as img:
            img.thumbnail((MAX_SIZE, MAX_SIZE))
            img.save(out_path)

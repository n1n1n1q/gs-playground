from PIL import Image
import os
import argparse

supported_formats = ('.png', '.jpg', '.jpeg')

def resize_images(folder, max_size):
    for filename in os.listdir(folder):
        if filename.lower().endswith(supported_formats):
            filepath = os.path.join(folder, filename)
            
            with Image.open(filepath) as img:
                w, h = img.size
                scale = max_size / max(w, h)
                
                if scale < 1.0: 
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    resized_img = img.resize((new_w, new_h), Image.LANCZOS)
                    resized_img.save(filepath)
                    print(f"Resized {filename} to {new_w}x{new_h}")
                else:
                    print(f"Skipped {filename}, size {w}x{h}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Resize images to max size.")
    parser.add_argument('folder', type=str)
    parser.add_argument('--max_size', type=int, default=1000)
    args = parser.parse_args()

    resize_images(args.folder, args.max_size)
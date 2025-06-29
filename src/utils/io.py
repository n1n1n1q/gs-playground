"""
I/O utils
"""
import os
import cv2
import open3d as o3d
from pathlib import Path
from typing import List
from camera.camera import Camera
from view.camera_view import CameraView

def save_cameras_txt(cameras: List[Camera], dir: str):
    """
    Save camera parameters to a text file.
    """
    os.makedirs(dir, exist_ok=True)
    with open(f"{dir}/cameras.txt", "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for cam in cameras:
            params = [cam.focal_length, cam.focal_length, cam.width / 2, cam.height / 2]
            f.write(f"{cam.id} {cam.model} {cam.width} {cam.height} {' '.join(map(str, params))}\n")

def save_images_txt(views: List[CameraView], cameras: List[Camera], dir: str, 
                    save_new_images: bool = True, conf_threshold: float = 0.0):
    """
    Save camera views to a text file.
    """
    os.makedirs(dir, exist_ok=True)
    os.makedirs(f"{dir}/images", exist_ok=True)
    qvecs = [cameras[view.camera_id].qvec() for view in views]
    tvecs = [cameras[view.camera_id].tvec() for view in views]
    with open(f"{dir}/images.txt", "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for id, view in enumerate(views):
            if view.confidence < conf_threshold:
                print(f"[WARNING] View {id + 1} has confidence {view.confidence}, which is below the threshold {conf_threshold}. Skipping this view.")
                continue

            img_name = Path(view.img_path).name if view.img_path else f"IMG{id + 1}.jpg"
            f.write(f"{id + 1} {' '.join(map(str, qvecs[id]))} {' '.join(map(str, tvecs[id]))} {view.camera_id} {img_name}\n")
            f.write("\n")

            if view.img is not None and save_new_images:
                img = cv2.cvtColor(view.img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"{dir}/images/{img_name}", img)
            elif view.img_path:
                cv2.imwrite(f"{dir}/images/{img_name}", cv2.imread(view.img_path))
            else:
                print(f"[WARNING] Image for view {id + 1} is None, skipping saving image.")

def save_points3D_txt(pcd: o3d.geometry.PointCloud, dir: str):
    with open(f"{dir}/points3D.txt", "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for i in range(len(pcd.points)):
            x, y, z = pcd.points[i]
            r, g, b = pcd.colors[i]
            f.write(f"{i} {x} {y} {z} {int(r)} {int(g)} {int(b)} {0}\n")

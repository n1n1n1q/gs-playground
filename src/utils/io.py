"""
I/O utils
"""

import os
import struct

import cv2
import open3d as o3d

from pathlib import Path
from typing import List

from src.camera.camera import Camera
from src.view.camera_view import CameraView


def save_image(
    view: CameraView, dir: str, img_name: str, save_new_images: bool = True
) -> None:
    if view.img is not None and save_new_images:
        img = cv2.cvtColor(view.img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{dir}/images/{img_name}", img)
    elif view.img_path:
        cv2.imwrite(f"{dir}/images/{img_name}", cv2.imread(view.img_path))
    else:
        print(f"[WARNING] Image for view {id + 1} is None, skipping saving image.")


def write_cameras_txt(cameras: List[Camera], dir: str) -> None:
    """
    Save camera parameters to a text file.
    """
    os.makedirs(dir, exist_ok=True)
    with open(f"{dir}/cameras.txt", "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for cam in cameras:
            params = [cam.focal_length, cam.focal_length, cam.width / 2, cam.height / 2]
            f.write(
                f"{cam.id} {cam.model} {cam.width} {cam.height} {' '.join(map(str, params))}\n"
            )


def write_images_txt(
    views: List[CameraView],
    dir: str,
    save_new_images: bool = True,
    conf_threshold: float = 0.0,
) -> None:
    """
    Save camera views to a text file.
    """
    os.makedirs(dir, exist_ok=True)
    os.makedirs(f"{dir}/images", exist_ok=True)
    qvecs = [view.qvec() for view in views]
    tvecs = [view.tvec() for view in views]
    with open(f"{dir}/images.txt", "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for id, view in enumerate(views):
            if view.confidence < conf_threshold:
                print(
                    f"[WARNING] View {id + 1} has confidence {view.confidence}, which is below the threshold {conf_threshold}. Skipping this view."
                )
                continue

            img_name = Path(view.img_path).name if view.img_path else f"IMG{id + 1}.jpg"
            f.write(
                f"{id + 1} {' '.join(map(str, qvecs[id]))} {' '.join(map(str, tvecs[id]))} {view.camera_id} {img_name}\n"
            )
            f.write("\n")

            save_image(view, dir, img_name, save_new_images)


def write_points3D_txt(pcd: o3d.geometry.PointCloud, dir: str) -> None:
    with open(f"{dir}/points3D.txt", "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write(
            "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        )
        for i in range(len(pcd.points)):
            x, y, z = pcd.points[i]
            r, g, b = pcd.colors[i] * 255
            f.write(f"{i} {x} {y} {z} {int(r)} {int(g)} {int(b)} {0}\n")


def write_next_bytes(
    fid, data: any, format_char_sequence: str, endian_character: str = "<"
) -> None:
    """pack and write to a binary file.
    :param fid:
    :param data: data to send, if multiple elements are sent at the same time,
    they should be encapsuled either in a list or a tuple
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    should be the same length as the data list or tuple
    :param endian_character: Any of {@, =, <, >, !}
    """
    if isinstance(data, (list, tuple)):
        bytes = struct.pack(endian_character + format_char_sequence, *data)
    else:
        bytes = struct.pack(endian_character + format_char_sequence, data)
    fid.write(bytes)


def write_cameras_binary(cameras: List[Camera], dir: str) -> None:
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    with open(f"{dir}/cameras.bin", "wb") as fid:
        write_next_bytes(fid, len(cameras), "Q")
        for cam in cameras:
            camera_properties = [cam.id, 1, cam.width, cam.height]
            write_next_bytes(fid, camera_properties, "iiQQ")
            params = [cam.focal_length, cam.focal_length, cam.width / 2, cam.height / 2]
            for p in params:
                write_next_bytes(fid, float(p), "d")
    return cameras


def write_images_binary(
    views: List[CameraView], dir: str, conf_threshold: float = 0.0
) -> None:
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    with open(f"{dir}/images.bin", "wb") as fid:
        write_next_bytes(
            fid, len([v for v in views if v.confidence >= conf_threshold]), "Q"
        )
        for id, view in enumerate(views):
            if view.confidence < conf_threshold:
                print(
                    f"[WARNING] View {id + 1} has confidence {view.confidence}, which is below the threshold {conf_threshold}. Skipping this view."
                )
                continue

            write_next_bytes(fid, id, "i")
            write_next_bytes(fid, list(view.qvec()), "dddd")
            write_next_bytes(fid, list(view.tvec()), "ddd")
            write_next_bytes(fid, view.camera_id, "i")

            img_name = Path(view.img_path).name if view.img_path else f"IMG{id}.jpg"
            fid.write(img_name.encode("utf-8") + b"\x00")

            write_next_bytes(fid, 0, "Q")
            save_image(view, dir, img_name, save_new_images=True)


def write_points3D_binary(pcd: o3d.geometry.PointCloud, dir: str) -> None:
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    with open(f"{dir}/points3D.bin", "wb") as fid:
        write_next_bytes(fid, len(pcd.points), "Q")
        for i in range(len(pcd.points)):
            write_next_bytes(fid, i, "Q")
            write_next_bytes(fid, list(pcd.points[i]), "ddd")
            write_next_bytes(fid, [int(i) for i in pcd.colors[i] * 255], "BBB")
            write_next_bytes(fid, 0, "d")
            write_next_bytes(fid, 0, "Q")

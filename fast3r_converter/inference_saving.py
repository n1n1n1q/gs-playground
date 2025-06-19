import torch
import os
import cv2
import argparse
import time
import numpy as np
from fast3r.dust3r.utils.image import load_images
from fast3r.dust3r.inference_multiview import inference
from fast3r.models.fast3r import Fast3R
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule
from scipy.spatial.transform import Rotation as R

from postproccess import *

CONF_THRESHOLD = 2

def save_cameras_txt(views, estimated_focals, save_dir):
    width, height = views[0]["img"].shape[3], views[0]["img"].shape[2]
    print(views[0]["img"].shape)
    focal = estimated_focals[0][0]
    camera_model = "PINHOLE"
    camera_id = 1
    cameras_path = os.path.join(save_dir, "cameras.txt")

    with open(cameras_path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"{camera_id} {camera_model} {width} {height} {focal} {focal} {width/2} {height/2}\n")


def save_images_txt(views, camera_poses, save_dir):
    image_dir = os.path.join(save_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    images_path = os.path.join(save_dir, "images.txt")
    camera_id = 1

    with open(images_path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

        for img_id, (view, pose) in enumerate(zip(views, camera_poses), start=1):
            if max_confs[img_id] < CONF_THRESHOLD:
                continue
            R_c2w = pose[:3, :3]
            t_c2w = pose[:3, 3]

            R_w2c = R_c2w.T
            t_w2c = -R_w2c @ t_c2w
            q_w2c = R.from_matrix(R_w2c).as_quat()
            q_w2c = [q_w2c[3], q_w2c[0], q_w2c[1], q_w2c[2]]

            img_filename = f"IMG{img_id}.png"
            img_save_path = os.path.join(image_dir, img_filename)
            img = np.transpose(view["img"][0].cpu().numpy(), (1, 2, 0))
            img = ((img + 1) * 127.5).clip(0, 255).astype(np.uint8)
            cv2.imwrite(img_save_path, img)

            f.write(f"{img_id} {' '.join(map(str, q_w2c))} {' '.join(map(str, t_w2c))} {camera_id} {img_filename}\n")
            f.write("\n")

def save_points3D_txt(preds, views, confidence, save_dir):
    points_path = os.path.join(save_dir, "points3D.txt")
    with open(points_path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        point_id = 0
        
        keep_frac = 1.0 - confidence

        for i in range(len(preds)):
            if max_confs[i] < CONF_THRESHOLD:
                continue
            pts3d = preds[i]["pts3d_local_aligned_to_global"].cpu().numpy()
            confidences = preds[i]["conf"].cpu().numpy()
            colors = views[i]["img"].cpu().numpy()
            print(f"=======\nShapes\n pts3d: {pts3d.shape}, confidences: {confidences.shape}, colors: {colors.shape}")
            
            pts3d_flat = pts3d.reshape(-1, 3)
            confidences_flat = confidences.reshape(-1)
            colors_flat = colors.transpose(0, 2, 3, 1).reshape(-1, 3)
            
            conf_min, conf_max = confidences_flat.min(), confidences_flat.max()
            confidences_normalized = (confidences_flat - conf_min) / (conf_max - conf_min + 1e-8)
            
            k = max(1, int(len(confidences_flat) * keep_frac))
            idx = np.argpartition(-confidences_flat, k - 1)[:k]
            pts3d_filtered = pts3d_flat[idx]
            confidences_filtered = confidences_normalized[idx]
            colors_filtered = colors_flat[idx]
            
            for xyz, conf, rgb in zip(pts3d_filtered, confidences_filtered, colors_filtered):
                x, y, z = xyz
                r, g, b = rgb
                r, g, b = ((r + 1) * 127.5).clip(0, 255), ((g + 1) * 127.5).clip(0, 255), ((b + 1) * 127.5).clip(0, 255)
                error = 1.0 - conf
                f.write(f"{point_id} {x} {y} {z} {int(r)} {int(g)} {int(b)} {error} \n")
                point_id += 1

def get_confidence_per_view(preds):
    confidences = []
    for i in range(len(preds)):
        conf = np.max(preds[i]["conf"].cpu().numpy())
        confidences.append(conf)
    return confidences

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference and save results from Fast3R model.")
    parser.add_argument("--input", "-i", type=str, help="Path to the input directory containing images.", default="data")
    parser.add_argument("--raw", "-r", action="store_true", help="Save raw results without processing.", default=False)
    args = parser.parse_args()
    try:
        model = Fast3R.from_pretrained("models/fast3r")
    except:
        model = Fast3R.from_pretrained("jedyang97/Fast3R_ViT_Large_512")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    lit_module = MultiViewDUSt3RLitModule.load_for_inference(model)
    model.eval()
    lit_module.eval()

    load_start = time.time()
    images = load_images(args.input, size=512)
    load_end = time.time()
    print(f"Images loaded in {load_end - load_start:.2f} seconds")

    output_dict, profiling_info = inference(
        images,
        model,
        device,
        dtype=torch.float32,
        verbose=True,
        profiling=True,
    )
    align_start = time.time()
    confidence = 0.7
    lit_module.align_local_pts3d_to_global(
        preds=output_dict["preds"],
        views=output_dict["views"],
        min_conf_thr_percentile=confidence,
    )
    align_end = time.time()
    print(f"Local points aligned to global in {align_end - align_start:.2f} seconds")

    pose_start = time.time()
    print("Estimating camera poses...")
    poses_c2w_batch, estimated_focals = MultiViewDUSt3RLitModule.estimate_camera_poses(
        output_dict["preds"],
        niter_PnP=100,
        focal_length_estimation_method="first_view_from_global_head",
    )
    camera_poses = poses_c2w_batch[0]
    pose_end = time.time()
    print(f"Camera poses estimated in {pose_end - pose_start:.2f} seconds")

    save_start = time.time()

    max_confs = get_confidence_per_view(output_dict["preds"])

    print("Saving results...")
    save_dir_raw = "raw"
    save_dir = "processed"

    os.makedirs(save_dir, exist_ok=True)
    if args.raw:
        os.makedirs(save_dir_raw, exist_ok=True)
        save_cameras_txt(output_dict["views"], estimated_focals, save_dir_raw)
        save_images_txt(output_dict["views"], camera_poses, save_dir_raw)
        save_points3D_txt(output_dict["preds"], output_dict["views"], confidence, save_dir_raw)
        save_raw_end = time.time()
        print(f"Results saved in {save_dir_raw} in {save_raw_end - save_start:.2f} seconds")

    save_start = time.time()
    save_cameras_txt(output_dict["views"], estimated_focals, save_dir)
    save_images_txt(output_dict["views"], camera_poses, save_dir)
    pcds = inference_to_pcds(output_dict["preds"], output_dict["views"], conf_threshold=confidence, debug=True)
    merged_pcd = {0: merge_pointclouds(pcds)}
    proccessed_pcd = downsample_per_frame(merged_pcd, voxel_size=0.02)
    save_points3D(proccessed_pcd[0], save_dir)
    save_end = time.time()
    print(f"Processed results saved in {save_dir} in {save_end - save_start:.2f} seconds")
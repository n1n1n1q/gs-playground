"""

"""
import open3d as o3d
import numpy as np

def inference_to_pcds(preds, views, conf_threshold = 0., debug = False):
    """
    Convert Fast3R inference results to custom Open3D-based PointCloud objects.
    """
    keep_frac = 1.0 - conf_threshold
    clouds = dict()
    for i in range(len(preds)):
        pts3d = preds[i]["pts3d_local_aligned_to_global"].cpu().numpy()
        confidences = preds[i]["conf"].cpu().numpy()
        colors = views[i]["img"].cpu().numpy()
        if debug:
            print(f"=======\nProcessing view {i}\nShapes\n pts3d: {pts3d.shape}, confidences: {confidences.shape}, colors: {colors.shape}")
        
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
        colors_filtered = ((colors_filtered + 1) * 127.5).clip(0, 255).astype(np.uint8)
        clouds[i] = o3d.geometry.PointCloud()
        clouds[i].points = o3d.utility.Vector3dVector(pts3d_filtered)
        clouds[i].colors = o3d.utility.Vector3dVector(colors_filtered)
    return clouds

def merge_pointclouds(pointclouds):
    """
    Merge multiple PointCloud objects into a single PointCloud.
    """
    merged_points = np.concatenate([pc.points for pc in pointclouds.values()])
    merged_colors = np.concatenate([pc.colors for pc in pointclouds.values()])

    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(merged_points)
    merged_pcd.colors = o3d.utility.Vector3dVector(merged_colors)

    return merged_pcd

def downsample_per_frame(pointclouds, voxel_size=0.01):
    """
    Downsample each PointCloud in the dictionary using Open3D's voxel downsampling.
    """
    downsampled_clouds = {}
    for i, pc in pointclouds.items():
        o3d_pc = o3d.geometry.PointCloud()
        o3d_pc.points = o3d.utility.Vector3dVector(pc.points)
        o3d_pc.colors = o3d.utility.Vector3dVector(pc.colors)
        downsampled_pc = o3d_pc.voxel_down_sample(voxel_size=voxel_size)
        downsampled_clouds[i] = o3d.geometry.PointCloud()
        downsampled_clouds[i].points = o3d.utility.Vector3dVector(np.asarray(downsampled_pc.points))
        downsampled_clouds[i].colors = o3d.utility.Vector3dVector(np.asarray(downsampled_pc.colors))
    return downsampled_clouds

def save_points3D(pcd, dir):
    with open(f"{dir}/points3D.txt", "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for i in range(len(pcd.points)):
            x, y, z = pcd.points[i]
            r, g, b = pcd.colors[i]
            f.write(f"{i} {x} {y} {z} {int(r)} {int(g)} {int(b)} {0}\n")

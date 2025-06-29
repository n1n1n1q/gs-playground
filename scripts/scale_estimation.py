import open3d as o3d
import numpy as np

import argparse

def estimate_scale(A, B):
    A_mean = A.mean(0)
    B_mean = B.mean(0)
    A_centered = A - A_mean
    B_centered = B - B_mean

    norm_A = np.linalg.norm(A_centered)
    norm_B = np.linalg.norm(B_centered)

    return norm_B / norm_A

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Estimate scale between two point clouds.")
    parser.add_argument("-p1", "--pointcloud1", type=str, default="faster.ply",)
    parser.add_argument("-p2", "--pointcloud2", type=str, default="colmap.ply",)
    parser.add_argument("-n", "--n_experiments", type=int, default=1000,
                        help="Number of experiments to run for scale estimation.")
    args = parser.parse_args()

    ply1_init = o3d.t.io.read_point_cloud(args.pointcloud1, format="ply")
    ply2_init = o3d.t.io.read_point_cloud(args.pointcloud2, format="ply")

    ply1 = ply1_init.to_legacy()
    ply2 = ply2_init.to_legacy()

    ply1.paint_uniform_color([0.0, 1.0, 0.0])
    ply2.paint_uniform_color([1.0, 0.0, 0.0])

    ply1 = ply1.voxel_down_sample(voxel_size=0.01)
    ply2 = ply2.voxel_down_sample(voxel_size=0.01)

    cl, ind = ply1.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    ply1 = ply1.select_by_index(ind)

    cl, ind = ply2.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    ply2 = ply2.select_by_index(ind)
    
    scales = []

    for i in range(args.n_experiments):
        A = np.asarray(ply1.points)
        B = np.asarray(ply2.points)
    
        A = A[np.random.choice(len(A), 1000, replace=False)]
        B = B[np.random.choice(len(B), 1000, replace=False)]

        scales.append(estimate_scale(A, B))
    scale = np.mean(scales)
    print(f"Estimated scale from sparse to dense: {scale}")
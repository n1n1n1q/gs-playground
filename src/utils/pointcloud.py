"""

"""
import open3d as o3d
import numpy as np


def scale_pointcloud(pointcloud, scaling_factor):
    """
    Scale the point cloud by a given factor (in place).
    """
    points = pointcloud.points
    pointcloud.points = o3d.utility.Vector3dVector(
    np.asarray(points) * scaling_factor
)
"""
Sparse SfM from Fast3R
"""
import open3d as o3d
import numpy as np

from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule

from src.camera.camera import Camera
from src.view.camera_view import CameraView
from src.utils.pointcloud import scale_pointcloud

SCALING_FACTOR = 29.4

class Fast3RSfM:
    def __init__(self, output_dict):
        self.output_dict = output_dict
        self.cameras = []
        self.views = []
        self.pcd = None
        self.resolution_scaling = 1.0

    def __call__(self, conf_thr=0.0, downsample=False, voxel_size=0.01):
        """
        Preprocess the output dictionary to filter views based on confidence.
        """
        poses_c2w_batch, estimated_focals = MultiViewDUSt3RLitModule.estimate_camera_poses(
            self.output_dict["preds"],
            niter_PnP=100,
            focal_length_estimation_method="first_view_from_global_head",
        )
        camera_poses = poses_c2w_batch[0]
        self._save_cameras(estimated_focals)
        self._save_views(camera_poses)
        self._inference_to_pcds(conf_thr)

        if downsample:
            self.pcd = self.pcd.voxel_down_sample(voxel_size=voxel_size)

        
        # self.cameras[0] *= SCALING_FACTOR * self.resolution_scaling
        self.views = [view * SCALING_FACTOR * self.resolution_scaling for view in self.views]
        scale_pointcloud(self.pcd, SCALING_FACTOR * self.resolution_scaling)

    def _save_cameras(self, focals):
        width, height = self.output_dict["views"][0]["img"].shape[3], self.output_dict["views"][0]["img"].shape[2]
        self.cameras = [
            Camera(
                id = 1,
                model = "PINHOLE",
                width = width,
                height = height,
                focal_length = focals[0][0]
            )
        ]

    def _save_views(self, camera_poses):

        for img_id, (view, pred, pose) in enumerate(zip(self.output_dict["views"], 
                                                self.output_dict["preds"], camera_poses), start=1):
            img = np.transpose(view["img"][0].cpu().numpy(), (1, 2, 0))
            img = ((img + 1) * 127.5).clip(0, 255).astype(np.uint8)
            self.views.append(
                CameraView(
                    img_path=None,
                    camera_id=1,
                    confidence=np.max(pred["conf"].cpu().numpy()),
                    extrinsics=pose,
                    img=img,
                )
            )

    def _inference_to_pcds(self, conf_thr = 0.0):
        """

        """
        keep_frac = 1.0 - conf_thr

        preds = self.output_dict["preds"]
        views = self.output_dict["views"]

        cl_points = np.ndarray([])
        cl_colors = np.ndarray([])
        for i in range(len(preds)):
            pts3d = preds[i]["pts3d_local_aligned_to_global"].cpu().numpy()
            confidences = preds[i]["conf"].cpu().numpy()
            colors = views[i]["img"].cpu().numpy()
            
            pts3d_flat = pts3d.reshape(-1, 3)
            confidences_flat = confidences.reshape(-1)
            colors_flat = colors.transpose(0, 2, 3, 1).reshape(-1, 3)
            
            k = max(1, int(len(confidences_flat) * keep_frac))
            idx = np.argpartition(-confidences_flat, k - 1)[:k]
            pts3d_filtered = pts3d_flat[idx]

            colors_filtered = colors_flat[idx]
            colors_filtered = ((colors_filtered + 1) * 127.5).clip(0, 255).astype(np.uint8)

            cl_points = np.concatenate((cl_points, pts3d_filtered), axis=0) if cl_points.shape else pts3d_filtered
            cl_colors = np.concatenate((cl_colors, colors_filtered), axis=0) if cl_colors.shape else colors_filtered

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cl_points)
        pcd.colors = o3d.utility.Vector3dVector(cl_colors / 255.0)
        
        self.pcd = pcd

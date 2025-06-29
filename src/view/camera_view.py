"""
Camera view / image class
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

@dataclass
class CameraView:
    """
    Camera view class to hold image and camera parameters.
    """
    img_path: str
    confidence: float
    img: Optional[any] = None
    camera_id: int = 1
    extrinsics: np.ndarray = None

    def qvec(self) -> np.ndarray:
        """
        Get the rotation vector from the extrinsics matrix.
        If extrinsics are not set, return an identity rotation.
        """
        if self.extrinsics is None:
            return np.array([1, 0, 0, 0])
        
        R = self.extrinsics[:3, :3]
        Rxx, Rxy, Rxz = R[0]
        Ryx, Ryy, Ryz = R[1]
        Rzx, Rzy, Rzz = R[2]
        
        K = np.array([
            [Rxx - Ryy - Rzz,     Ryx + Rxy,       Rzx + Rxz,       Ryz - Rzy],
            [Ryx + Rxy,           Ryy - Rxx - Rzz, Rzy + Ryz,       Rzx - Rxz],
            [Rzx + Rxz,           Rzy + Ryz,       Rzz - Rxx - Ryy, Rxy - Ryx],
            [Ryz - Rzy,           Rzx - Rxz,       Rxy - Ryx,       Rxx + Ryy + Rzz]
        ]) / 3.0
        
        eigvals, eigvecs = np.linalg.eigh(K)
        qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]  # [w, x, y, z]
        
        if qvec[0] < 0:
            qvec *= -1
        
        return qvec
    
    def tvec(self) -> np.ndarray:
        """
        Get the translation vector from the extrinsics matrix.
        Converts from camera-to-world to world-to-camera pose.
        If extrinsics are not set, return a zero vector.
        """
        if self.extrinsics is None:
            return np.zeros(3)

        R_c2w = self.extrinsics[:3, :3]
        t_c2w = self.extrinsics[:3, 3]

        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ t_c2w

        return t_w2c
    
    def __imul__(self, num: float) -> 'CameraView':
        """
        Scale the camera view parameters in place by a given factor.
        """
        if self.extrinsics is not None:
            self.extrinsics[:3, 3] *= num
        return self
    
    def __mul__(self, num: float) -> 'CameraView':
        """
        Scale the camera view parameters by a given factor.
        """
        new_view = self.__copy__()
        new_view *= num
        return new_view
    
    def __copy__(self):
        """
        Create a copy of the camera view instance.
        """
        return CameraView(
            img_path=self.img_path,
            img=self.img,
            camera_id=self.camera_id,
            confidence=self.confidence,
            extrinsics=np.copy(self.extrinsics) if self.extrinsics is not None else None
        )
    
    def __repr__(self):
        """
        String representation of the CameraView instance.
        """
        return (f"CameraView(img_path={self.img_path}, confidence={self.confidence}, "
                f"camera_id={self.camera_id}, extrinsics={self.extrinsics})")
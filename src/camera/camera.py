"""
Camera class
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    focal_length: float

    def __imul__(self, num: float) -> "Camera":
        """
        Scale the camera parameters in place by a given factor
        """
        self.width = int(self.width * num)
        self.height = int(self.height * num)
        self.focal_length = self.focal_length * num
        return self

    def __mul__(self, num: float) -> "Camera":
        """
        Scale the camera parameters by a given factor
        """
        new_camera = self.__copy__()
        new_camera *= num
        return new_camera

    def __copy__(self) -> "Camera":
        """
        Create a copy of the camera instance.
        """
        return Camera(
            id=self.id,
            model=self.model,
            width=self.width,
            height=self.height,
            focal_length=self.focal_length,
            extrinsics=(
                np.copy(self.extrinsics) if self.extrinsics is not None else None
            ),
        )

"""
Base class for Raycast engines for mesh-ray intersection.
"""

from abc import ABC, abstractmethod

import numpy as np
import open3d as o3d

from lit.lidar import Lidar


class RaycastEngine(ABC):
    """
    Abstract class for raycast engine.

    Notes:
        - Currently we assume a scene is only used for raycasting once.
          To support multiple raycastings per scene, APIs need to be changed
          to store the raycasting scene in the class.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def rays_intersect_mesh(
        self,
        rays: np.ndarray,
        mesh: o3d.geometry.TriangleMesh,
    ):
        """
        Intersect the mesh with the given rays.

        Args:
            rays: (N, 6) float32 numpy array
            mesh: o3d.geometry.TriangleMesh

        Returns:
            points: (N, 3) float32 numpy array
        """

    @abstractmethod
    def lidar_intersect_mesh(
        self,
        lidar: Lidar,
        mesh: o3d.geometry.TriangleMesh,
    ):
        """
        Intersect the mesh with the lidar rays.

        Args:
            lidar: Lidar
            mesh: o3d.geometry.TriangleMesh

        Returns:
            points: (N, 3) float32 numpy array
        """

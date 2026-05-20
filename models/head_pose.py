import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from common.landmarks import LANDMARKS
from utils import camera_matrix, dist_coefficients


def estimate_head_pose(face) -> None:
    """
    Estimate the 3D head pose from 2D facial landmarks using the Perspective-n-Point (PnP) algorithm.
    
    This function computes the rotation and translation vectors that describe the head's orientation
    and position relative to the camera. The results are stored in the face object.
    
    Args:
        face: Face object containing 2D landmarks
        
    Returns:
        None - results are stored in face.head_pose_rot and face.head_position
    """
    rvec = np.zeros(3, dtype=np.float32)
    tvec = np.array([0, 0, 1], dtype=np.float32)
    _, rvec, tvec = cv2.solvePnP(
        LANDMARKS,
        face.landmarks,
        camera_matrix,
        dist_coefficients,
        rvec,
        tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    rot = Rotation.from_rotvec(rvec)
    face.head_pose_rot = rot
    face.head_position = tvec

import os
from typing import Optional

import cv2
import numpy as np
import timm
import torch
from scipy.spatial.transform import Rotation

# Camera intrinsic matrix (3x3)
# Format: [fx, 0, cx; 0, fy, cy; 0, 0, 1]
# Where fx, fy are focal lengths, cx, cy are principal point coordinates
# Suitable for original image size 640x480
camera_matrix = np.array([640., 0., 320.,
                          0., 640., 240.,
                          0., 0., 1.]).reshape(3, 3)

# Normalized camera intrinsic matrix (3x3)
# Used for coordinate transformation after image normalization to unified size
# Suitable for normalized image size 224x224
normalized_camera_matrix = np.array([960., 0., 112.,
                                     0., 960., 112.,
                                     0., 0., 1.]).reshape(3, 3)

# Camera distortion coefficients (5x1)
# Format: [k1, k2, p1, p2, k3]
# Currently set to zero vector, indicating no lens distortion
dist_coefficients = np.array([0., 0., 0., 0., 0.]).reshape(-1, 1)


def project_points(
        points3d: np.ndarray,
        rvec: Optional[np.ndarray] = None,
        tvec: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Project 3D points to 2D image plane using camera parameters.
    
    Args:
        points3d: Array of 3D points with shape (N, 3)
        rvec: Rotation vector (optional, defaults to zero vector)
        tvec: Translation vector (optional, defaults to zero vector)
        
    Returns:
        Array of 2D projected points with shape (N, 2)
    """
    assert points3d.shape[1] == 3
    if rvec is None:
        rvec = np.zeros(3, dtype=np.float32)
    if tvec is None:
        tvec = np.zeros(3, dtype=np.float32)
    points2d, _ = cv2.projectPoints(points3d, rvec, tvec,
                                    camera_matrix,
                                    dist_coefficients)
    return points2d.reshape(-1, 2)


def render(img, face, draw_face_bbox=True, draw_face_landmarks=True, draw_head_pose=True,
           draw_gaze_vector=True,
           color=(0, 255, 0)):
    """
    Render visualization of face detection and gaze estimation results.
    
    Args:
        img: Input image to draw on
        face: Face object containing detection results
        draw_face_bbox: Whether to draw face bounding box
        draw_face_landmarks: Whether to draw facial landmarks
        draw_head_pose: Whether to draw head pose axes
        draw_gaze_vector: Whether to draw gaze direction vector
        color: Primary color for drawing (default: green)
        
    Returns:
        Image with rendered annotations
    """
    size = 1

    if draw_face_bbox:
        bbox = np.round(face.bbox).astype(int).tolist()
        cv2.rectangle(img, tuple(bbox[0]), tuple(bbox[1]), color, size)

    if draw_face_landmarks:
        for pt in face.landmarks:
            pt = tuple(np.round(pt).astype(int).tolist())
            cv2.circle(img, pt, size, color, cv2.FILLED)

    if draw_head_pose:
        axes3d = np.eye(3, dtype=float) @ Rotation.from_euler('XYZ', [0, np.pi, 0]).as_matrix()
        axes3d = axes3d * 0.05
        axes2d = project_points(axes3d, face.head_pose_rot.as_rotvec(), face.head_position)
        center = face.landmarks[1]
        center = tuple(np.round(center).astype(int).tolist())
        for pt, axis_color in zip(axes2d, [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
            pt = tuple(np.round(pt).astype(int).tolist())
            cv2.line(img, center, pt, axis_color, 2, cv2.LINE_AA)

    if draw_gaze_vector:
        start = face.center
        end = face.center + 0.05 * face.gaze_vector
        points3d = np.vstack([start, end])
        points2d = project_points(points3d)
        pt0 = tuple(np.round(points2d[0]).astype(int).tolist())
        pt1 = tuple(np.round(points2d[1]).astype(int).tolist())
        cv2.line(img, pt0, pt1, color, 1, cv2.LINE_AA)

    return img


def load_model(device="cpu"):
    """
    Load the pre-trained gaze estimation model.
    
    Args:
        device: Target device to load the model onto (default: "cpu")
        
    Returns:
        Loaded and initialized gaze estimation model in evaluation mode
    """
    model = timm.create_model("resnet18", num_classes=2)
    checkpoint = torch.load(os.path.join(os.getcwd(), "../checkpoints", "resnet18.pth"), weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    return model

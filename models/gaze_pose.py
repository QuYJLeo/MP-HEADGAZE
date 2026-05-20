import cv2
import numpy as np
import torch
import torchvision.transforms as T
from scipy.spatial.transform import Rotation

from common.landmarks import LANDMARKS
from utils import load_model, normalized_camera_matrix, camera_matrix


def compute_3d_pose(face) -> None:
    """
    Compute the 3D positions of facial landmarks by transforming the 3D model
    using the estimated head pose rotation and translation.
    
    Args:
        face: Face object containing head_pose_rot and head_position
    """
    rot = face.head_pose_rot.as_matrix()
    face.model3d = LANDMARKS @ rot.T + face.head_position


def compute_face_eye_centers(face) -> None:
    """
    Compute the 3D center point of the face by averaging specific landmark positions.
    
    Uses eye corners (33, 133, 362, 263) and nose tip (240, 460) landmarks.
    
    Args:
        face: Face object containing model3d landmarks
    """
    face.center = face.model3d[
        np.concatenate([
            np.array([33, 133]),
            np.array([362, 263]),
            np.array([240, 460])]
        )].mean(axis=0)


def compute_normalizing_rotation(center, head_rot: Rotation) -> Rotation:
    """
    Compute the rotation matrix to normalize the face orientation.
    
    This creates a coordinate system where:
    - Z-axis points from camera to face center
    - Y-axis is perpendicular to both Z-axis and head's X-axis
    - X-axis completes the orthogonal basis
    
    Args:
        center: 3D center point of the face
        head_rot: Current head rotation
        
    Returns:
        Rotation object representing the normalizing rotation
    """
    def _normalize_vector(vector):
        return vector / np.linalg.norm(vector)

    z_axis = _normalize_vector(center.ravel())
    head_rot = head_rot.as_matrix()
    head_x_axis = head_rot[:, 0]
    y_axis = _normalize_vector(np.cross(z_axis, head_x_axis))
    x_axis = _normalize_vector(np.cross(y_axis, z_axis))
    return Rotation.from_matrix(np.vstack([x_axis, y_axis, z_axis]))


def get_scale_matrix(distance: float):
    """
    Create a scaling matrix to normalize face size based on distance.
    
    Args:
        distance: Distance from camera to face center
        
    Returns:
        3x3 scaling matrix
    """
    return np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0.6 / distance],
    ], dtype=np.float32)


data_transform = T.Compose([
    T.Lambda(lambda x: cv2.resize(x, (224, 224))),
    T.Lambda(lambda x: x[:, :, ::-1].copy()),  # BGR -> RGB
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # RGB
])


gaze_model = load_model()


def estimate_gaze_pose(image, face):
    """
    Estimate the gaze direction from the input image and face information.
    
    This function:
    1. Computes 3D landmarks from the head pose
    2. Calculates the face center point
    3. Computes normalizing rotation for face orientation
    4. Warps the image to a normalized perspective
    5. Runs the gaze estimation model on the normalized image
    6. Converts the predicted angles to a 3D gaze vector
    
    Args:
        image: Input RGB image
        face: Face object with head pose already estimated
    """
    compute_3d_pose(face)
    compute_face_eye_centers(face)
    face.normalizing_rot = compute_normalizing_rotation(face.center, face.head_pose_rot)

    scale = get_scale_matrix(face.distance)
    conversion_matrix = scale @ face.normalizing_rot.as_matrix()
    projection_matrix = normalized_camera_matrix @ conversion_matrix @ np.linalg.inv(camera_matrix)
    face.normalized_image = cv2.warpPerspective(
        image, projection_matrix,
        (224, 224))

    image = data_transform(face.normalized_image).unsqueeze(0)
    image = image.to("cpu")
    with torch.no_grad():
        prediction = gaze_model(image)
    prediction = prediction.cpu().numpy()
    face.normalized_gaze_angles = prediction[0]
    face.angle_to_vector()
    face.denormalize_gaze_vector()



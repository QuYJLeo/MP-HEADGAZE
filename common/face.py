import numpy as np


class Face:
    """Represents a detected face with its landmarks and pose information."""

    def __init__(self, bbox, landmarks):
        """
        Initialize a Face object.
        
        Args:
            bbox: Bounding box coordinates of the face
            landmarks: 2D facial landmarks detected in the image
        """
        self.bbox = bbox
        self.landmarks = landmarks

        self.head_position = None  # [x, y, z] - 3D position of the head
        self.head_pose_rot = None  # Rotation object representing head orientation
        self.model3d = None  # 3D model landmarks transformed to camera space
        self.normalizing_rot = None  # Rotation to normalize face orientation
        self.normalized_image = None  # Normalized face image for gaze estimation
        self.normalized_head_rot2d = None  # Normalized head rotation in 2D

        self.center = None  # 3D center point between eyes and nose

        self.normalized_gaze_angles = None  # Gaze angles in normalized coordinate system
        self.normalized_gaze_vector = None  # Gaze vector in normalized coordinate system
        self.gaze_vector = None  # Final gaze vector in camera coordinate system

    @property
    def distance(self) -> float:
        """Calculate the Euclidean distance from the face center to the camera."""
        return np.linalg.norm(self.center)

    def angle_to_vector(self) -> None:
        """Convert normalized gaze angles (pitch, yaw) to a 3D gaze vector."""
        pitch, yaw = self.normalized_gaze_angles
        self.normalized_gaze_vector = -np.array([
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch),
            np.cos(pitch) * np.cos(yaw)
        ])

    def denormalize_gaze_vector(self) -> None:
        """Transform the normalized gaze vector back to camera coordinate system."""
        normalizing_rot = self.normalizing_rot.as_matrix()
        self.gaze_vector = self.normalized_gaze_vector @ normalizing_rot

    @staticmethod
    def vector_to_angle(vector: np.ndarray) -> np.ndarray:
        """
        Convert a 3D gaze vector to pitch and yaw angles.
        
        Args:
            vector: 3D gaze vector with shape (3,)
            
        Returns:
            Array containing pitch and yaw angles in radians
        """
        assert vector.shape == (3,)
        x, y, z = vector
        pitch = np.arcsin(-y)
        yaw = np.arctan2(-x, -z)
        return np.array([pitch, yaw])

    def get_head_angles(self):
        """
        Get the head pose angles (pitch, yaw, roll) in degrees.
        
        Returns:
            Tuple of (pitch, yaw, roll) angles in degrees
        """
        euler_angles = self.head_pose_rot.as_euler('XYZ', degrees=True)
        pitch, yaw, roll = self.change_coordinate_system(euler_angles)
        return pitch, yaw, roll

    @staticmethod
    def change_coordinate_system(euler_angles: np.ndarray) -> np.ndarray:
        """
        Transform Euler angles to the desired coordinate system.
        
        Args:
            euler_angles: Array of Euler angles
            
        Returns:
            Transformed Euler angles
        """
        return euler_angles * np.array([-1, 1, -1])

    def get_gaze_angles(self):
        """
        Get the gaze angles (pitch, yaw) in degrees.
        
        Returns:
            Tuple of (pitch, yaw) gaze angles in degrees
        """
        pitch, yaw = np.rad2deg(self.vector_to_angle(self.gaze_vector))
        return pitch, yaw

    def __repr__(self):
        """Return an empty string representation."""
        return ''

import cv2
import mediapipe
import numpy as np

from common.face import Face
from models.gaze_pose import estimate_gaze_pose
from models.head_pose import estimate_head_pose
from models.utils import render

# Initialize MediaPipe FaceMesh for face detection
# max_num_faces=1: detect only one face at a time
# static_image_mode=False: optimized for video stream
face_mesh = mediapipe.solutions.face_mesh.FaceMesh(max_num_faces=1, static_image_mode=False)


def detect_faces(image) -> list[Face]:
    """
    Detect faces in an image using MediaPipe FaceMesh.
    
    Args:
        image: Input RGB image
        
    Returns:
        List of detected Face objects containing bounding boxes and landmarks
    """
    h, w = image.shape[:2]
    predictions = face_mesh.process(image[:, :, ::-1])
    detected = []
    if predictions.multi_face_landmarks:
        for prediction in predictions.multi_face_landmarks:
            landmarks = np.array([(pt.x * w, pt.y * h) for pt in prediction.landmark], dtype=np.float64)
            bbox = np.vstack([landmarks.min(axis=0), landmarks.max(axis=0)])
            bbox = np.round(bbox).astype(np.int32)
            detected.append(Face(bbox, landmarks))
    return detected


if __name__ == '__main__':
    """
    Main entry point for real-time gaze estimation application.
    
    This script:
    1. Captures video from the default camera
    2. Detects faces in each frame
    3. Estimates head pose for each detected face
    4. Estimates gaze direction using a pre-trained model
    5. Visualizes results and prints pose information
    6. Exits when 'q' key is pressed
    """
    v = cv2.VideoCapture(0)

    while True:
        ret, frame = v.read()
        if ret:
            faces = detect_faces(frame)
            for face in faces:
                estimate_head_pose(face)
                estimate_gaze_pose(frame, face)

                pitch, yaw, roll = face.get_head_angles()
                print(f"Face angles: pitch={pitch}, yaw={yaw}, roll={roll}.")
                g_pitch, g_yaw = face.get_gaze_angles()
                print(f"Gaze angles: pitch={g_pitch}, yaw={g_yaw}")
                print(f"Gaze vector: {face.gaze_vector}")
                print(f"Distance to camera: {face.distance}")

            for face in faces:
                render(
                    frame,
                    face,
                    draw_face_bbox=True,
                    draw_face_landmarks=True,
                    draw_head_pose=True,
                    draw_gaze_vector=True,
                    color=(0, 255, 0)
                )
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    v.release()
    cv2.destroyAllWindows()

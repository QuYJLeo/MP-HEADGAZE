# Real-Time Gaze Estimation System

A real-time gaze estimation system that combines head pose estimation with eye gaze direction prediction using deep learning.

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Features](#features)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Project Structure](#project-structure)
8. [Methodology](#methodology)
    - [Face Detection](#face-detection)
    - [Head Pose Estimation](#head-pose-estimation)
    - [Gaze Estimation](#gaze-estimation)
    - [Visualization](#visualization)
9. [Configuration](#configuration)
10. [Model Details](#model-details)
11. [License](#license)

## Project Overview

This project implements a real-time gaze estimation system that can detect a person's gaze direction from a video stream. The system combines:
- **Face Detection**: Using MediaPipe FaceMesh to detect facial landmarks
- **Head Pose Estimation**: Using Perspective-n-Point (PnP) algorithm to estimate 3D head orientation
- **Gaze Estimation**: Using a pre-trained ResNet18 model to predict gaze angles from normalized facial images

The system outputs real-time head pose angles (pitch, yaw, roll), gaze angles (pitch, yaw), and gaze vector in 3D space.

## System Architecture

```
Video Input → Face Detection → Head Pose Estimation → Gaze Estimation → Visualization
                   ↓                  ↓                    ↓
              Landmarks          3D Pose              Normalized Image
                                          ↓
                                   Gaze Vector
```

## Features

- **Real-Time Processing**: Processes video frames at high frame rates
- **Accurate Face Detection**: Uses MediaPipe FaceMesh for robust facial landmark detection
- **3D Head Pose Estimation**: Computes precise head orientation using PnP algorithm
- **Deep Learning-based Gaze Estimation**: Uses ResNet18 model trained on gaze datasets
- **Visualization**: Draws bounding boxes, landmarks, head pose axes, and gaze vectors
- **Distance Calculation**: Computes distance from face to camera

## Requirements

- Python 3.8+
- OpenCV (cv2)
- MediaPipe
- NumPy
- PyTorch
- torchvision
- scipy
- timm

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd MP-HEAD_GAZE
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the pre-trained model:
   - Place `resnet18.pth` in the `checkpoints/` directory
   - The model should be trained on a gaze estimation dataset

## Usage

### Running the Real-Time Demo

```bash
python main.py
```

### Key Operations

- **Quit**: Press 'q' key to exit the application
- **Output**: The system prints the following information to console:
  - Face angles (pitch, yaw, roll) in degrees
  - Gaze angles (pitch, yaw) in degrees
  - Gaze vector in 3D space
  - Distance to camera

### Program Flow

1. **Video Capture**: Opens the default camera (index 0)
2. **Face Detection**: Detects faces in each frame using MediaPipe
3. **Head Pose Estimation**: Estimates head orientation using PnP algorithm
4. **Gaze Estimation**: 
   - Normalizes the face image based on head pose
   - Runs the ResNet18 model to predict gaze angles
   - Converts angles to 3D gaze vector
5. **Visualization**: Draws results on the frame
6. **Display**: Shows the annotated frame in a window

## Project Structure

```
MP-HEAD_GAZE/
├── main.py                 # Main entry point
├── requirements.txt        # Dependencies
├── LICENSE                 # License file
├── checkpoints/            # Pre-trained models
│   └── resnet18.pth        # Gaze estimation model
├── common/                 # Common utilities
│   ├── face.py             # Face class definition
│   └── landmarks.py        # 3D facial landmark coordinates
└── models/                 # Model implementations
    ├── gaze_pose.py        # Gaze estimation pipeline
    ├── head_pose.py        # Head pose estimation
    └── utils.py            # Utility functions
```

### File Descriptions

#### main.py
Main script that orchestrates the entire gaze estimation pipeline. Handles video capture, face detection, pose estimation, and visualization.

#### common/face.py
Defines the `Face` class that encapsulates all face-related data including:
- Bounding box
- 2D landmarks
- 3D head pose (rotation and position)
- Gaze vector and angles
- Normalized image for gaze estimation

#### common/landmarks.py
Contains 3D coordinates of facial landmarks used for head pose estimation. These are standard 478 landmarks from MediaPipe FaceMesh.

#### models/head_pose.py
Implements head pose estimation using OpenCV's solvePnP function. Estimates rotation and translation vectors from 2D landmarks.

#### models/gaze_pose.py
Implements the gaze estimation pipeline:
- Computes 3D face model from head pose
- Normalizes face orientation for consistent input
- Runs ResNet18 model to predict gaze angles
- Converts angles to 3D gaze vector

#### models/utils.py
Provides utility functions:
- Camera matrix definitions
- 3D to 2D point projection
- Visualization rendering
- Model loading

## Methodology

### Face Detection

The system uses MediaPipe FaceMesh, a lightweight and efficient face detection solution that provides:
- 478 2D facial landmarks
- Real-time performance
- Robust detection under varying lighting conditions

### Head Pose Estimation

The head pose estimation uses the Perspective-n-Point (PnP) algorithm:

1. **3D Model**: Uses predefined 3D landmark coordinates
2. **2D Observations**: Detected landmarks from the image
3. **Camera Parameters**: Intrinsic matrix and distortion coefficients
4. **Algorithm**: `cv2.SOLVEPNP_ITERATIVE` for accurate pose estimation

The output is a rotation vector and translation vector describing the head's orientation and position relative to the camera.

### Gaze Estimation

The gaze estimation pipeline consists of several steps:

1. **3D Face Model Construction**: Transform 3D landmarks using estimated head pose
2. **Face Normalization**: 
   - Compute face center from eye and nose landmarks
   - Create a normalizing rotation to align face with camera
   - Apply perspective transformation to create normalized face image
3. **Model Inference**:
   - Resize image to 224x224
   - Convert BGR to RGB
   - Apply ImageNet normalization
   - Run ResNet18 model to predict gaze angles (pitch, yaw)
4. **Vector Conversion**: Convert normalized gaze angles to 3D vector
5. **Denormalization**: Transform gaze vector back to camera coordinate system

### Visualization

The system provides comprehensive visualization:

| Element | Description | Color |
|---------|-------------|-------|
| Bounding Box | Face detection boundary | Green |
| Landmarks | 478 facial points | Green |
| Head Pose Axes | X, Y, Z axes from nose | Red, Green, Blue |
| Gaze Vector | Direction of gaze | Green |

## Configuration

### Camera Parameters

The system uses fixed camera intrinsic parameters in `models/utils.py`:

```python
# Camera matrix (640x480 resolution)
camera_matrix = np.array([640., 0., 320., 0., 640., 240., 0., 0., 1.]).reshape(3, 3)

# Normalized camera matrix (224x224 resolution)
normalized_camera_matrix = np.array([960., 0., 112., 0., 960., 112., 0., 0., 1.]).reshape(3, 3)

# Distortion coefficients (no distortion)
dist_coefficients = np.array([0., 0., 0., 0., 0.]).reshape(-1, 1)
```

### Visualization Options

The `render` function in `models/utils.py` supports:
- `draw_face_bbox`: Draw face bounding box
- `draw_face_landmarks`: Draw facial landmarks
- `draw_head_pose`: Draw head pose axes
- `draw_gaze_vector`: Draw gaze direction vector
- `color`: Primary drawing color

## Model Details

### Architecture

The gaze estimation model uses ResNet18 with the following modifications:
- Input: 224x224 RGB image
- Backbone: ResNet18
- Output: 2-dimensional vector (pitch, yaw angles in radians)

### Training

The model should be trained on a gaze estimation dataset with:
- Normalized face images
- Ground truth gaze angles
- Appropriate data augmentation

### Pre-trained Model

The pre-trained model should be placed at `checkpoints/resnet18.pth` with the following structure:
```python
checkpoint = {
    'model': state_dict,  # Model weights
    # Additional metadata as needed
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
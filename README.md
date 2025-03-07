# Face Detection and Recognition System

A comprehensive Python-based system for face detection, recognition, and management using OpenCV.

## Features

- **Face Detection**: Detects faces in images and video streams using either DNN (more accurate) or Haar Cascade methods.
- **Face Recognition**: Identifies known faces using LBPH (Local Binary Patterns Histograms) recognition.
- **Dataset Creation**: Easily create face datasets for new people using a webcam.
- **Model Training**: Train the recognition model with collected face datasets.
- **Real-time Recognition**: Perform face detection and recognition in real-time using a webcam.
- **People Management**: Add, list, and delete people from the face database.
- **Image Processing**: Process individual images for face detection and recognition.

## System Requirements

- Python 3.6 or higher
- OpenCV 4.x with contrib modules (for face recognition)
- Webcam (for dataset creation and real-time recognition)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yohanzytoon/face-detection-system.git
cd face-detection-system
```

### 2. Create a virtual environment (recommended)

```bash
# Using venv
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install required packages

```bash
pip install -r requirements.txt
```


### 4. Download required model files

Run the model downloader script to get the necessary files:

```bash
python model_downloader.py
```

This will download:
- DNN face detection model files
- Haar Cascade classifier file


## Usage

### 1. Run the application

```bash
python face_detection_project.py
```

### 2. Main Menu Options

When you run the application, you'll see the following menu:

1. **Create Face Dataset**: Capture face images for a new or existing person
2. **Train Recognition Model**: Train the face recognition model with captured datasets
3. **Process Image**: Detect and recognize faces in a specific image file
4. **Real-time Recognition**: Start real-time webcam face detection and recognition
5. **Manage People**: List or delete people in the database
6. **Exit**: Quit the application

### 3. Typical Workflow

1. First, download the required models using `python model_downloader.py`
2. Run the main program: `python face_detection_project.py`
3. Add a person to the database (Option 1)
4. Train the recognition model (Option 2)
5. Test with real-time recognition (Option 4) or process individual images (Option 3)

## Important Notes

- **OpenCV Requirement**: This system requires OpenCV with the `face` module. If you see warnings about missing face modules, follow the installation instructions above.
- **Camera Access**: Make sure your webcam is connected and accessible. The default camera (index 0) is used for capturing.
- **Face Recognition Quality**: For best recognition results:
  - Capture face images in good lighting conditions
  - Capture multiple angles (front, slight left/right)
  - Aim for at least 20-30 samples per person
  - Ensure the face is properly centered in the frame during capture
- **Storage**: The system will create necessary directories for storing face data, models, and results.

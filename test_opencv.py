
import cv2
import sys

print(f"OpenCV version: {cv2.__version__}")

# Check face detector
try:
    face_cascade = cv2.CascadeClassifier()
    print("Face detection available!")
except Exception as e:
    print(f"Face detection error: {e}")

# Try different ways to access face module
try:
    # Try OpenCV 3+ API
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    print("Face recognition available (cv2.face)!")
except AttributeError:
    try:
        # Try different spelling/format
        recognizer = cv2.face_LBPHFaceRecognizer.create()
        print("Face recognition available (cv2.face_LBPHFaceRecognizer)!")
    except Exception as e:
        print(f"Face recognition not available: {e}")


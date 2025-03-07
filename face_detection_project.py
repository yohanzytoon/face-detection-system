import os
import time
import cv2
import numpy as np
import json
from datetime import datetime
import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("FaceSystem")

# Create needed directories
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("dataset", exist_ok=True)
os.makedirs("results", exist_ok=True)

class FaceDatabase:
    """Simple database for storing person information"""
    
    def __init__(self, db_path="data/face_database.json"):
        self.db_path = db_path
        self.data = {"people": {}, "last_id": 0}
        
        # Load database if it exists
        if os.path.exists(db_path):
            try:
                with open(db_path, "r") as f:
                    self.data = json.load(f)
                print(f"Loaded database with {len(self.data['people'])} people")
            except Exception as e:
                print(f"Error loading database: {e}")
    
    def save(self):
        """Save database to disk"""
        try:
            with open(self.db_path, "w") as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"Error saving database: {e}")
    
    def add_person(self, name):
        """Add a new person to the database"""
        person_id = self.data["last_id"] + 1
        self.data["last_id"] = person_id
        
        self.data["people"][str(person_id)] = {
            "id": person_id,
            "name": name,
            "created_at": datetime.now().isoformat(),
            "face_count": 0
        }
        
        self.save()
        return person_id
    
    def get_person(self, person_id):
        """Get person by ID"""
        person_id_str = str(person_id)
        return self.data["people"].get(person_id_str)
    
    def get_person_by_name(self, name):
        """Get person by name"""
        for person in self.data["people"].values():
            if person["name"].lower() == name.lower():
                return person
        return None
    
    def update_face_count(self, person_id, count=1):
        """Increment face count for person"""
        person_id_str = str(person_id)
        if person_id_str in self.data["people"]:
            self.data["people"][person_id_str]["face_count"] += count
            self.save()
    
    def delete_person(self, person_id):
        """Delete person from database"""
        person_id_str = str(person_id)
        if person_id_str in self.data["people"]:
            del self.data["people"][person_id_str]
            self.save()
            return True
        return False
    
    def get_all_people(self):
        """Get all people in the database"""
        return list(self.data["people"].values())
    
    def id_to_name(self, person_id):
        """Convert person ID to name"""
        person = self.get_person(person_id)
        return person["name"] if person else "Unknown"


class FaceSystem:
    """Main face detection and recognition system"""
    
    def __init__(self):
        # Initialize database
        self.database = FaceDatabase()
        
        # Initialize face detection
        self.init_face_detector()
        
        # Initialize face recognition
        self.init_face_recognizer()
        
        # Camera handle
        self.camera = None
    
    def init_face_detector(self):
        """Initialize face detector"""
        # Try to use DNN detector (more accurate)
        self.use_dnn = False
        try:
            proto_path = "models/deploy.prototxt"
            model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"
            
            # Check if model files exist
            if os.path.exists(proto_path) and os.path.exists(model_path):
                self.face_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
                self.use_dnn = True
                print("Using DNN face detector (more accurate)")
            else:
                print(f"DNN model files not found. Place them in the models directory:")
                print(f"  - {proto_path}")
                print(f"  - {model_path}")
        except Exception as e:
            print(f"Could not initialize DNN detector: {e}")
        
        # Always initialize Haar Cascade as fallback
        try:
            cascade_path = "models/haarcascade_frontalface_default.xml"
            
            # Check if cascade file exists, try to find it in OpenCV if not
            if not os.path.exists(cascade_path):
                opencv_cascade_paths = []
                for path in cv2.__path__:
                    candidate = os.path.join(path, 'data', 'haarcascade_frontalface_default.xml')
                    if os.path.exists(candidate):
                        opencv_cascade_paths.append(candidate)
                
                if opencv_cascade_paths:
                    # Use the first found cascade file
                    import shutil
                    os.makedirs("models", exist_ok=True)
                    shutil.copy(opencv_cascade_paths[0], cascade_path)
                    print(f"Copied OpenCV's cascade file to {cascade_path}")
            
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                print("Using Haar Cascade face detector (faster but less accurate)")
            else:
                print("Haar cascade file not found. Face detection may not work.")
                self.face_cascade = None
        except Exception as e:
            print(f"Could not initialize Haar Cascade detector: {e}")
            self.face_cascade = None
    
    def init_face_recognizer(self):
        """Initialize face recognizer"""
        try:
            # Try LBPH recognizer (works best for frontal faces)
            self.recognizer = cv2.face_LBPHFaceRecognizer.create()
        except AttributeError:
            try:
                # Fallback to face module in cv2
                self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            except Exception as e:
                print(f"Error creating face recognizer: {e}")
                self.recognizer = None
        
        # Set confidence threshold (lower values are stricter)
        self.confidence_threshold = 70.0
        
        # Load recognizer model if available
        self.model_path = "models/face_recognizer.yml"
        if os.path.exists(self.model_path) and self.recognizer:
            try:
                self.recognizer.read(self.model_path)
                print("Loaded face recognition model")
            except Exception as e:
                print(f"Error loading recognition model: {e}")
    
    def detect_faces(self, image):
        """Detect faces in an image using the available detectors"""
        if image is None:
            return []
        
        faces = []
        
        # Try DNN detection first (if available)
        if self.use_dnn:
            try:
                (h, w) = image.shape[:2]
                blob = cv2.dnn.blobFromImage(
                    cv2.resize(image, (300, 300)), 
                    1.0, 
                    (300, 300), 
                    (104.0, 177.0, 123.0)
                )
                
                self.face_net.setInput(blob)
                detections = self.face_net.forward()
                
                for i in range(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    
                    # Filter weak detections
                    if confidence > 0.5:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        
                        # Ensure bounding boxes are within image
                        startX = max(0, startX)
                        startY = max(0, startY)
                        endX = min(w - 1, endX)
                        endY = min(h - 1, endY)
                        
                        # Add face coordinates
                        faces.append((startX, startY, endX - startX, endY - startY))
            except Exception as e:
                print(f"DNN detection error: {e}")
        
        # If no faces found or DNN not available, try Haar Cascade
        if not faces and self.face_cascade:
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                haar_faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.3,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                if len(haar_faces) > 0:
                    faces = [(x, y, w, h) for (x, y, w, h) in haar_faces]
            except Exception as e:
                print(f"Haar Cascade detection error: {e}")
        
        return faces
    
    def preprocess_face(self, image):
        """Preprocess a face image for better recognition"""
        if len(image.shape) == 3:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply histogram equalization to improve contrast
        gray = cv2.equalizeHist(gray)
        
        return gray
    
    def recognize_face(self, face_img):
        """Recognize a face using the trained model"""
        if self.recognizer is None:
            return -1, 0.0
        
        try:
            # Preprocess face
            processed_face = self.preprocess_face(face_img)
            
            # Perform recognition
            label, confidence = self.recognizer.predict(processed_face)
            
            # Only return if confidence is good enough
            if confidence < self.confidence_threshold:
                return label, confidence
            else:
                return -1, confidence
        except Exception as e:
            print(f"Error in face recognition: {e}")
            return -1, 0.0
    
    def create_dataset(self, person_name, camera_source=0, samples=30):
        """Create a face dataset for a person"""
        # Check if person already exists
        person = self.database.get_person_by_name(person_name)
        
        if person:
            person_id = person["id"]
            print(f"Person {person_name} already exists with ID {person_id}")
        else:
            # Add person to database
            person_id = self.database.add_person(person_name)
            print(f"Added {person_name} to database with ID {person_id}")
        
        # Create directory for person's face images
        person_dir = os.path.join("dataset", str(person_id))
        os.makedirs(person_dir, exist_ok=True)
        
        # Initialize camera
        self.camera = cv2.VideoCapture(camera_source)
        if not self.camera.isOpened():
            print(f"Could not open camera {camera_source}")
            return False
        
        try:
            count = 0
            last_capture_time = time.time() - 1.0  # Allow immediate first capture
            
            print(f"Capturing {samples} face images for {person_name}...")
            print("Position your face in front of the camera and move slightly between captures.")
            print("Press ESC to cancel.")
            
            while count < samples:
                ret, frame = self.camera.read()
                if not ret:
                    continue
                
                # Detect faces
                display_frame = frame.copy()
                faces = self.detect_faces(frame)
                
                # Draw rectangles around all detected faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Process one face if detected
                if len(faces) == 1:
                    (x, y, w, h) = faces[0]
                    
                    # Capture face every 0.5 seconds
                    current_time = time.time()
                    if current_time - last_capture_time >= 0.5:
                        # Extract face
                        face_img = frame[y:y+h, x:x+w]
                        
                        # Resize to standard size
                        face_img = cv2.resize(face_img, (200, 200))
                        
                        # Save face image
                        face_path = os.path.join(person_dir, f"{person_id}_{count}.jpg")
                        cv2.imwrite(face_path, face_img)
                        
                        # Update counter and timestamp
                        count += 1
                        last_capture_time = current_time
                        
                        # Update database
                        self.database.update_face_count(person_id)
                        
                        # Display progress
                        print(f"Captured image {count}/{samples}")
                
                # Show status
                status_text = f"Captured: {count}/{samples}"
                cv2.putText(display_frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show instructions
                cv2.putText(display_frame, "Position face in frame", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show the frame
                cv2.imshow("Face Capture", display_frame)
                
                # Check for ESC key
                key = cv2.waitKey(1)
                if key == 27:  # ESC key
                    print("Face capture cancelled")
                    break
            
            print(f"Dataset creation completed for {person_name}")
            return count > 0
            
        except Exception as e:
            print(f"Error creating dataset: {e}")
            return False
            
        finally:
            # Clean up
            if self.camera:
                self.camera.release()
                self.camera = None
            cv2.destroyAllWindows()
    
    def train_model(self):
        """Train face recognition model using the dataset"""
        faces = []
        labels = []
        
        print("Training face recognition model...")
        
        try:
            # Find all person folders in dataset directory
            dataset_path = "dataset"
            for person_dir in os.listdir(dataset_path):
                try:
                    person_id = int(person_dir)
                    person = self.database.get_person(person_id)
                    
                    if not person:
                        print(f"Warning: Person ID {person_id} not found in database")
                        continue
                    
                    person_path = os.path.join(dataset_path, person_dir)
                    if not os.path.isdir(person_path):
                        continue
                    
                    print(f"Processing images for {person['name']} (ID: {person_id})")
                    
                    # Load all face images for this person
                    image_count = 0
                    for img_file in os.listdir(person_path):
                        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                            continue
                        
                        img_path = os.path.join(person_path, img_file)
                        img = cv2.imread(img_path)
                        
                        if img is None:
                            print(f"Could not read image: {img_path}")
                            continue
                        
                        # Preprocess face image
                        gray = self.preprocess_face(img)
                        
                        # Add to training data
                        faces.append(gray)
                        labels.append(person_id)
                        image_count += 1
                    
                    print(f"Processed {image_count} images for {person['name']}")
                    
                except Exception as e:
                    print(f"Error processing person directory {person_dir}: {e}")
            
            # Train the model if we have face images
            if not faces:
                print("No face images found. Please create a dataset first.")
                return False
            
            print(f"Training with {len(faces)} face images across {len(set(labels))} people...")
            
            if self.recognizer:
                # Train the model
                self.recognizer.train(faces, np.array(labels))
                
                # Save the model
                self.recognizer.write(self.model_path)
                print("Training completed and model saved!")
                return True
            else:
                print("Face recognizer not available.")
                return False
                
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    
    def process_image(self, image_path, output_path=None):
        """Process an image for face detection and recognition"""
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            return None
        
        # Create a copy for drawing
        output_image = image.copy()
        
        # Detect faces
        faces = self.detect_faces(image)
        print(f"Detected {len(faces)} faces")
        
        # Process each face
        face_data = []
        for i, (x, y, w, h) in enumerate(faces):
            face_info = {
                "id": i,
                "location": {"x": x, "y": y, "width": w, "height": h},
                "recognition": {"id": -1, "name": "Unknown", "confidence": 0.0}
            }
            
            # Skip tiny faces
            if w < 30 or h < 30:
                continue
            
            # Extract face
            face_img = image[y:y+h, x:x+w]
            
            # Recognize face
            person_id, confidence = self.recognize_face(face_img)
            
            # Update recognition info
            if person_id >= 0:
                person_name = self.database.id_to_name(person_id)
                face_info["recognition"] = {
                    "id": person_id,
                    "name": person_name,
                    "confidence": confidence
                }
            
            face_data.append(face_info)
            
            # Draw on output image
            if person_id >= 0:
                # Recognized face - green
                color = (0, 255, 0)
                name = self.database.id_to_name(person_id)
                text = f"{name} ({confidence:.1f})"
            else:
                # Unknown face - red
                color = (0, 0, 255)
                text = "Unknown"
            
            # Draw rectangle and name
            cv2.rectangle(output_image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(output_image, text, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save output image if path provided
        if output_path and len(faces) > 0:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, output_image)
            print(f"Saved result to {output_path}")
        
        return output_image, face_data
    
    def real_time_recognition(self, camera_source=0):
        """Perform real-time face detection and recognition"""
        # Initialize camera
        self.camera = cv2.VideoCapture(camera_source)
        if not self.camera.isOpened():
            print(f"Could not open camera {camera_source}")
            return
        
        print("Starting real-time face recognition. Press 'q' to quit.")
        
        # FPS calculation
        fps_counter = 0
        fps_start_time = time.time()
        fps = 0
        
        try:
            while True:
                # Read frame
                ret, frame = self.camera.read()
                if not ret:
                    break
                
                # Create output frame
                output_frame = frame.copy()
                
                # Update FPS counter
                fps_counter += 1
                elapsed_time = time.time() - fps_start_time
                if elapsed_time >= 1.0:
                    fps = fps_counter / elapsed_time
                    fps_counter = 0
                    fps_start_time = time.time()
                
                # Detect faces
                faces = self.detect_faces(frame)
                
                # Process each face
                for (x, y, w, h) in faces:
                    # Skip tiny faces
                    if w < 30 or h < 30:
                        continue
                    
                    # Extract face
                    face_img = frame[y:y+h, x:x+w]
                    
                    # Recognize face
                    person_id, confidence = self.recognize_face(face_img)
                    
                    # Draw rectangle and name
                    if person_id >= 0:
                        # Recognized face - green
                        color = (0, 255, 0)
                        name = self.database.id_to_name(person_id)
                        text = f"{name} ({confidence:.1f})"
                    else:
                        # Unknown face - red
                        color = (0, 0, 255)
                        text = "Unknown"
                    
                    cv2.rectangle(output_frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(output_frame, text, (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Display FPS
                cv2.putText(output_frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Show frame
                cv2.imshow("Face Recognition", output_frame)
                
                # Check for quit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
        except Exception as e:
            print(f"Error in real-time recognition: {e}")
            
        finally:
            # Clean up
            if self.camera:
                self.camera.release()
                self.camera = None
            cv2.destroyAllWindows()
    
    def manage_people(self):
        """Manage people in the database"""
        while True:
            print("\nPeople Management:")
            print("1. List All People")
            print("2. Delete Person")
            print("3. Back to Main Menu")
            
            choice = input("\nEnter your choice (1-3): ")
            
            if choice == '1':
                # List all people
                people = self.database.get_all_people()
                
                if not people:
                    print("\nNo people in database")
                else:
                    print("\nPeople in Database:")
                    print("ID\tName\t\tFaces\tCreated")
                    print("-" * 50)
                    for person in people:
                        date = datetime.fromisoformat(person['created_at']).strftime("%Y-%m-%d")
                        print(f"{person['id']}\t{person['name']}\t\t{person['face_count']}\t{date}")
            
            elif choice == '2':
                # Delete person
                person_id = input("\nEnter person ID to delete: ")
                
                if not person_id.isdigit():
                    print("Invalid ID")
                    continue
                
                person_id = int(person_id)
                person = self.database.get_person(person_id)
                
                if not person:
                    print(f"Person ID {person_id} not found")
                    continue
                
                confirm = input(f"Are you sure you want to delete {person['name']} (y/n)? ")
                
                if confirm.lower() == 'y':
                    # Delete from database
                    self.database.delete_person(person_id)
                    
                    # Delete person's dataset directory
                    person_dir = os.path.join("dataset", str(person_id))
                    if os.path.exists(person_dir):
                        try:
                            import shutil
                            shutil.rmtree(person_dir)
                        except Exception as e:
                            print(f"Error deleting person directory: {e}")
                    
                    print(f"Person {person['name']} deleted")
            
            elif choice == '3':
                break
            
            else:
                print("Invalid choice")

# Main CLI interface
def run_face_system():
    """Run the face detection and recognition system"""
    print("\n===== Simple Face Detection System =====\n")
    
    # Initialize the system
    face_system = FaceSystem()
    
    try:
        while True:
            print("\nMain Menu:")
            print("1. Create Face Dataset")
            print("2. Train Recognition Model")
            print("3. Process Image")
            print("4. Real-time Recognition")
            print("5. Manage People")
            print("6. Exit")
            
            choice = input("\nEnter your choice (1-6): ")
            
            if choice == '1':
                # Create dataset
                person_name = input("Enter person's name: ")
                samples = input("Number of samples to capture (default 30): ")
                samples = int(samples) if samples.isdigit() else 30
                
                face_system.create_dataset(person_name, 0, samples)
                
            elif choice == '2':
                # Train model
                face_system.train_model()
                
            elif choice == '3':
                # Process image
                image_path = input("Enter path to image file: ")
                
                if not os.path.exists(image_path):
                    print(f"Error: File not found: {image_path}")
                    continue
                
                # Create output path
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join("results", f"result_{timestamp}.jpg")
                
                # Process the image
                result_image, face_data = face_system.process_image(image_path, output_path)
                
                if result_image is not None:
                    # Display the result image
                    cv2.imshow("Detection Result", result_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                
            elif choice == '4':
                # Real-time recognition
                face_system.real_time_recognition(camera_source=0)
                
            elif choice == '5':
                # Manage people
                face_system.manage_people()
                
            elif choice == '6':
                # Exit
                print("\nExiting...")
                break
                
            else:
                print("Invalid choice")
    
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    # Check OpenCV version
    print(f"Using OpenCV version: {cv2.__version__}")
    
    # Check for OpenCV face module
    face_module_available = False
    try:
        cv2.face_LBPHFaceRecognizer.create()
        face_module_available = True
    except AttributeError:
        try:
            cv2.face.LBPHFaceRecognizer_create()
            face_module_available = True
        except:
            pass
    
    if not face_module_available:
        print("Warning: OpenCV face module not available. Try installing opencv-contrib:")
        print("conda install -c conda-forge opencv-contrib-python")
    
    # Run the face detection system
    run_face_system()
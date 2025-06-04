import sys
import cv2
import numpy as np
import os
import pygame
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QFrame, QScrollArea)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont

# Import recognition system functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from recognition_system import (
    TFLiteModel, load_faiss_data, preprocess_image_blazeface,
    preprocess_image_mobilefacenet, postprocess_blazeface_output,
    recognize_face, BLAZEFACE_MODEL_PATH, MOBILEFACENET_MODEL_PATH,
    MIN_DETECTION_SCORE
)

class AttendanceUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Attendance System")
        
        # Initialize pygame for sound
        pygame.mixer.init()
        
        # Initialize detection state
        self.last_detection_state = False
        self.current_detected_name = "Loading system..."
        self.recognition_system_loaded = False
        self.frame_count = 0
        
        # Initialize model variables
        self.blazeface_model = None
        self.mobilefacenet_model = None
        self.faiss_index = None
        self.user_id_map = []
        
        # Setup UI first (so we can update status)
        self.setup_ui()
        
        # Load face recognition models and data
        self.load_recognition_system()
        
        self.setup_camera()

    def load_recognition_system(self):
        """Load face recognition models and FAISS data"""
        print("Starting face recognition system initialization...")
        self.title_label.setText("Loading recognition models...")
        self.title_label.setStyleSheet("color: orange; font-size: 24pt; font-weight: bold;")
        
        try:
            print("Loading TensorFlow Lite models...")
            
            # Load models as instance variables
            self.blazeface_model = TFLiteModel(BLAZEFACE_MODEL_PATH)
            print("BlazeFace model loaded successfully.")
            
            self.mobilefacenet_model = TFLiteModel(MOBILEFACENET_MODEL_PATH)
            print("MobileFaceNet model loaded successfully.")
            
            print("Loading FAISS index and user data...")
            self.faiss_index, self.user_id_map = load_faiss_data()
            
            if self.faiss_index and len(self.user_id_map) > 0:
                print(f"Face recognition system loaded successfully. {len(self.user_id_map)} users enrolled.")
                self.recognition_system_loaded = True
                self.title_label.setText("System ready - No face detected")
                self.title_label.setStyleSheet("color: blue; font-size: 24pt; font-weight: bold;")
            else:
                print("Warning: No enrolled users found in the system.")
                self.recognition_system_loaded = False
                self.title_label.setText("No users enrolled")
                self.title_label.setStyleSheet("color: orange; font-size: 24pt; font-weight: bold;")
                
        except Exception as e:
            print(f"Error loading recognition system: {e}")
            self.blazeface_model = None
            self.mobilefacenet_model = None
            self.faiss_index = None
            self.user_id_map = []
            self.recognition_system_loaded = False
            self.title_label.setText("System error - Check console")
            self.title_label.setStyleSheet("color: red; font-size: 24pt; font-weight: bold;")

    def play_beep_sound(self):
        """Play a beep sound when face is detected"""
        try:
            # Create a simple beep sound
            frequency = 800  # Hz
            duration = 0.2  # seconds
            sample_rate = 22050
            frames = int(duration * sample_rate)
            arr = np.sin(2 * np.pi * frequency * np.linspace(0, duration, frames))
            arr = (arr * 32767).astype(np.int16)
            arr = np.repeat(arr.reshape(frames, 1), 2, axis=1)
            
            # Play the sound
            pygame.mixer.Sound(arr).play()
        except Exception as e:
            print(f"Error playing beep sound: {e}")

    def setup_ui(self):
        # Set up the central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left section with dark gray background
        left_widget = QWidget()
        left_widget.setStyleSheet("background-color: #2C2F33;")
        left_layout = QVBoxLayout(left_widget)

        # Morning session label
        session_label = QLabel("MORNING SESSION")
        session_label.setStyleSheet("color: white; font-size: 24pt;")
        left_layout.addWidget(session_label)

        # Attendance count moved below morning session with doubled font size
        attendance_label = QLabel("240/241")
        attendance_label.setStyleSheet("color: white; font-size: 48pt; font-weight: bold;")
        left_layout.addWidget(attendance_label)

        # Room list in a scrollable area
        rooms = [
            "Jhansi : 23/24", "Aakash : 23/24", "Pruthvi : 23/24",
            "Virat : 23/24", "Sindhurakshak : 23/24", "Cheetah : 23/24",
            "Tejas : 23/24"
        ]
        room_widget = QWidget()
        room_layout = QVBoxLayout(room_widget)
        for room in rooms:
            label = QLabel(room)
            label.setStyleSheet("color: white; font-size: 16pt; padding: 2px;")
            room_layout.addWidget(label)

        room_scroll = QScrollArea()
        room_scroll.setWidget(room_widget)
        room_scroll.setWidgetResizable(True)
        left_layout.addWidget(room_scroll)

        # Right section with lighter gray background
        right_widget = QWidget()
        right_widget.setStyleSheet("background-color: #D3D3D3;")
        right_layout = QVBoxLayout(right_widget)

        # Title and attendance count at the top-right (this will be updated with detection results)
        title_widget = QWidget()
        title_layout = QVBoxLayout(title_widget)
        self.title_label = QLabel("Initializing system...")
        self.title_label.setStyleSheet("color: black; font-size: 24pt; font-weight: bold;")
        title_layout.addWidget(self.title_label)
        right_layout.addWidget(title_widget)

        # Camera feed label below the title
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.camera_label, 1)  # Stretch factor to fill remaining space

        # Add widgets to the main layout with appropriate stretch factors
        main_layout.addWidget(left_widget, 1)
        main_layout.addWidget(right_widget, 2)

    def setup_camera(self):
        # Initialize camera capture and timer for video updates
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30 ms
        print("Camera initialized and timer started.")

    def detect_and_recognize_faces(self, frame_rgb):
        """Detect and recognize faces in the frame"""
        try:
            # Check if all components are properly loaded
            if not self.recognition_system_loaded or self.faiss_index is None or self.blazeface_model is None or self.mobilefacenet_model is None:
                print(f"System check failed: recognition_loaded={self.recognition_system_loaded}, faiss_index={self.faiss_index is not None}, blazeface={self.blazeface_model is not None}, mobilefacenet={self.mobilefacenet_model is not None}")
                return "System not ready"

            original_frame_shape = frame_rgb.shape

            # 1. Face Detection
            input_blazeface = preprocess_image_blazeface(frame_rgb)
            detection_outputs = self.blazeface_model.run(input_blazeface)
            regressors = detection_outputs[0]
            classificators = detection_outputs[1]

            faces_found = postprocess_blazeface_output(
                regressors, classificators, original_frame_shape,
                score_threshold=MIN_DETECTION_SCORE
            )

            if not faces_found:
                # Log every 60 frames (about every 2 seconds) to avoid spam
                if self.frame_count % 60 == 0:
                    print("No face detected in current frame")
                return "No face detected"

            # Process the first detected face
            face = faces_found[0]
            x1, y1, x2, y2 = face['bbox']
            detection_score = face['score']

            print(f"Face detected! Bounding box: ({x1}, {y1}, {x2}, {y2}), Score: {detection_score:.3f}")

            # Ensure ROI is valid before cropping
            if x2 <= x1 or y2 <= y1:
                print("Invalid bounding box detected, skipping recognition")
                return "No face detected"

            face_roi = frame_rgb[y1:y2, x1:x2]

            if face_roi.size == 0 or face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
                print("Empty face ROI detected, skipping recognition")
                return "No face detected"

            # 2. Embedding Extraction
            input_mobilefacenet = preprocess_image_mobilefacenet(face_roi)
            embedding_output = self.mobilefacenet_model.run(input_mobilefacenet)
            embedding = embedding_output[0].flatten().astype(np.float32)

            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)

            # 3. Face Recognition
            recognized_name = recognize_face(embedding, self.faiss_index, self.user_id_map)
            print(f"Recognition result: {recognized_name}")

            return recognized_name

        except Exception as e:
            print(f"Error in face detection/recognition: {e}")
            return "Detection error"

    def update_frame(self):
        # Capture and display the camera frame
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Only run face detection if system is ready
            if self.recognition_system_loaded:
                # Perform face detection and recognition
                detected_name = self.detect_and_recognize_faces(frame_rgb)
                
                # Update UI based on detection results
                if detected_name not in ["No face detected", "Detection error", "System not ready"]:
                    # Face detected and recognized
                    if not self.last_detection_state:
                        # New detection - play beep sound and log
                        print(f"*** NEW FACE DETECTED: {detected_name} ***")
                        self.play_beep_sound()
                        self.last_detection_state = True
                    
                    self.title_label.setText(f"Detected: {detected_name}")
                    self.title_label.setStyleSheet("color: green; font-size: 24pt; font-weight: bold;")
                elif detected_name == "No face detected":
                    # No face detected
                    if self.last_detection_state:
                        print("Face detection lost")
                        self.last_detection_state = False
                    self.title_label.setText("No face detected")
                    self.title_label.setStyleSheet("color: red; font-size: 24pt; font-weight: bold;")
                else:
                    # Error states
                    self.last_detection_state = False
                    self.title_label.setText(detected_name)
                    self.title_label.setStyleSheet("color: orange; font-size: 24pt; font-weight: bold;")
            
            # Display the camera frame
            h, w, ch = frame_rgb.shape
            image = QImage(frame_rgb.data, w, h, w * ch, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.camera_label.setPixmap(pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio))

    def closeEvent(self, event):
        # Clean up camera resources on window close
        print("Shutting down attendance system...")
        self.cap.release()
        pygame.mixer.quit()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AttendanceUI()
    window.showFullScreen()
    sys.exit(app.exec())
import sys
import os

# Add the 'src' directory to the Python path. This allows the script to be run
# from the project root and still find modules in 'core', etc.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap

# Import recognition system functions
from core.recognition_system import FaceRecognitionSystem

class AttendanceUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Attendance System")
        
        # Initialize detection state
        self.last_detection_state = False
        self.current_detected_name = "Loading system..."
        self.recognition_system_loaded = False
        self.frame_count = 0
        
        # Setup UI first (so we can update status)
        self.setup_ui()

        # Initialize face recognition system
        self.recognizer = FaceRecognitionSystem()
        self.recognition_system_loaded = True
        self.title_label.setText("System ready - No face detected")
        self.title_label.setStyleSheet("color: blue; font-size: 24pt; font-weight: bold;")

        self.setup_camera()

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
        self.cap = None
        print("Searching for a working camera...")

        for i in range(5):  # Try indices 0 to 4
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"Success: Found working camera at index {i}.")
                    self.cap = cap
                    break
                else:
                    print(f"Camera at index {i} opened but failed to capture a frame. Releasing.")
                    cap.release()
            else:
                print(f"Could not open camera at index {i}.")

        if not self.cap or not self.cap.isOpened():
            print("Error: Could not find or open a working camera.")
            self.title_label.setText("Error: Camera not found")
            self.title_label.setStyleSheet("color: red; font-size: 24pt; font-weight: bold;")
            return

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(40)  # Update every 40 ms
        print("Camera initialized and timer started.")

    def detect_and_recognize_faces(self, frame_rgb):
        # Delegate detection and recognition to core FaceRecognitionSystem
        return self.recognizer.detect_and_recognize(frame_rgb)

    def update_frame(self):
        # Capture and display the camera frame
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Only run face detection if system is ready
            if self.recognition_system_loaded:
                # Delegate detection and recognition
                detected_name = self.detect_and_recognize_faces(frame_rgb)
                
                # Update UI based on detection results
                if detected_name not in ["No face detected", "Detection error", "System not ready"]:
                    # Face detected and recognized
                    if not self.last_detection_state:
                        # New detection - play beep sound and log
                        print(f"*** NEW FACE DETECTED: {detected_name} ***")
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
        else:
            print("Failed to grab frame from camera. Check camera connection.")

    def closeEvent(self, event):
        # Clean up camera resources on window close
        print("Shutting down attendance system...")
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AttendanceUI()
    window.showFullScreen()
    sys.exit(app.exec())
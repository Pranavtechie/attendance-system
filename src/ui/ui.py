import json
import os
import sys
from datetime import date, datetime, time

# Add the 'src' directory to the Python path. This allows the script to be run
# from the project root and still find modules in 'core', etc.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# ------------------------------------------------------------------
# Configurations
# ------------------------------------------------------------------
from config import WAIT_TIME_AFTER_RECOGNITION_MS

# Import recognition system functions
from core.recognition_system import FaceRecognitionSystem

# ------------------------------------------------------------------
# Database models (for dynamic room statistics)
# ------------------------------------------------------------------
from db.index import Person, Room, db
from ui.ipc_client import SocketThread


class AttendanceUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Attendance System")

        # Initialize detection state
        self.last_detection_state = False
        self.current_detected_name = "Loading system..."
        self.recognition_system_loaded = False
        self.frame_count = 0
        self.attendance_cache = {}
        self.distinct_detections = [0, 0, 0]
        self.date = date.today()
        self.SHIFT_1_START = time(9, 0)
        self.SHIFT_2_START = time(12, 0)
        self.SHIFT_3_START = time(15, 0)
        self.SHIFT_4_START = time(18, 0)
        self.SHIFT_END = time(23, 59)

        # Setup UI first (so we can update status)
        self.setup_ui()

        # Initialize face recognition system
        self.recognizer = FaceRecognitionSystem()
        self.recognition_system_loaded = True
        self.title_label.setText("System ready - No face detected")
        self.title_label.setStyleSheet(
            "color: blue; font-size: 24pt; font-weight: bold;"
        )

        # ------------------------------------------------------------------
        # IPC Socket client setup
        # ------------------------------------------------------------------
        self.socket_thread = SocketThread()
        self.socket_thread.message_received.connect(self.on_ipc_message)
        self.socket_thread.start()

        self.setup_camera()

    def get_current_session(self, check_time=None):
        check_time = check_time or datetime.now().time()

        def is_time_between(begin_time, end_time, check_time=None):
            if begin_time < end_time:
                # Standard range (e.g., 9am to 5pm)
                return begin_time <= check_time < end_time
            else:
                # Overnight range (e.g., 10pm to 2am)
                return check_time >= begin_time or check_time < end_time

        if is_time_between(self.SHIFT_1_START, self.SHIFT_2_START, check_time):
            return 1
        elif is_time_between(self.SHIFT_2_START, self.SHIFT_3_START, check_time):
            return 2
        elif is_time_between(self.SHIFT_3_START, self.SHIFT_4_START, check_time):
            return 3
        elif is_time_between(self.SHIFT_4_START, self.SHIFT_END, check_time):
            return 4
        else:
            return 0

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

        # Overall attendance counter – will be kept in sync with table data
        self.attendance_label = QLabel("0/0")
        self.attendance_label.setStyleSheet(
            "color: white; font-size: 48pt; font-weight: bold;"
        )
        left_layout.addWidget(self.attendance_label)

        # ------------------------------------------------------------------
        # Dynamic room attendance table
        # ------------------------------------------------------------------
        self.rooms_table = QTableWidget()
        self.rooms_table.setColumnCount(4)
        self.rooms_table.setHorizontalHeaderLabels(
            [
                "Room",
                "Total",
                "Present",
                "Pending",
            ]
        )
        self.rooms_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.rooms_table.verticalHeader().setVisible(False)
        self.rooms_table.setEditTriggers(QTableWidget.NoEditTriggers)

        # ------------------------------------------------------------------
        # Style: double default font size for cells and bold header
        # ------------------------------------------------------------------
        cell_font = self.rooms_table.font()
        cell_font.setPointSize(cell_font.pointSize() * 2)
        self.rooms_table.setFont(cell_font)

        header = self.rooms_table.horizontalHeader()
        header_font = header.font()
        header_font.setBold(True)
        header_font.setPointSize(header_font.pointSize() * 1.25)
        header.setFont(header_font)

        # Populate table from DB
        self.load_room_data()

        left_layout.addWidget(self.rooms_table)

        # Right section with lighter gray background
        right_widget = QWidget()
        right_widget.setStyleSheet("background-color: #D3D3D3;")
        right_layout = QVBoxLayout(right_widget)

        # Title and attendance count at the top-right (this will be updated with detection results)
        title_widget = QWidget()
        title_layout = QVBoxLayout(title_widget)
        self.title_label = QLabel("Initializing system...")
        self.title_label.setStyleSheet(
            "color: black; font-size: 24pt; font-weight: bold;"
        )
        title_layout.addWidget(self.title_label)
        right_layout.addWidget(title_widget)

        # Camera feed label below the title
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.camera_label, 1)

        # ------------------------------------------------------------------
        # Message log below camera view
        # ------------------------------------------------------------------
        self.message_view = QTextEdit()
        self.message_view.setReadOnly(True)
        self.message_view.setFixedHeight(120)
        self.message_view.setStyleSheet(
            "background-color: #F0F0F0; color: black; font-size: 12pt;"
        )
        right_layout.addWidget(self.message_view)

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
                    print(
                        f"Camera at index {i} opened but failed to capture a frame. Releasing."
                    )
                    cap.release()
            else:
                print(f"Could not open camera at index {i}.")

        if not self.cap or not self.cap.isOpened():
            print("Error: Could not find or open a working camera.")
            self.title_label.setText("Error: Camera not found")
            self.title_label.setStyleSheet(
                "color: red; font-size: 24pt; font-weight: bold;"
            )
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
                if detected_name not in [
                    "No face detected",
                    "Detection error",
                    "System not ready",
                    "Unknown",
                ]:
                    # Face detected and recognised
                    if self.date != date.today():
                        self.date = date.today()
                        self.attendance_cache = {}

                    session = self.get_current_session(datetime.now().time())
                    if (
                        detected_name not in self.distinct_detections
                        or detected_name not in self.attendance_cache
                        or self.attendance_cache[detected_name] != session
                    ):
                        self.attendance_cache[detected_name] = session
                        if not self.last_detection_state:
                            # New detection - play beep sound and log
                            print(f"*** NEW FACE DETECTED: {detected_name} ***")
                            self.last_detection_state = True

                        # --------------------------------------------------
                        # Notify server about recognition via IPC JSON payload
                        # --------------------------------------------------
                        try:
                            # Retrieve Person record once so it can be reused
                            person = Person.get_or_none(Person.name == detected_name)
                        except Exception as e:
                            print(f"DB lookup error: {e}")
                            person = None

                        if person is not None:
                            payload = json.dumps(
                                {
                                    "action": "person-recognized",
                                    "data": {
                                        "personId": person.uniqueId,
                                        "attendanceTimeStamp": datetime.now().isoformat(),
                                    },
                                }
                            )
                            try:
                                self.socket_thread.send(payload)
                            except Exception as exc:
                                print(f"IPC send error: {exc}")
                        else:
                            print(
                                "Person record not found for detected name - IPC not sent"
                            )

                        self.distinct_detections.append(detected_name)
                        del self.distinct_detections[0]

                        # --------------------------------------------------
                        # Update dynamic room statistics
                        # --------------------------------------------------
                        try:
                            person = Person.get_or_none(Person.name == detected_name)
                            # Update statistics only for cadets (exclude staff)
                            if person and person.personType == "Cadet":
                                room_id = person.roomId
                                if person.uniqueId not in self.present_per_room.get(
                                    room_id, set()
                                ):
                                    self.present_per_room.setdefault(
                                        room_id, set()
                                    ).add(person.uniqueId)
                                    self.update_room_present(room_id)
                        except Exception as e:
                            print(f"Room stats update error: {e}")

                        # --------------------------------------------------
                        # Pause camera updates briefly so the recognised
                        # frame stays visible before capturing again.
                        # --------------------------------------------------
                        self.pause_after_recognition()

                    else:
                        print("Duplicate Face ", detected_name)

                    print(self.distinct_detections)

                    # ------------------------------------------------------
                    # Simplified top label display
                    # ------------------------------------------------------
                    try:
                        person = (
                            person
                            if "person" in locals()
                            else Person.get_or_none(Person.name == detected_name)
                        )
                        if person:
                            if person.personType == "Cadet":
                                display_text = (
                                    f"{person.admissionNumber} - {person.name}"
                                )
                            else:
                                display_text = person.name
                        else:
                            # Fallback – shouldn't normally occur
                            display_text = detected_name
                    except Exception:
                        display_text = detected_name

                    self.title_label.setText(display_text)
                    self.title_label.setStyleSheet(
                        "color: green; font-size: 24pt; font-weight: bold;"
                    )
                else:
                    # For all non-recognition states (no face, unknown, errors) – clear label
                    if self.last_detection_state and detected_name in [
                        "No face detected",
                        "Unknown",
                        "Detection error",
                        "System not ready",
                    ]:
                        # Reset detection state when face is lost or becomes unknown
                        print("Face detection lost or unknown face")
                        self.last_detection_state = False

                    self.title_label.setText("")
                    # Keep previous style but could adjust if desired
                    # self.title_label.setStyleSheet("color: grey; font-size: 24pt; font-weight: bold;")

            # Display the camera frame
            h, w, ch = frame_rgb.shape
            image = QImage(frame_rgb.data, w, h, w * ch, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.camera_label.setPixmap(
                pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio)
            )
        else:
            print("Failed to grab frame from camera. Check camera connection.")

    def closeEvent(self, event):
        # Clean up camera resources on window close
        print("Shutting down attendance system...")
        self.cap.release()
        if self.socket_thread.isRunning():
            self.socket_thread.quit()
        event.accept()

    # ------------------------------------------------------------------
    # IPC callbacks
    # ------------------------------------------------------------------

    def on_ipc_message(self, message: str):
        """Handle messages broadcast by the Flask server."""
        print(f"[IPC] Message from server: {message}")
        # Append message to log view
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.message_view.append(f"[{timestamp}] {message}")

    # ------------------------------------------------------------------
    # Helpers for dynamic room statistics
    # ------------------------------------------------------------------

    def load_room_data(self):
        """Initialise room statistics and fill the QTableWidget."""
        if db.is_closed():
            db.connect(reuse_if_open=True)

        rooms = list(Room.select())

        self.room_row_index = {}
        self.present_per_room = {}
        self.room_stats = []  # Keeps total counts per room

        self.rooms_table.setRowCount(len(rooms))

        for idx, room in enumerate(rooms):
            # Count only cadets (exclude staff) assigned to this room
            total = (
                Person.select()
                .where((Person.roomId == room.roomId) & (Person.personType == "Cadet"))
                .count()
                if room.roomId
                else 0
            )

            # Cache
            self.room_stats.append({"roomId": room.roomId, "total": total})
            self.room_row_index[room.roomId] = idx
            self.present_per_room[room.roomId] = set()

            # Populate table
            self.rooms_table.setItem(idx, 0, QTableWidgetItem(room.roomName))

            # Create number cells and center-align them
            num_cols_values = [(1, total), (2, 0), (3, total)]
            for col, val in num_cols_values:
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignCenter)
                self.rooms_table.setItem(idx, col, item)

        self.total_cadets = sum(stat["total"] for stat in self.room_stats)
        self.update_attendance_label()

    def update_room_present(self, room_id: str):
        """Refresh a single row in the table after a new detection."""
        row = self.room_row_index.get(room_id)
        if row is None:
            return

        present_cnt = len(self.present_per_room[room_id])
        total_cnt = int(self.rooms_table.item(row, 1).text())

        self.rooms_table.item(row, 2).setText(str(present_cnt))
        self.rooms_table.item(row, 3).setText(str(total_cnt - present_cnt))

        self.update_attendance_label()

    def update_attendance_label(self):
        """Update the big attendance ratio label under the session header."""
        present_total = sum(len(s) for s in self.present_per_room.values())
        self.attendance_label.setText(f"{present_total}/{self.total_cadets}")

    # ------------------------------------------------------------------
    # Helpers for pausing/resuming camera capture
    # ------------------------------------------------------------------

    def pause_after_recognition(self):
        """Temporarily stop the camera timer after a successful recognition."""
        if hasattr(self, "timer") and self.timer.isActive():
            self.timer.stop()
            # Resume after the configured wait time
            QTimer.singleShot(WAIT_TIME_AFTER_RECOGNITION_MS, self.resume_camera)

    def resume_camera(self):
        """Restart the periodic camera capture timer."""
        if hasattr(self, "timer") and not self.timer.isActive():
            self.timer.start(40)  # Restore the original frame interval


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AttendanceUI()
    window.showFullScreen()
    sys.exit(app.exec())

import sys
import cv2
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera Feed")
        self.label = QLabel(self)
        self.setCentralWidget(self.label)
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
        self.showFullScreen()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            image = QImage(frame.data, w, h, w * ch, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.KeepAspectRatio))

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

app = QApplication(sys.argv)
window = MainWindow()
sys.exit(app.exec())

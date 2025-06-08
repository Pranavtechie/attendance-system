import sys
from PySide6.QtWidgets import QApplication
from ui.ui import AttendanceUI

def main():
    app = QApplication(sys.argv)
    window = AttendanceUI()
    window.showFullScreen()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 
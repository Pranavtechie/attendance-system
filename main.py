def main():
    print("Hello from attendance-system!")


if __name__ == "__main__":
    main()
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow

app = QApplication(sys.argv)
window = QMainWindow()
window.setWindowTitle("Basic PyQt6")
window.showFullScreen()
sys.exit(app.exec())

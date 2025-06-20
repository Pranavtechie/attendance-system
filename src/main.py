import sys

# ------------------------------------------------------------------
# Initialise logging before importing the rest of the application so
# that every module picks up the configuration.
# ------------------------------------------------------------------
from src.core.log_config import setup_logging

setup_logging()

from PySide6.QtWidgets import QApplication

from src.ui.ui import AttendanceUI


def main():
    app = QApplication(sys.argv)
    window = AttendanceUI()
    window.showFullScreen()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

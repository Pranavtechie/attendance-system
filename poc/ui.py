import sys
import socket
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QLineEdit, QTextEdit, QVBoxLayout, QWidget
from PySide6.QtCore import QThread, Signal

SOCKET_PATH = "/tmp/app.sock"

class SocketThread(QThread):
    message_received = Signal(str)

    def run(self):
        """Connect to the server socket and listen for messages."""
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.sock.connect(SOCKET_PATH)
            while True:
                data = self.sock.recv(1024).decode()
                if not data:
                    break
                self.message_received.emit(data)
        except Exception as e:
            self.message_received.emit(f"Connection error: {str(e)}")
        finally:
            self.sock.close()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PySide6 UI")
        self.resize(400, 300)

        # UI elements
        self.input = QLineEdit()
        self.button = QPushButton("Send")
        self.messages = QTextEdit()
        self.messages.setReadOnly(True)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.input)
        layout.addWidget(self.button)
        layout.addWidget(self.messages)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Socket thread for receiving messages
        self.socket_thread = SocketThread()
        self.socket_thread.message_received.connect(self.append_message)
        self.socket_thread.start()

        # Button click to send message
        self.button.clicked.connect(self.send_message)

    def send_message(self):
        """Send message to the server via socket."""
        message = self.input.text()
        if message:
            try:
                self.socket_thread.sock.send(message.encode())
                self.append_message(f"Me: {message}")
                self.input.clear()
            except Exception as e:
                self.append_message(f"Send error: {str(e)}")

    def append_message(self, message):
        """Display received or sent messages in the UI."""
        self.messages.append(message)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
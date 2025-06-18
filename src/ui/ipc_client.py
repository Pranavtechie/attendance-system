import socket
from typing import Optional

from PySide6.QtCore import QThread, Signal

try:
    # When running from within the 'src' directory (e.g., unit tests)
    from src.config import SOCKET_PATH  # type: ignore
except ModuleNotFoundError:
    # When UI app manipulates sys.path to include 'src' directly
    from config import SOCKET_PATH


class SocketThread(QThread):
    """Background Qt thread for IPC communication with Flask server."""

    message_received = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sock: Optional[socket.socket] = None

    # ------------------------------------------------------------------
    def run(self):
        """Connect to the Unix socket and forward incoming messages."""
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self._sock.connect(str(SOCKET_PATH))
            while True:
                data = self._sock.recv(1024).decode()
                if not data:
                    break
                self.message_received.emit(data)
        except Exception as exc:
            self.message_received.emit(f"IPC connection error: {exc}")
        finally:
            if self._sock:
                self._sock.close()

    # ------------------------------------------------------------------
    def send(self, message: str):
        """Send a message to the server over the socket."""
        if self._sock is None:
            return  # Not connected yet
        try:
            self._sock.send(message.encode())
        except Exception as exc:
            self.message_received.emit(f"IPC send error: {exc}")

import os
import queue
import socket
import threading
from typing import List

from src.config import SOCKET_PATH


class _Client:
    """Internal helper to manage read/write loops for a connected UI client."""

    def __init__(self, conn: socket.socket, remove_callback):
        self._conn = conn
        self._send_q: "queue.Queue[str]" = queue.Queue()
        self._stop = threading.Event()
        self._remove_callback = remove_callback
        # Start I/O threads.
        self._recv_thr = threading.Thread(target=self._recv_loop, daemon=True)
        self._send_thr = threading.Thread(target=self._send_loop, daemon=True)
        self._recv_thr.start()
        self._send_thr.start()

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    def enqueue(self, message: str):
        """Queue a message to be sent to this client."""
        # Non‐blocking, drop message if client already closed
        if not self._stop.is_set():
            self._send_q.put(message)

    def close(self):
        self._stop.set()
        try:
            self._conn.close()
        finally:
            # drain queue to unblock send loop
            self._send_q.put(None)

    # ---------------------------------------------------------------------
    # Private loops
    # ---------------------------------------------------------------------
    def _recv_loop(self):
        try:
            while not self._stop.is_set():
                data = self._conn.recv(1024).decode()
                if not data:
                    break
                # Currently we only log incoming messages, but this is where
                # server-side handling can be attached.
                print(f"[IPC] Received from UI: {data}")
        except Exception as exc:
            print(f"[IPC] Receive error: {exc}")
        finally:
            self._stop.set()
            self._remove_callback(self)

    def _send_loop(self):
        try:
            while not self._stop.is_set():
                message = self._send_q.get()
                if message is None:
                    break
                try:
                    self._conn.send(message.encode())
                except Exception as exc:
                    print(f"[IPC] Send error: {exc}")
                    break
        finally:
            self._stop.set()
            self._remove_callback(self)


class SocketServer:
    """Unix domain socket server for IPC with the Qt UI."""

    def __init__(self, socket_path: str = str(SOCKET_PATH)):
        self._socket_path = socket_path
        # Ensure old socket file is removed.
        if os.path.exists(self._socket_path):
            try:
                os.unlink(self._socket_path)
            except OSError as exc:
                raise RuntimeError(f"Unable to unlink stale socket: {exc}")
        # Create server socket.
        self._srv_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._srv_sock.bind(self._socket_path)
        self._srv_sock.listen()
        # Track clients.
        self._clients: List[_Client] = []
        self._clients_lock = threading.Lock()
        # Start accept thread.
        threading.Thread(target=self._accept_loop, daemon=True).start()
        print(f"[IPC] Unix socket server listening at {self._socket_path}")

    # ------------------------------------------------------------------
    def _remove_client(self, client: _Client):
        with self._clients_lock:
            if client in self._clients:
                self._clients.remove(client)

    def _accept_loop(self):
        while True:
            conn, _ = self._srv_sock.accept()
            client = _Client(conn, self._remove_client)
            with self._clients_lock:
                self._clients.append(client)
            print("[IPC] UI client connected – total:", len(self._clients))

    # ------------------------------------------------------------------
    def broadcast(self, message: str):
        """Broadcast `message` to every connected UI client."""
        with self._clients_lock:
            for c in list(self._clients):
                c.enqueue(message)


# Singleton instance lazily created on import so that simply importing this
# module starts the server and makes `broadcast` available.
_socket_server = SocketServer()


def broadcast_message(message: str):
    """Module-level helper to broadcast a message to all UI clients."""
    _socket_server.broadcast(message)

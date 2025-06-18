from flask import Flask, request, jsonify
import socket
import os
import threading
import queue

app = Flask(__name__)
SOCKET_PATH = "/tmp/app.sock"

# Remove existing socket file if present
if os.path.exists(SOCKET_PATH):
    os.unlink(SOCKET_PATH)

# Create Unix socket
server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server_socket.bind(SOCKET_PATH)
server_socket.listen()

# List of connected clients and lock for thread safety
clients = []
clients_lock = threading.Lock()

class Client:
    def __init__(self, conn):
        self.conn = conn
        self.send_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.recv_thread = threading.Thread(target=self.recv_loop)
        self.send_thread = threading.Thread(target=self.send_loop)
        self.recv_thread.start()
        self.send_thread.start()

    def recv_loop(self):
        """Handle messages received from the UI via socket."""
        try:
            while not self.stop_event.is_set():
                data = self.conn.recv(1024).decode()
                if not data:
                    break
                print(f"Server received from UI: {data}")
                # Optionally echo back to the sender or process the message
                self.send_queue.put(f"Echo: {data}")
        except Exception as e:
            print(f"Receive error: {e}")
        finally:
            self.stop_event.set()
            self.conn.close()
            with clients_lock:
                if self in clients:
                    clients.remove(self)

    def send_loop(self):
        """Send messages to the UI from the send queue."""
        try:
            while not self.stop_event.is_set():
                message = self.send_queue.get()
                if message is None:
                    break
                self.conn.send(message.encode())
        except Exception as e:
            print(f"Send error: {e}")
        finally:
            self.stop_event.set()
            self.conn.close()

def socket_listener():
    """Accept new socket connections from UI clients."""
    while True:
        conn, _ = server_socket.accept()
        client = Client(conn)
        with clients_lock:
            clients.append(client)

# Start socket listener in a separate thread
threading.Thread(target=socket_listener, daemon=True).start()

@app.route('/')
def index():
    return "Flask server running"

@app.route('/api/send', methods=['POST'])
def send_message():
    """Handle POST requests and broadcast message to all UI clients."""
    data = request.json
    message = data.get('message')
    if not message:
        return jsonify({"error": "No message provided"}), 400
    # Broadcast the message to all connected clients
    with clients_lock:
        for client in clients:
            client.send_queue.put(f"Server broadcast: {message}")
    return jsonify({"status": "Message broadcasted", "message": message}), 200

if __name__ == "__main__":
    app.run(debug=True)  # Use `gunicorn -w 4 server:app` in production
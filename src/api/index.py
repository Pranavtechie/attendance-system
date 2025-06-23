import json
import os
from datetime import datetime
from urllib.parse import urlparse

import pytz
import requests as req
from flask import Flask, request
from flask_cors import CORS
from uuid_extensions import uuid7str

from src.config import ENROLLMENT_IMAGES_DIR
from src.core.enrollment_processor import enroll_user
from src.db.index import CadetAttendance, Person, Room, db
from src.ipc.socket_server import (  # IPC helper
    broadcast_message,
    register_event_handler,
)


def ist_timestamp():
    dt = datetime.now(pytz.timezone("Asia/Kolkata"))
    milliseconds = dt.microsecond // 1000  # Convert microseconds to milliseconds
    dt = dt.replace(
        microsecond=milliseconds * 1000
    )  # Set precision to 3 decimal places
    return dt.isoformat()


def string_to_timestamp(s):
    if " " in s:
        s = s.replace(" ", "T", 1)  # Replace first space with 'T'
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"  # Replace 'Z' with '+00:00'
    dt = datetime.fromisoformat(s)  # Parse ISO string
    return dt.astimezone(pytz.timezone("Asia/Kolkata")).isoformat()


app = Flask(__name__)
CORS(
    app,
    resources={
        r"/*": {
            "origins": [
                "http://localhost:8787",
                "https://api.korukondacoachingcentre.com",
                "http://localhost:3000",
            ],
            "methods": ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
            "allow_headers": ["Content-Type"],
        }
    },
)


@app.before_request
def _open_db():
    if db.is_closed():
        db.connect(reuse_if_open=True)


@app.teardown_request
def _close_db(exc):
    if not db.is_closed():
        db.close()


@app.route("/")
def hello_world():
    return {"message": "Hello, World!"}


@app.route("/test", methods=["GET"])
def test():
    return {"uuid": uuid7str()}


@app.route("/enroll", methods=["POST"])
def enroll():
    data = request.json
    print(data)
    syncedAt = ist_timestamp()

    picture_url = data.get("picture")

    # --- Validate and download the image --- #
    if not picture_url:
        return {"message": "pictureUrl missing from payload"}, 400

    # Ensure the URL has a .jpg filename
    parsed_url = urlparse(picture_url)
    filename = os.path.basename(parsed_url.path)

    if not filename.lower().endswith(".jpg"):
        return {"message": "Only .jpg images are supported."}, 400

    local_path = os.path.join(ENROLLMENT_IMAGES_DIR, filename)

    # ------------------------------------------------------------
    # Download the image if it is not already present locally.
    # This avoids unnecessary network calls and file overwrites
    # when the same image has already been cached on disk.
    # ------------------------------------------------------------

    downloaded_now = False  # Track whether we fetched the file in this request

    if not os.path.exists(local_path):
        try:
            response = req.get(picture_url, timeout=15)
            if response.status_code != 200:
                return {
                    "message": "Failed to download image",
                    "status": response.status_code,
                }, 502

            # Basic content‐type validation (allows e.g. image/jpeg)
            content_type = response.headers.get("Content-Type", "")
            if "image/jpeg" not in content_type.lower():
                return {"message": "URL does not point to a JPEG image"}, 400

            # Write the image to disk
            with open(local_path, "wb") as f:
                f.write(response.content)

            downloaded_now = True  # Mark that we downloaded the file

        except Exception as e:
            print(e)
            return {"message": "Error downloading image", "error": str(e)}, 500
    else:
        # Reuse the cached image instead of downloading again
        print(f"[Enroll] Reusing cached image {local_path}")

    try:
        if data["userType"] == "Cadet":
            Person.insert(
                uniqueId=data["personId"],
                name=data["preferredName"],
                admissionNumber=data["admissionNumber"],
                roomId=data["roomId"],
                pictureFileName=filename,
                personType=data["userType"],
                syncedAt=syncedAt,
            ).on_conflict_replace().execute()
        elif data["userType"] == "Employee":
            print("\n\n using employee ")
            Person.insert(
                uniqueId=data["personId"],
                name=data["preferredName"],
                pictureFileName=filename,
                personType=data["userType"],
            ).on_conflict_replace().execute()

    except Exception as e:
        # Cleanup the saved image only if we downloaded it in this request
        print(e)
        if downloaded_now and os.path.exists(local_path):
            try:
                os.remove(local_path)
            except OSError:
                pass
        return {"message": "Enrollment failed", "error": str(e)}, 500

    try:
        enroll_user(data["personId"], local_path)
    except Exception as e:
        print(e)
        return {
            "message": "Enrollment failed - Couldn't enroll to FAISS",
            "error": str(e),
        }, 500

    # Notify UI clients about new enrollment
    broadcast_message(f"Enrollment completed for {data['preferredName']}")

    return {"syncedAt": syncedAt}, 200


@app.route("/ipc/send", methods=["POST"])
def ipc_send():
    """Broadcast a message received via HTTP to all connected UI clients."""
    data = request.json or {}
    message = data.get("message")
    if not message:
        return {"error": "No message provided"}, 400

    broadcast_message(message)
    return {"status": "Message broadcasted", "message": message}, 200


@app.route("/setup-rooms", methods=["POST"])
def setup_rooms():
    rooms = [
        {
            "unique_id": "01978221-6a29-70f0-99f0-996d856ecf47",
            "room_name": "Jhansi",
            "warden_name": None,
            "created_at": "2025-06-18 08:21:57.417",
            "updated_at": "2025-06-18 08:21:57.426",
            "gender": "Female",
            "place": "0-Left",
        },
        {
            "unique_id": "01978221-769d-768f-894e-e833a9cbcfd4",
            "room_name": "Aakash",
            "warden_name": None,
            "created_at": "2025-06-18 08:22:00.605",
            "updated_at": "2025-06-18 08:22:00.607",
            "gender": "Male",
            "place": "1-Left",
        },
        {
            "unique_id": "01978221-819e-734c-a2f0-7e58b7ebd42d",
            "room_name": "Prithvi",
            "warden_name": None,
            "created_at": "2025-06-18 08:22:03.422",
            "updated_at": "2025-06-18 08:22:03.423",
            "gender": "Male",
            "place": "1-Right",
        },
        {
            "unique_id": "01978221-87db-750a-b10e-119bdc4b88be",
            "room_name": "Viraat",
            "warden_name": None,
            "created_at": "2025-06-18 08:22:05.019",
            "updated_at": "2025-06-18 08:22:05.021",
            "gender": "Male",
            "place": "2-Left",
        },
        {
            "unique_id": "01978221-8f44-70d6-88f3-b459ac5273df",
            "room_name": "Sindhurakshak",
            "warden_name": None,
            "created_at": "2025-06-18 08:22:06.916",
            "updated_at": "2025-06-18 08:22:06.918",
            "gender": "Male",
            "place": "2-Right",
        },
        {
            "unique_id": "01978221-9589-726c-98d8-64b502ca02dc",
            "room_name": "Tejas",
            "warden_name": None,
            "created_at": "2025-06-18 08:22:08.521",
            "updated_at": "2025-06-18 08:22:08.522",
            "gender": "Male",
            "place": "3-Left",
        },
        {
            "unique_id": "01978221-9cf8-72ea-bbf5-1ad2a85b5393",
            "room_name": "Cheetah",
            "warden_name": None,
            "created_at": "2025-06-18 08:22:10.424",
            "updated_at": "2025-06-18 08:22:10.426",
            "gender": "Male",
            "place": "3-Right",
        },
    ]

    try:
        for room in rooms:
            Room.insert(
                roomId=room["unique_id"],
                roomName=room["room_name"],
                syncedAt=ist_timestamp(),
            ).on_conflict_replace().execute()
    except Exception as e:
        print(e)
        return {"message": "Failed to setup rooms", "error": str(e)}, 500

    return {"message": "Rooms setup successfully"}, 200


# ---------------------------------------------------------------------------
# IPC message handling
# ---------------------------------------------------------------------------


def _mark_attendance_remote(person_id: str, attendanceTimeStamp: str):
    """Send attendance mark request to the remote Axon API and return the
    `syncedAt` timestamp if the call is successful. Returns None on failure.
    """
    url = "https://api.korukondacoachingcenter.com/axon/mark-attendance"
    try:
        resp = req.post(
            url,
            json={"personId": person_id, "attendanceTimeStamp": attendanceTimeStamp},
            timeout=15,
        )
        if resp.status_code != 200:
            print(
                f"[Attendance] Remote API error {resp.status_code}: {resp.text[:200]}"
            )
            return None
        data = resp.json()
        return data.get("syncedAt")
    except Exception as exc:
        print(f"[Attendance] Remote API request failed: {exc}")
        return None


def _handle_ipc_message(message: str):
    """Callback invoked for every message received over the IPC socket."""
    try:
        payload = json.loads(message)
    except json.JSONDecodeError:
        # Ignore non-JSON messages
        return

    action = payload.get("action")
    data = payload.get("data", {}) or {}

    if action == "person-recognized":
        person_id = data.get("personId")
        attendanceTimeStamp = data.get("attendanceTimeStamp")
        if not person_id:
            print("[Attendance] person-recognized payload missing personId")
            return

        synced_at = _mark_attendance_remote(person_id, attendanceTimeStamp)

        # Persist to local DB regardless of remote sync success
        try:
            if db.is_closed():
                db.connect(reuse_if_open=True)

            CadetAttendance.insert(
                personId=person_id,
                attendanceTimeStamp=datetime.now(),
                sessionId="",  # TODO: store actual session when available
                syncedAt=synced_at,
            ).execute()
        except Exception as exc:
            print(f"[Attendance] DB write error: {exc}")


# Register the handler as soon as the module is imported so that the socket
# server can forward messages.
register_event_handler(_handle_ipc_message)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=1337)

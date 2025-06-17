import os
from datetime import datetime
from urllib.parse import urlparse

import pytz
import requests as req
from flask import Flask, jsonify, request
from flask_cors import CORS
from uuid_extensions import uuid7str

from src.db.index import Cadet, CadetAttendance, Room, Session, db


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

    picture_url = data.get("pictureUrl")

    # --- Validate and download the image --- #
    if not picture_url:
        return {"message": "pictureUrl missing from payload"}, 400

    # Ensure the URL has a .jpg filename
    parsed_url = urlparse(picture_url)
    filename = os.path.basename(parsed_url.path)

    if not filename.lower().endswith(".jpg"):
        return {"message": "Only .jpg images are supported."}, 400

    local_path = os.path.join(ENROLLMENT_IMAGES_DIR, filename)

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

    except Exception as e:
        return {"message": "Error downloading image", "error": str(e)}, 500

    try:
        Cadet.insert(
            uniqueId=data["uniqueId"],
            name=data["preferredName"],
            admissionNumber=data["admissionNumber"],
            roomId=data["roomId"],
            pictureFileName=filename,
            syncedAt=syncedAt,
        ).on_conflict_replace().execute()

    except Exception as e:
        # Cleanup the saved image if DB write fails
        if os.path.exists(local_path):
            try:
                os.remove(local_path)
            except OSError:
                pass

        return {"message": "Enrollment failed", "error": str(e)}, 500

    return {"syncedAt": syncedAt}, 200


# @app.route('/api/enroll', methods=['POST'])
# def api_enroll_user():
#     # Check if the post request has the file part
#     if 'image' not in request.files:
#         return jsonify({"status": "error", "message": "No image file part in the request."}), 400

#     file = request.files['image']

#     # If the user does not select a file, the browser submits an empty file without a filename.
#     if file.filename == '':
#         return jsonify({"status": "error", "message": "No image selected for uploading."}), 400

#     # Check for other form data
#     name = request.form.get('name')
#     admission_number = request.form.get('admissionNumber')
#     room = request.form.get('room')

#     if not all([name, admission_number, room]):
#         return jsonify({"status": "error",
#                         "message": "Missing data: name, admissionNumber, or room is required."}), 400

#     if file: # and allowed_file(file.filename) # You can add file type validation if needed
#         try:
#             # Create a unique filename to prevent overwrites
#             # Using original extension, or force .jpg/.png if you standardize
#             original_filename = file.filename
#             extension = os.path.splitext(original_filename)[1].lower()
#             if extension not in ['.jpg', '.jpeg', '.png']: # Basic validation
#                 return jsonify({"status": "error", "message": "Invalid image file type. Use JPG or PNG."}), 400

#             filename = f"{uuid.uuid4()}{extension}"
#             image_path = os.path.join(ENROLLMENT_IMAGES_DIR, filename)

#             file.save(image_path)
#             print(f"Image saved to: {image_path}")

#             # Call the enrollment logic from Phase 1 (enrollment_processor.py)
#             success, message, user_id = enroll_new_user(name, admission_number, room, image_path)

#             if success:
#                 return jsonify({"status": "success", "message": message, "user_id": user_id}), 201 # 201 Created
#             else:
#                 # Optionally, delete the saved image if enrollment processing failed significantly
#                 # This depends on whether you want to keep images from failed attempts for debugging
#                 # if os.path.exists(image_path):
#                 #     try:
#                 #         os.remove(image_path)
#                 #     except Exception as e_del:
#                 #         print(f"Error deleting image after failed enrollment: {e_del}")
#                 return jsonify({"status": "error", "message": message, "user_id": user_id}), 400 # Bad Request or 500 if server error

#         except Exception as e:
#             # Catch any other unexpected errors during file save or processing
#             print(f"An unexpected error occurred during enrollment: {e}")
#             # If an image_path was determined, try to clean it up
#             if 'image_path' in locals() and os.path.exists(image_path):
#                 try:
#                     os.remove(image_path)
#                 except OSError as e_del:
#                     print(f"Error deleting image after failed enrollment: {e_del}")
#             return jsonify({"status": "error", "message": f"An unexpected server error occurred: {str(e)}"}), 500

#     return jsonify({"status": "error", "message": "Image processing failed."}), 400

# Initialize DB schema once when the app starts
# This ensures the table and columns are ready before the first request.
# with app.app_context():
#     init_enrollment_db() # From core.enrollment_processor

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=1337)

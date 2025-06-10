from flask import Flask, request, jsonify
import os
import uuid
import requests as req
from src.core.enrollment_processor import enroll_new_user, init_enrollment_db


ENROLLMENT_IMAGES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'enrollment_images')
os.makedirs(ENROLLMENT_IMAGES_DIR, exist_ok=True)


app = Flask(__name__)

@app.route("/")
def hello_world():
    return {"message": "Hello, World!"}


@app.route("/test", methods=["GET"])
def test():
    r = req.get('https://api.github.com/events')
    
    return r.json()

@app.route('/api/enroll', methods=['POST'])
def api_enroll_user():
    # Check if the post request has the file part
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "No image file part in the request."}), 400
    
    file = request.files['image']

    # If the user does not select a file, the browser submits an empty file without a filename.
    if file.filename == '':
        return jsonify({"status": "error", "message": "No image selected for uploading."}), 400

    # Check for other form data
    name = request.form.get('name')
    admission_number = request.form.get('admissionNumber')
    room = request.form.get('room')

    if not all([name, admission_number, room]):
        return jsonify({"status": "error", 
                        "message": "Missing data: name, admissionNumber, or room is required."}), 400

    if file: # and allowed_file(file.filename) # You can add file type validation if needed
        try:
            # Create a unique filename to prevent overwrites
            # Using original extension, or force .jpg/.png if you standardize
            original_filename = file.filename
            extension = os.path.splitext(original_filename)[1].lower()
            if extension not in ['.jpg', '.jpeg', '.png']: # Basic validation
                return jsonify({"status": "error", "message": "Invalid image file type. Use JPG or PNG."}), 400

            filename = f"{uuid.uuid4()}{extension}"
            image_path = os.path.join(ENROLLMENT_IMAGES_DIR, filename)
            
            file.save(image_path)
            print(f"Image saved to: {image_path}")

            # Call the enrollment logic from Phase 1 (enrollment_processor.py)
            success, message, user_id = enroll_new_user(name, admission_number, room, image_path)

            if success:
                return jsonify({"status": "success", "message": message, "user_id": user_id}), 201 # 201 Created
            else:
                # Optionally, delete the saved image if enrollment processing failed significantly
                # This depends on whether you want to keep images from failed attempts for debugging
                # if os.path.exists(image_path):
                #     try:
                #         os.remove(image_path)
                #     except Exception as e_del:
                #         print(f"Error deleting image after failed enrollment: {e_del}")
                return jsonify({"status": "error", "message": message, "user_id": user_id}), 400 # Bad Request or 500 if server error

        except Exception as e:
            # Catch any other unexpected errors during file save or processing
            print(f"An unexpected error occurred during enrollment: {e}")
            # If an image_path was determined, try to clean it up
            if 'image_path' in locals() and os.path.exists(image_path):
                try:
                    os.remove(image_path)
                except OSError as e_del:
                    print(f"Error deleting image after failed enrollment: {e_del}")
            return jsonify({"status": "error", "message": f"An unexpected server error occurred: {str(e)}"}), 500
    
    return jsonify({"status": "error", "message": "Image processing failed."}), 400

# Initialize DB schema once when the app starts
# This ensures the table and columns are ready before the first request.
with app.app_context():
    init_enrollment_db() # From core.enrollment_processor

if __name__ == '__main__':
    # For development: host='0.0.0.0' makes it accessible on your network
    app.run(debug=True, host='0.0.0.0', port=5000)


'''
Sample Usage for enrollment api

import requests
import os

# --- Client-side code (e.g., in your PySide6 UI after capturing/selecting an image) ---

def enroll_user_via_api(name, admission_number, room, image_file_path):
    # Flask API endpoint URL
    api_url = "http://localhost:5000/api/enroll" # Or your server's IP if remote

    if not os.path.exists(image_file_path):
        print(f"Error: Image file not found at {image_file_path}")
        return None

    # Prepare the data payload (form fields)
    payload = {
        'name': name,
        'admissionNumber': admission_number,
        'room': room
    }

    # Prepare the file to be uploaded
    # 'image' here must match the key expected by request.files['image'] in Flask
    try:
        with open(image_file_path, 'rb') as f:
            files = {'image': (os.path.basename(image_file_path), f)}
            
            # Make the POST request
            response = requests.post(api_url, data=payload, files=files)
            
            # Print server response
            print(f"Status Code: {response.status_code}")
            print(f"Response JSON: {response.json()}")
            return response.json()

    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None
    except FileNotFoundError:
        print(f"Client-side error: Image file not found at {image_file_path}")
        return None
    except Exception as e:
        print(f"An unexpected client-side error occurred: {e}")
        return None


if __name__ == '__main__': # Example usage of the client function
    # Assume you have an image file for testing
    # Create a dummy image for this example to run
    dummy_image_filename = "test_enroll_image.jpg"
    if not os.path.exists(dummy_image_filename):
        import cv2
        import numpy as np
        cv2.imwrite(dummy_image_filename, np.zeros((100,100,3), dtype=np.uint8))


    print("Attempting enrollment via API...")
    result = enroll_user_via_api(
        name="John Doe (API)",
        admission_number="API001",
        room="CyberHub",
        image_file_path=dummy_image_filename # Path to the image file on the client machine
    )

    if result and result.get("status") == "success":
        print("Enrollment successful through API!")
    elif result:
        print(f"Enrollment failed through API: {result.get('message')}")
    else:
        print("API call did not return a valid response or failed.")
    
    # Clean up dummy image
    if os.path.exists(dummy_image_filename):
        os.remove(dummy_image_filename)
'''
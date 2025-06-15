from flask import Flask, request, jsonify
import os
from uuid_extensions import uuid7str
import requests as req


ENROLLMENT_IMAGES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'enrollment_images')
os.makedirs(ENROLLMENT_IMAGES_DIR, exist_ok=True)


app = Flask(__name__)

@app.route("/")
def hello_world():
    return {"message": "Hello, World!"}

@app.route("/test", methods=["GET"])
def test():
    return {"uuid": uuid7str()}

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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=1337)

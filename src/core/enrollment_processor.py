import time

import cv2
import numpy as np

import src.core.blazeface_utils as bf_utils

# (No direct config imports needed here now)
# Shared utilities
from src.core.face_utils import (
    get_mobilefacenet_model,
    load_faiss_data,
    preprocess_image_mobilefacenet,
    save_faiss_data,
)
from src.db.index import Person, db

# --- Database & FAISS Setup ---


def enroll_user(unique_id, image_path):
    """
    Enroll a user using their unique_id and image path.
    The person must already exist in the database.
    """
    # Load (or initialise) the MobileFaceNet model once
    mobilefacenet_tflite = get_mobilefacenet_model()

    # Connect to database
    if db.is_closed():
        db.connect(reuse_if_open=True)

    try:
        # Check if person exists in database
        person_data = Person.get_or_none(Person.uniqueId == unique_id)
        if not person_data:
            print(f"Error: No person found with unique_id: {unique_id}")
            return False

        print(
            f"Enrolling Person: {person_data.name} (ID: {unique_id}) from {image_path}"
        )

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return False

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # 1. Face Detection (MediaPipe)
        start_time = time.perf_counter()
        faces_found = bf_utils.detect_faces(image_rgb)
        blazeface_time = (time.perf_counter() - start_time) * 1000
        print(f"MediaPipe FaceDetector time: {blazeface_time:.2f} ms")

        if not faces_found:
            print(
                "No face detected for enrollment. Please ensure the image has a clear face."
            )
            return False

        # For enrollment, we typically take the first (or most confident) face
        best_face = faces_found[0]
        x1, y1, x2, y2 = best_face["bbox"]

        # Expand bounding box by 20% (10% per side) to match recognition behaviour
        x1, y1, x2, y2 = bf_utils.expand_bbox(
            (x1, y1, x2, y2), image_rgb.shape, fraction=0.2
        )

        # Ensure ROI is valid and not empty
        if x2 <= x1 or y2 <= y1:
            print(
                f"Warning: Invalid bounding box coordinates ({x1},{y1},{x2},{y2}). Skipping enrollment."
            )
            return False

        face_roi = image_rgb[y1:y2, x1:x2]
        if face_roi.size == 0 or face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
            print(
                "Error: Face ROI is empty after cropping. Check bounding box coordinates."
            )
            return False

        # 2. Embedding Extraction (MobileFaceNet - TFLite)
        start_time = time.perf_counter()
        input_mobilefacenet = preprocess_image_mobilefacenet(face_roi)
        embedding_output = mobilefacenet_tflite.run(input_mobilefacenet)
        embedding = embedding_output[0].flatten().astype(np.float32)
        mobilefacenet_time = (time.perf_counter() - start_time) * 1000
        print(f"MobileFaceNet inference time: {mobilefacenet_time:.2f} ms")

        # Normalize embedding to unit vector (L2 normalization)
        embedding = embedding / np.linalg.norm(embedding)

        # 3. Store in FAISS with unique_id
        # Load FAISS index and unique ID map (or initialize if first enrollment)
        faiss_index, unique_id_map = load_faiss_data()

        # Check if this unique_id is already enrolled
        if unique_id in unique_id_map:
            print(f"Warning: Person {unique_id} is already enrolled. Skipping.")
            return False

        faiss_index.add(embedding.reshape(1, -1))  # Add the embedding to FAISS
        unique_id_map.append(
            unique_id
        )  # Add the unique_id to the map, corresponding to the new FAISS index ID

        save_faiss_data(faiss_index, unique_id_map)

        print(f"Enrollment successful for {person_data.name} (unique_id: {unique_id}).")
        return True

    except Exception as e:
        print(f"Error during enrollment: {e}")
        return False
    finally:
        if not db.is_closed():
            db.close()

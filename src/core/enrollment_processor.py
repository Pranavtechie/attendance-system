import os
import cv2
import numpy as np
import sqlite3
import faiss

# --- Import from the central recognition_system ---
from .recognition_system import (
    TFLiteModel,
    preprocess_image_blazeface,
    preprocess_image_mobilefacenet,
    postprocess_blazeface_output,
    BLAZEFACE_MODEL_PATH,
    MOBILEFACENET_MODEL_PATH,
    EMBEDDING_DIM,
    MIN_DETECTION_SCORE
)

# --- Paths specific to enrollment data management (DB, FAISS index) ---
CORE_DIR_EP = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_EP = os.path.dirname(CORE_DIR_EP)

DB_PATH_EP = os.path.join(PROJECT_ROOT_EP, "attendance_system.db")
FAISS_INDEX_PATH_EP = os.path.join(CORE_DIR_EP, "faiss_index.bin") # Shared with recognition
USER_ID_MAP_PATH_EP = os.path.join(CORE_DIR_EP, "faiss_user_id_map.npy") # Shared

# --- Model Instances for Enrollment (loaded on demand) ---
_blazeface_model_enroll_instance = None
_mobilefacenet_model_enroll_instance = None

def get_enrollment_blazeface_model():
    global _blazeface_model_enroll_instance
    if _blazeface_model_enroll_instance is None:
        _blazeface_model_enroll_instance = TFLiteModel(BLAZEFACE_MODEL_PATH)
        print("BlazeFace model loaded for enrollment.")
    return _blazeface_model_enroll_instance

def get_enrollment_mobilefacenet_model():
    global _mobilefacenet_model_enroll_instance
    if _mobilefacenet_model_enroll_instance is None:
        _mobilefacenet_model_enroll_instance = TFLiteModel(MOBILEFACENET_MODEL_PATH)
        print("MobileFaceNet model loaded for enrollment.")
    return _mobilefacenet_model_enroll_instance

# --- Database Initialization (specific to enrollment needs: adding columns) ---
def init_enrollment_db():
    """Initializes the users table, ensuring new columns exist."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH_EP)
        cursor = conn.cursor()
        
        # Check current columns
        cursor.execute("PRAGMA table_info(users)")
        existing_columns = [info[1] for info in cursor.fetchall()]

        if 'admissionNumber' not in existing_columns:
            cursor.execute("ALTER TABLE users ADD COLUMN admissionNumber TEXT")
            print("Added 'admissionNumber' column to users table.")
        if 'room' not in existing_columns:
            cursor.execute("ALTER TABLE users ADD COLUMN room TEXT")
            print("Added 'room' column to users table.")

        # Create table if it doesn't exist (idempotent)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                admissionNumber TEXT,
                room TEXT
            )
        ''')
        conn.commit()
        print("Users table schema ensured for enrollment.")
    except sqlite3.Error as e: # Catch specific sqlite errors
        print(f"DB schema update/check error: {e}")
        if conn: conn.rollback()
    finally:
        if conn:
            conn.close()

# --- FAISS Data Handling for Enrollment (WRITE operations) ---
def load_faiss_for_enrollment():
    """Loads FAISS index and map for enrollment, or creates new if not found/corrupt."""
    if os.path.exists(FAISS_INDEX_PATH_EP) and os.path.exists(USER_ID_MAP_PATH_EP):
        try:
            index = faiss.read_index(FAISS_INDEX_PATH_EP)
            user_id_map_array = np.load(USER_ID_MAP_PATH_EP, allow_pickle=True)
            user_id_map = user_id_map_array.tolist() if user_id_map_array.ndim > 0 else []
            # Basic validation: if index has items, map should roughly correspond
            if index.ntotal == len(user_id_map) or (index.ntotal == 0 and not user_id_map):
                print(f"FAISS for enrollment: {index.ntotal} vectors, map: {len(user_id_map)} entries.")
                return index, user_id_map
            else:
                print(f"FAISS index/map mismatch for enrollment. Index: {index.ntotal}, Map: {len(user_id_map)}. Re-initializing.")
        except Exception as e: # Catch broad exceptions during load
            print(f"Error loading FAISS for enrollment: {e}. Creating new.")
    
    print("Creating new FAISS index (FlatIP) and map for enrollment.")
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    user_id_map = []
    return index, user_id_map

def save_faiss_after_enrollment(index, user_id_map):
    """Saves the FAISS index and user ID map after enrollment."""
    faiss.write_index(index, FAISS_INDEX_PATH_EP)
    np.save(USER_ID_MAP_PATH_EP, np.array(user_id_map, dtype=object)) # dtype=object for list of ints
    print("FAISS index and user ID map saved after enrollment update.")

# --- High-Level Enrollment Function ---
def enroll_new_user(name, admission_number, room, image_path):
    blazeface_model = get_enrollment_blazeface_model()
    mobilefacenet_model = get_enrollment_mobilefacenet_model()

    if not os.path.exists(image_path):
        return False, f"Image path does not exist: {image_path}", None
    
    image = cv2.imread(image_path)
    if image is None:
        return False, f"Could not read image from {image_path}", None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 1. Face Detection
    input_blaze = preprocess_image_blazeface(image_rgb)
    det_outputs = blazeface_model.run(input_blaze)
    # Using MIN_DETECTION_SCORE from recognition_system
    faces_found = postprocess_blazeface_output(det_outputs[0], det_outputs[1], image_rgb.shape, 
                                               score_threshold=MIN_DETECTION_SCORE)

    if not faces_found:
        return False, "No face detected in the enrollment image.", None
    if len(faces_found) > 1:
        # Sort by score and pick the highest one, or reject if strict single-face policy
        # For now, let's be strict for enrollment:
        # faces_found = sorted(faces_found, key=lambda x: x['score'], reverse=True)
        return False, "Multiple faces detected. Please use an image with a single clear face.", None
    
    best_face = faces_found[0]
    x1, y1, x2, y2 = best_face['bbox']

    if x2 <= x1 or y2 <= y1:
        return False, "Invalid bounding box detected for enrollment.", None
    face_roi = image_rgb[y1:y2, x1:x2]
    if face_roi.size == 0:
        return False, "Face ROI is empty after cropping for enrollment.", None

    # 2. Embedding Extraction
    input_mfn = preprocess_image_mobilefacenet(face_roi)
    embedding_out = mobilefacenet_model.run(input_mfn)[0].flatten().astype(np.float32)
    embedding_normalized = embedding_out / np.linalg.norm(embedding_out)

    # 3. Store in SQLite
    user_id = None
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH_EP)
        cursor = conn.cursor()
        # Attempt to insert; if name is UNIQUE and exists, it will raise IntegrityError
        cursor.execute("INSERT INTO users (name, admissionNumber, room) VALUES (?, ?, ?)",
                       (name, admission_number, room))
        user_id = cursor.lastrowid
        conn.commit()
        print(f"User '{name}' (Adm: {admission_number}) added to DB with user_id: {user_id}")
    except sqlite3.IntegrityError: # Likely due to UNIQUE constraint on name
        if conn: conn.rollback()
        print(f"User '{name}' likely already exists. Updating details and fetching ID.")
        # Need a new connection or ensure cursor is valid after rollback for further operations
        if conn: conn.close() # Close and reopen or handle cursor state carefully
        conn = sqlite3.connect(DB_PATH_EP)
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET admissionNumber = ?, room = ? WHERE name = ?",
                       (admission_number, room, name))
        conn.commit() # Commit the update
        cursor.execute("SELECT user_id FROM users WHERE name = ?", (name,))
        res = cursor.fetchone()
        if res:
            user_id = res[0]
            print(f"Updated details for existing user '{name}', user_id: {user_id}")
        else: # Should not happen if IntegrityError was on name
            if conn: conn.close()
            return False, "DB error: Failed to retrieve user_id for existing user after name conflict.", None
    except sqlite3.Error as e:
        if conn: conn.rollback()
        if conn: conn.close()
        return False, f"Database error during enrollment: {e}", None
    finally:
        if conn:
            conn.close()

    if user_id is None:
        return False, "Failed to obtain user_id from database for enrollment.", None

    # 4. Store in FAISS
    # For re-enrollment, a more robust strategy would remove old embeddings for this user_id.
    # Current simple append might lead to multiple embeddings for the same user_id in FAISS.
    faiss_idx, id_map = load_faiss_for_enrollment()
    faiss_idx.add(embedding_normalized.reshape(1, -1))
    id_map.append(int(user_id)) # Ensure user_id is int
    
    try:
        save_faiss_after_enrollment(faiss_idx, id_map)
    except Exception as e:
        # DB part was successful, but FAISS save failed. This is a partial success/failure state.
        return False, f"User info saved to DB (ID: {user_id}), but failed to save FAISS data: {e}", user_id

    return True, f"User '{name}' enrolled successfully with user_id: {user_id}", user_id
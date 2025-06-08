import os
import cv2 # For creating dummy images
import sys
import numpy as np # For creating dummy images
import uuid

# --- Add src to Python path to allow direct import from src.core ---
# This is one way to handle imports when running a script from the project root.
# Alternatively, set PYTHONPATH environment variable.
PROJECT_ROOT_SCRIPT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR_SCRIPT = os.path.join(PROJECT_ROOT_SCRIPT, "src")
if SRC_DIR_SCRIPT not in sys.path:
    sys.path.append(SRC_DIR_SCRIPT)
# --- End of sys.path modification ---

try:
    from core.enrollment_processor import enroll_new_user, init_enrollment_db_peewee
    # To access FAISS_INDEX_PATH_EP etc. if needed for manual cleanup, not typically required by test script
    from src.db.index import db as main_peewee_db, Cadet, Room, SyncValidator, CadetAttendance
except ImportError as e:
    print(f"Failed to import from src.core.enrollment_processor: {e}")
    print("Ensure you are running this script from the project root, or PYTHONPATH is set correctly.")
    print(f"Current sys.path: {sys.path}")
    exit(1)

if __name__ == "__main__":
    print("Running enrollment test script using src.core.enrollment_processor...")

    # 1. Initialize Peewee DB schema (connects, creates tables, closes)
    print("Initializing Peewee database schema (people.db)...")
    try:
        main_peewee_db.connect(reuse_if_open=True)
        main_peewee_db.create_tables([Cadet, Room, SyncValidator, CadetAttendance], safe=True)
        print("Schema ensured.")
    except Exception as e_db_init:
        print(f"Error during main DB init: {e_db_init}")
    finally:
        if not main_peewee_db.is_closed(): main_peewee_db.close()
    
    # Call the specific enrollment DB init (might be redundant if above is comprehensive)
    init_enrollment_db_peewee()

    # 2. Prepare enrollment images directory
    enrollment_images_dir = os.path.join(os.path.dirname(PROJECT_ROOT_SCRIPT), "enrollment_images")
    os.makedirs(enrollment_images_dir, exist_ok=True)


    # 3. Define test user data and image paths
    #    IMPORTANT: For real tests, use actual images with clear faces.
    #    These dummy images will likely fail face detection.
    test_users = [
        {"uid": str(uuid.uuid4()),"name": "Alice Wonderland", "adm": "ADM001", "room": "GRYFF", "img": "person1_alice.jpg"},
        {"uid": str(uuid.uuid4()),"name": "Bob The Builder", "adm": "ADM002", "room": "SLYTH", "img": "person2_bob.jpg"},
        {"uid": str(uuid.uuid4()),"name": "Pranav", "adm": "ADM003", "room": "HUFFL", "img": "person3_pranav.jpg"},
    ]

    # Create dummy images if they don't exist (these will NOT work for actual enrollment)
    for user_data in test_users:
        img_path = os.path.join(enrollment_images_dir, user_data["img"])
        print(img_path)
        if not os.path.exists(img_path):
            # Create a very basic dummy image. Real face detection will fail on this.
            dummy_cv_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            cv2.rectangle(dummy_cv_image, (50, 50), (150, 150), (0, 255, 0), 2) # "Simulate" a face area
            cv2.imwrite(img_path, dummy_cv_image)
            print(f"Created dummy image for testing: {img_path}")

    # --- Optional: Cleanup for fresh run ---
    print("\n--- Optional: Clearing FAISS and Cadet table for fresh test ---")
    try:
        if os.path.exists(os.path.join(SRC_DIR_SCRIPT, "core", "faiss_index.bin")):
            os.remove(os.path.join(SRC_DIR_SCRIPT, "core", "faiss_index.bin"))
        if os.path.exists(os.path.join(SRC_DIR_SCRIPT, "core", "faiss_user_id_map.npy")):
            os.remove(os.path.join(SRC_DIR_SCRIPT, "core", "faiss_user_id_map.npy"))
        main_peewee_db.connect(reuse_if_open=True)
        Cadet.delete().execute() # Clear all cadets
        print("FAISS files removed (if existed) and Cadet table cleared.")
    except Exception as e_clean: print(f"Error during cleanup: {e_clean}")
    finally:
        if not main_peewee_db.is_closed(): main_peewee_db.close()
    # --- End of optional cleanup ---


    # 4. Perform enrollments
    for user in test_users:
        print(f"\nEnrolling: {user['name']} (UUID: {user['uid']})")
        img_full_p = os.path.join(enrollment_images_dir, user["img"])
        success, msg, stored_uid = enroll_new_user(
            user['uid'], user['name'], user['adm'], user['room'], img_full_p
        )
        print(f"Enrollment for {user['name']}: Success={success}, Msg='{msg}', StoredID='{stored_uid}'")

    # 5. Test re-enrollment (update details for an existing user by name)
    if test_users:
        first_user = test_users[0]
        print(f"\nRe-enrolling/updating: {first_user['name']} (UUID: {first_user['uid']}) with new details")
        img_full_p_re = os.path.join(enrollment_images_dir, first_user["img"]) # Using same image for test
        success, msg, stored_uid = enroll_new_user(
            first_user['uid'], # Same UUID
            f"{first_user['name']} Updated", # New Name
            f"{first_user['adm']}_U", # New AdmNo
            f"{first_user['room']}_U", # New Room
            img_full_p_re
        )
        print(f"Re-enrollment for {first_user['name']}: Success={success}, Msg='{msg}', StoredID='{stored_uid}'")

    print("\n--- Enrollment test script with Peewee and UUIDs finished ---")
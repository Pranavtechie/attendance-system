import os
import cv2 # For creating dummy images
import sys
import numpy as np # For creating dummy images

# --- Add src to Python path to allow direct import from src.core ---
# This is one way to handle imports when running a script from the project root.
# Alternatively, set PYTHONPATH environment variable.
PROJECT_ROOT_SCRIPT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR_SCRIPT = os.path.join(PROJECT_ROOT_SCRIPT, "src")
if SRC_DIR_SCRIPT not in sys.path:
    sys.path.append(SRC_DIR_SCRIPT)
# --- End of sys.path modification ---

try:
    from core.enrollment_processor import enroll_new_user, init_enrollment_db
    # To access FAISS_INDEX_PATH_EP etc. if needed for manual cleanup, not typically required by test script
    from core.enrollment_processor import FAISS_INDEX_PATH_EP, USER_ID_MAP_PATH_EP, DB_PATH_EP
except ImportError as e:
    print(f"Failed to import from src.core.enrollment_processor: {e}")
    print("Ensure you are running this script from the project root, or PYTHONPATH is set correctly.")
    print(f"Current sys.path: {sys.path}")
    exit(1)

if __name__ == "__main__":
    print("Running enrollment test script using src.core.enrollment_processor...")

    # 1. Initialize/Update DB schema
    init_enrollment_db()

    # 2. Prepare enrollment images directory
    enrollment_images_dir = os.path.join(PROJECT_ROOT_SCRIPT, "enrollment_images")
    os.makedirs(enrollment_images_dir, exist_ok=True)

    # 3. Define test user data and image paths
    #    IMPORTANT: For real tests, use actual images with clear faces.
    #    These dummy images will likely fail face detection.
    test_users = [
        {"name": "Alice Wonderland", "adm": "ADM001", "room": "GRYFF", "img": "alice.jpg"},
        {"name": "Bob The Builder", "adm": "ADM002", "room": "SLYTH", "img": "bob.jpg"},
        {"name": "Charlie Brown", "adm": "ADM003", "room": "HUFFL", "img": "charlie.jpg"},
    ]

    # Create dummy images if they don't exist (these will NOT work for actual enrollment)
    for user_data in test_users:
        img_path = os.path.join(enrollment_images_dir, user_data["img"])
        if not os.path.exists(img_path):
            # Create a very basic dummy image. Real face detection will fail on this.
            dummy_cv_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            cv2.rectangle(dummy_cv_image, (50, 50), (150, 150), (0, 255, 0), 2) # "Simulate" a face area
            cv2.imwrite(img_path, dummy_cv_image)
            print(f"Created dummy image for testing: {img_path}")

    # --- Optional: Clean up FAISS and DB for a fresh test run ---
    # print("\n--- WARNING: Optionally clearing previous enrollment data for fresh test ---")
    # if os.path.exists(FAISS_INDEX_PATH_EP): os.remove(FAISS_INDEX_PATH_EP)
    # if os.path.exists(USER_ID_MAP_PATH_EP): os.remove(USER_ID_MAP_PATH_EP)
    # try:
    #     conn_cleanup = sqlite3.connect(DB_PATH_EP)
    #     cursor_cleanup = conn_cleanup.cursor()
    #     cursor_cleanup.execute("DELETE FROM users")
    #     conn_cleanup.commit()
    #     conn_cleanup.close()
    #     print("Cleared 'users' table in attendance_system.db.")
    # except sqlite3.Error as e:
    #     print(f"Error clearing DB: {e}")
    # print("--- End of optional cleanup ---\n")
    # --- End of optional cleanup ---


    # 4. Perform enrollments
    for user_data in test_users:
        print(f"\nAttempting to enroll: {user_data['name']}")
        image_full_path = os.path.join(enrollment_images_dir, user_data["img"])
        success, message, user_id = enroll_new_user(
            user_data["name"],
            user_data["adm"],
            user_data["room"],
            image_full_path
        )
        print(f"Enrollment for {user_data['name']}: Success={success}, Msg='{message}', UserID={user_id}")

    # 5. Test re-enrollment (update details for an existing user by name)
    print(f"\nAttempting to re-enroll/update: {test_users[0]['name']}")
    image_full_path_reenroll = os.path.join(enrollment_images_dir, test_users[0]["img"])
    success, message, user_id = enroll_new_user(
        test_users[0]["name"], # Same name
        "ADM001_NEW",         # New admission number
        "GRYFF_ANNEX",        # New room
        image_full_path_reenroll
    )
    print(f"Re-enrollment for {test_users[0]['name']}: Success={success}, Msg='{message}', UserID={user_id}")


    print("\n--- Enrollment test script finished ---")
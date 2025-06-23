from pathlib import Path

# Project root = one level above the `src` directory
ROOT_DIR = Path(__file__).resolve().parent.parent

CORE_DIR = ROOT_DIR / "src" / "core"

BLAZEFACE_MODEL_PATH = ROOT_DIR / "detector.tflite"
MOBILEFACENET_MODEL_PATH = CORE_DIR / "mobilefacenet.tflite"

DB_PATH = ROOT_DIR / "attendance_system.db"
FAISS_INDEX_PATH = ROOT_DIR / "faiss_index.bin"
USER_ID_MAP_PATH = ROOT_DIR / "faiss_user_id_map.npy"
ENROLLMENT_IMAGES_DIR = ROOT_DIR / "enrollment_images"

EMBEDDING_DIM = 192
BLAZEFACE_INPUT_SIZE = (128, 128)
MOBILEFACENET_INPUT_SIZE = (112, 112)
RECOGNITION_THRESHOLD = 0.84
MIN_DETECTION_SCORE = 0.95

SOCKET_PATH = Path("/tmp/app.sock")

# -----------------------------------------------------------------------------
# Logging configuration paths
# -----------------------------------------------------------------------------

LOG_DIR = ROOT_DIR / "logs"

# Default log file for the application (can be overridden in log_config.py)
DEFAULT_LOG_FILE = LOG_DIR / "attendance-system.log"

WAIT_TIME_AFTER_RECOGNITION_MS = 800  # Time (in milliseconds) to pause the camera feed after a successful recognition

# -----------------------------------------------------------------------------
# UI display toggles
# -----------------------------------------------------------------------------

# Set this to any Python truthy value (True, 1, "yes", etc.) to enable drawing
# bounding boxes around detected faces in the PySide6 camera preview. Set to a
# falsey value to hide them.
SHOW_FACE_BOUNDING_BOX = True

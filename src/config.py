from pathlib import Path

# Project root = one level above the `src` directory
ROOT_DIR = Path(__file__).resolve().parent.parent

CORE_DIR = ROOT_DIR / "src" / "core"

BLAZEFACE_MODEL_PATH = CORE_DIR / "face_detection_front.tflite"
MOBILEFACENET_MODEL_PATH = CORE_DIR / "mobilefacenet.tflite"

DB_PATH = ROOT_DIR / "attendance_system.db"
FAISS_INDEX_PATH = CORE_DIR / "faiss_index.bin"
USER_ID_MAP_PATH = CORE_DIR / "faiss_user_id_map.npy"
ENROLLMENT_IMAGES_DIR = ROOT_DIR / "enrollment_images"

EMBEDDING_DIM = 192
BLAZEFACE_INPUT_SIZE = (128, 128)
MOBILEFACENET_INPUT_SIZE = (112, 112)
RECOGNITION_THRESHOLD = 0.75
MIN_DETECTION_SCORE = 0.7

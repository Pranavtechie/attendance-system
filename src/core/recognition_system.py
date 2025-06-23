import logging
import os
from typing import Optional, Tuple

import cv2
import faiss
import numpy as np
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter

import src.core.blazeface_utils as bf_utils
from src.config import (
    BLAZEFACE_INPUT_SIZE,
    BLAZEFACE_MODEL_PATH,
    EMBEDDING_DIM,
    FAISS_INDEX_PATH,
    MIN_DETECTION_SCORE,
    MOBILEFACENET_INPUT_SIZE,
    MOBILEFACENET_MODEL_PATH,
    RECOGNITION_THRESHOLD,
    USER_ID_MAP_PATH,
)
from src.db.index import Person, db

# Configure logging
logger = logging.getLogger(__name__)

# define global anchors variable for BlazeFace
blaze_face_anchors = None

# --- Paths and Configuration ---
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CORE_DIR))


# --- TensorFlow Lite Interpreter Wrapper ---
class TFLiteModel:
    def __init__(self, model_path):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print(f"TFLite model '{os.path.basename(model_path)}' loaded and initialized.")

    def run(self, input_data):
        input_data = input_data.astype(self.input_details[0]["dtype"])
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        outputs = []
        for output_detail in self.output_details:
            outputs.append(self.interpreter.get_tensor(output_detail["index"]))
        return outputs


def get_person_name_from_unique_id(unique_id):
    """Get person name from unique_id using the person model."""
    try:
        if db.is_closed():
            db.connect(reuse_if_open=True)

        person = Person.get_or_none(Person.uniqueId == unique_id)
        if person:
            return person.name
        else:
            logger.warning(f"No person found with unique_id: {unique_id}")
            return "Unknown"
    except Exception as e:
        logger.error(f"Error retrieving person name for {unique_id}: {e}")
        return "Unknown"
    finally:
        if not db.is_closed():
            db.close()


def load_faiss_data():
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(USER_ID_MAP_PATH):
        try:
            index = faiss.read_index(str(FAISS_INDEX_PATH))
            arr = np.load(USER_ID_MAP_PATH, allow_pickle=True)
            unique_id_map = arr.tolist() if arr.ndim > 0 else []
            return index, unique_id_map
        except Exception:
            pass
    # initialize new
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    return index, []


def recognize_face(embedding, faiss_index, unique_id_map):
    if faiss_index.ntotal == 0:
        logger.info("No enrolled persons in FAISS index")
        return "No Enrolled Users"
    D, I = faiss_index.search(embedding.reshape(1, -1), 1)
    sim = D[0][0]
    logger.info(
        f"Cosine similarity score: {sim:.4f} (threshold: {RECOGNITION_THRESHOLD})"
    )
    if sim >= RECOGNITION_THRESHOLD:
        unique_id = unique_id_map[I[0][0]]
        person_name = get_person_name_from_unique_id(unique_id)
        logger.info(
            f"Face recognized as: {person_name} (unique_id: {unique_id}, similarity: {sim:.4f})"
        )
        return person_name
    else:
        # logger.info(
        #    f"Face not recognized - similarity {sim:.4f} below threshold {RECOGNITION_THRESHOLD}"
        # )
        return "Unknown"


class FaceRecognitionSystem:
    """High-level API for face detection and recognition"""

    def __init__(self):
        print("Initializing face recognition system...")
        self.mobilefacenet_model = TFLiteModel(MOBILEFACENET_MODEL_PATH)
        self.faiss_index, self.unique_id_map = load_faiss_data()

    def detect_and_recognize(
        self, frame_rgb
    ) -> Tuple[str, Optional[Tuple[int, int, int, int]]]:
        """Detect faces in the frame, run recognition on the most confident one,
        and return the recognised name along with its bounding box.

        Returns
        -------
        Tuple[str, Optional[Tuple[int,int,int,int]]]
            A tuple containing the detection/recognition result string and the
            bounding box (x_min, y_min, x_max, y_max). If no face was found or
            an error occurred, the bounding box will be ``None``.
        """
        try:
            original_shape = frame_rgb.shape
            # 1. Detect (MediaPipe)
            faces = bf_utils.detect_faces(frame_rgb)

            if not faces:
                return "No face detected", None

            x1, y1, x2, y2 = faces[0]["bbox"]

            # Validate bbox
            if x2 <= x1 or y2 <= y1:
                return "No face detected", None

            # ------------------------------------------------------------------
            # Expand bbox by 20% (10 % on each side)
            x1, y1, x2, y2 = bf_utils.expand_bbox(
                (x1, y1, x2, y2), frame_rgb.shape, fraction=0.2
            )

            # 2. Embed
            roi = frame_rgb[y1:y2, x1:x2]
            if roi.size == 0:
                return "No face detected", None

            inp2 = preprocess_image_mobilefacenet(roi)
            emb_out = self.mobilefacenet_model.run(inp2)[0].flatten().astype(np.float32)
            emb = emb_out / np.linalg.norm(emb_out)

            # Refresh FAISS so we always see new enrolments
            self.faiss_index, self.unique_id_map = load_faiss_data()

            # 3. Recognize
            name = recognize_face(emb, self.faiss_index, self.unique_id_map)

            return name, (x1, y1, x2, y2)
        except Exception as e:
            logger.error(f"Error in detection/recognition: {e}")
            return "Detection error", None


# ------------------ Image preprocessing ------------------


def preprocess_image_mobilefacenet(face_roi):
    """Resize and normalise an RGB face crop for MobileFaceNet."""
    img = cv2.resize(face_roi, MOBILEFACENET_INPUT_SIZE, interpolation=cv2.INTER_AREA)
    img = (img.astype(np.float32) / 127.5) - 1.0
    return img[np.newaxis, :, :, :]

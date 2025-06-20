import logging
import os

import cv2
import faiss
import numpy as np
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter

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

# BLAZEFACE_MODEL_PATH = os.path.join(CORE_DIR, "face_detection_front.tflite")
# MOBILEFACENET_MODEL_PATH = os.path.join(CORE_DIR, "mobilefacenet.tflite")
# DB_PATH = os.path.join(PROJECT_ROOT, "attendance_system.db")
# FAISS_INDEX_PATH = os.path.join(CORE_DIR, "faiss_index.bin")
# USER_ID_MAP_PATH = os.path.join(CORE_DIR, "faiss_user_id_map.npy")

# EMBEDDING_DIM = 192  # Based on MobileFaceNet output
# BLAZEFACE_INPUT_SIZE = (128, 128)  # W, H
# MOBILEFACENET_INPUT_SIZE = (112, 112)  # W, H

# --- Recognition Thresholds ---
# RECOGNITION_THRESHOLD = 0.75  # Cosine similarity threshold for recognition
# MIN_DETECTION_SCORE = 0.7     # Minimum score for BlazeFace detection


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


# AnchorBoxes and post-processing for BlazeFace
class AnchorBoxes:
    def __init__(
        self, input_size=BLAZEFACE_INPUT_SIZE, min_scale=0.1484375, max_scale=0.75
    ):
        self.input_size = input_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.feature_map_sizes = [(16, 16), (8, 8), (4, 4), (2, 2)]
        self.num_anchors_per_location = [2, 2, 2, 56]
        self.strides = [8, 16, 32, 64]
        self.anchors = self._generate_anchors()

    def _generate_anchors(self):
        anchors = []
        num_layers = len(self.feature_map_sizes)
        scales = [
            self.min_scale + (self.max_scale - self.min_scale) * i / (num_layers - 1)
            for i in range(num_layers)
        ]
        scales[num_layers - 1] = (scales[num_layers - 1] + 1.0) / 2.0

        for layer_idx, (fm_h, fm_w) in enumerate(self.feature_map_sizes):
            for y in range(fm_h):
                for x in range(fm_w):
                    center_x = (x + 0.5) / fm_w
                    center_y = (y + 0.5) / fm_h

                    if layer_idx < 3:
                        scale_base = scales[layer_idx]
                        anchors.append([center_x, center_y, scale_base, scale_base])
                        anchors.append(
                            [center_x, center_y, scale_base * 1.2, scale_base * 1.2]
                        )
                    else:
                        for k in range(self.num_anchors_per_location[layer_idx]):
                            w = scales[layer_idx] * (1.0 + k * 0.0001)
                            h = scales[layer_idx] * (1.0 + k * 0.0001)
                            anchors.append([center_x, center_y, w, h])

        final_anchors = np.array(anchors, dtype=np.float32)
        if final_anchors.shape[0] != 896:
            raise ValueError(
                f"Anchor generation failed: Expected 896 anchors, but got {final_anchors.shape[0]}."
            )
        return final_anchors


def _decode_boxes(raw_boxes, anchors):
    if raw_boxes.shape[0] != anchors.shape[0]:
        raise ValueError("Shape mismatch between raw_boxes and anchors.")

    decoded_boxes = np.zeros_like(anchors)
    clamped_dh = np.clip(raw_boxes[:, 2], -10.0, 10.0)
    clamped_dw = np.clip(raw_boxes[:, 3], -10.0, 10.0)

    decoded_boxes[:, 0] = raw_boxes[:, 0] * anchors[:, 3] + anchors[:, 1]
    decoded_boxes[:, 1] = raw_boxes[:, 1] * anchors[:, 2] + anchors[:, 0]
    decoded_boxes[:, 2] = np.exp(clamped_dh) * anchors[:, 3]
    decoded_boxes[:, 3] = np.exp(clamped_dw) * anchors[:, 2]

    # Convert to ymin, xmin, ymax, xmax format
    boxes_yx = np.zeros_like(anchors)
    boxes_yx[:, 0] = decoded_boxes[:, 0] - decoded_boxes[:, 2] / 2
    boxes_yx[:, 1] = decoded_boxes[:, 1] - decoded_boxes[:, 3] / 2
    boxes_yx[:, 2] = decoded_boxes[:, 0] + decoded_boxes[:, 2] / 2
    boxes_yx[:, 3] = decoded_boxes[:, 1] + decoded_boxes[:, 3] / 2
    boxes_yx = np.clip(boxes_yx, 0.0, 1.0)

    # Landmarks (not used in recognition but kept for completeness)
    num_landmarks = 6
    landmarks = np.zeros((raw_boxes.shape[0], num_landmarks * 2), dtype=np.float32)
    for i in range(num_landmarks):
        landmarks[:, i * 2] = raw_boxes[:, 4 + i * 2] * anchors[:, 2] + anchors[:, 0]
        landmarks[:, i * 2 + 1] = (
            raw_boxes[:, 4 + i * 2 + 1] * anchors[:, 3] + anchors[:, 1]
        )
    landmarks = np.clip(landmarks, 0.0, 1.0)

    return boxes_yx, landmarks


def non_max_suppression(boxes, scores, threshold):
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 1]
    y1 = boxes[:, 0]
    x2 = boxes[:, 3]
    y2 = boxes[:, 2]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = np.zeros_like(union)
        mask = union > 1e-6
        iou[mask] = inter[mask] / union[mask]
        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]
    return keep


def preprocess_image_blazeface(frame):
    img = cv2.resize(frame, BLAZEFACE_INPUT_SIZE, interpolation=cv2.INTER_AREA)
    img = (img.astype(np.float32) / 127.5) - 1.0
    return img[np.newaxis, :, :, :]


def preprocess_image_mobilefacenet(face_roi):
    img = cv2.resize(face_roi, MOBILEFACENET_INPUT_SIZE, interpolation=cv2.INTER_AREA)
    img = (img.astype(np.float32) / 127.5) - 1.0
    return img[np.newaxis, :, :, :]


def postprocess_blazeface_output(
    regressors,
    classificators,
    original_image_shape,
    score_threshold=MIN_DETECTION_SCORE,
    iou_threshold=0.3,
):
    global blaze_face_anchors
    if blaze_face_anchors is None:
        blaze_face_anchors = AnchorBoxes()

    raw_regressors = regressors[0]
    raw_classificators = classificators[0]
    decoded_boxes, decoded_landmarks = _decode_boxes(
        raw_regressors, blaze_face_anchors.anchors
    )
    scores = tf.sigmoid(raw_classificators[:, 0]).numpy()

    # Log detection score statistics
    max_score = np.max(scores) if len(scores) > 0 else 0
    # logger.info(
    #    f"BlazeFace detection - Max score: {max_score:.4f} (threshold: {score_threshold})"
    # )

    mask = scores >= score_threshold
    valid_detections = np.sum(mask)
    logger.info(f"Valid detections above threshold: {valid_detections}/{len(scores)}")

    boxes_f = decoded_boxes[mask]
    lands_f = decoded_landmarks[mask]
    scores_f = scores[mask]
    keep = non_max_suppression(boxes_f, scores_f, iou_threshold)
    final_boxes = boxes_f[keep]
    final_landmarks = lands_f[keep]
    final_scores = scores_f[keep]
    h, w, _ = original_image_shape
    results = []
    for i, score in enumerate(final_scores):
        b = final_boxes[i]
        x_min = int(b[1] * w)
        y_min = int(b[0] * h)
        x_max = int(b[3] * w)
        y_max = int(b[2] * h)
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w - 1, x_max), min(h - 1, y_max)
        # logger.info(
        #    f"Face detection #{i}: score={score:.4f}, bbox=({x_min},{y_min},{x_max},{y_max})"
        # )
        results.append({"bbox": (x_min, y_min, x_max, y_max), "score": score})
    return results


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
        self.blazeface_model = TFLiteModel(BLAZEFACE_MODEL_PATH)
        self.mobilefacenet_model = TFLiteModel(MOBILEFACENET_MODEL_PATH)
        self.faiss_index, self.unique_id_map = load_faiss_data()

    def detect_and_recognize(self, frame_rgb):
        try:
            original_shape = frame_rgb.shape
            # logger.info(f"Processing frame of shape: {original_shape}")
            # 1. Detect
            inp = preprocess_image_blazeface(frame_rgb)
            regs, clss = self.blazeface_model.run(inp)
            faces = postprocess_blazeface_output(regs, clss, original_shape)
            if not faces:
                # logger.info("No faces detected in frame")
                return "No face detected"
            x1, y1, x2, y2 = faces[0]["bbox"]
            detection_score = faces[0]["score"]
            # logger.info(f"Using best face detection with score: {detection_score:.4f}")
            if x2 <= x1 or y2 <= y1:
                # logger.warning("Invalid bounding box detected")
                return "No face detected"
            roi = frame_rgb[y1:y2, x1:x2]
            if roi.size == 0:
                # logger.warning("Empty ROI extracted")
                return "No face detected"
            # logger.info(f"Extracted face ROI of size: {roi.shape}")
            # 2. Embed
            inp2 = preprocess_image_mobilefacenet(roi)
            emb_out = self.mobilefacenet_model.run(inp2)[0].flatten().astype(np.float32)
            emb = emb_out / np.linalg.norm(emb_out)
            # logger.info(f"Generated embedding with norm: {np.linalg.norm(emb):.4f}")

            # ---- Refresh FAISS so we always see new enrolments ----
            self.faiss_index, self.unique_id_map = load_faiss_data()

            # 3. Recognize
            name = recognize_face(emb, self.faiss_index, self.unique_id_map)
            return name
        except Exception as e:
            logger.error(f"Error in detection/recognition: {e}")
            return "Detection error"

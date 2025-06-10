import os
import cv2
import numpy as np
import faiss
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter
from src.db.index import db as peewee_db_rs, Cadet


# define global anchors variable for BlazeFace
blaze_face_anchors = None

# With a proper project setup (pyproject.toml and editable install),
# we don't need to manipulate sys.path anymore.


# --- Paths and Configuration ---
CORE_DIR = os.path.dirname(os.path.abspath(__file__))

BLAZEFACE_MODEL_PATH = os.path.join(CORE_DIR, "face_detection_front.tflite")
MOBILEFACENET_MODEL_PATH = os.path.join(CORE_DIR, "mobilefacenet.tflite")
FAISS_INDEX_PATH = os.path.join(CORE_DIR, "faiss_index.bin")
USER_ID_MAP_PATH = os.path.join(CORE_DIR, "faiss_user_id_map.npy")

EMBEDDING_DIM = 192  # Based on MobileFaceNet output
BLAZEFACE_INPUT_SIZE = (128, 128)  # W, H
MOBILEFACENET_INPUT_SIZE = (112, 112)  # W, H

# --- Recognition Thresholds ---
RECOGNITION_THRESHOLD = 0.75  # Cosine similarity threshold for recognition
MIN_DETECTION_SCORE = 0.7  # Minimum score for BlazeFace detection


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


# --- Preprocessing Functions (Exportable) ---
def preprocess_image_blazeface(frame):
    img = cv2.resize(frame, BLAZEFACE_INPUT_SIZE, interpolation=cv2.INTER_AREA)
    img = (img.astype(np.float32) / 127.5) - 1.0
    return img[np.newaxis, :, :, :]


def preprocess_image_mobilefacenet(face_roi):
    img = cv2.resize(face_roi, MOBILEFACENET_INPUT_SIZE, interpolation=cv2.INTER_AREA)
    img = (img.astype(np.float32) / 127.5) - 1.0
    return img[np.newaxis, :, :, :]


# --- Post-processing for BlazeFace (Exportable) ---
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

    decoded_boxes, _ = _decode_boxes(
        raw_regressors, blaze_face_anchors.anchors
    )
    scores = tf.sigmoid(raw_classificators[:, 0]).numpy()

    mask = scores >= score_threshold
    boxes_f = decoded_boxes[mask]
    scores_f = scores[mask]

    keep = non_max_suppression(boxes_f, scores_f, iou_threshold)
    final_boxes = boxes_f[keep]
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
        results.append({"bbox": (x_min, y_min, x_max, y_max), "score": score})
    return results


def get_user_name_from_id(user_id):
    name = "Unknown"
    try:
        peewee_db_rs.connect(reuse_if_open=True)
        cadet = Cadet.get_or_none(Cadet.uniqueId == user_id)
        if cadet:
            name = cadet.name
    except Exception as e:
        print(f"Peewee DB err fetch name for UUID {user_id}: {e}")
    finally:
        if not peewee_db_rs.is_closed():
            peewee_db_rs.close()
    return name


# --- FAISS and DB Utilities for Recognition (READ-ONLY for FAISS) ---
def load_faiss_for_recognition():
    """Loads FAISS index and user map for recognition."""
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(USER_ID_MAP_PATH):
        try:
            index = faiss.read_index(FAISS_INDEX_PATH)
            user_id_map_array = np.load(USER_ID_MAP_PATH, allow_pickle=True)
            user_id_map = (
                user_id_map_array.tolist() if user_id_map_array.ndim > 0 else []
            )
            if index.ntotal == len(user_id_map) or (
                index.ntotal == 0 and not user_id_map
            ):  # Basic check
                print(
                    f"FAISS index for recognition: {index.ntotal} vectors, map: {len(user_id_map)} entries."
                )
                return index, user_id_map
            else:
                print(
                    f"Warning: FAISS index size ({index.ntotal}) and map size ({len(user_id_map)}) mismatch. Re-initializing."
                )
        except Exception as e:
            print(f"Error loading FAISS for recognition: {e}. Re-initializing.")

    print("FAISS index/map not found or corrupted for recognition. Initializing empty.")
    # Return empty structures, recognition will report "No Enrolled Users"
    index = faiss.IndexFlatIP(EMBEDDING_DIM)  # Must match enrollment's index type
    user_id_map = []
    return index, user_id_map


class FaceRecognitionSystem:
    """High-level API for face detection and recognition"""

    def __init__(self):
        print("Initializing face recognition system...")
        self.blazeface_model = TFLiteModel(BLAZEFACE_MODEL_PATH)
        self.mobilefacenet_model = TFLiteModel(MOBILEFACENET_MODEL_PATH)
        self.faiss_index, self.user_id_map = load_faiss_for_recognition()

    def detect_and_recognize(self, frame_rgb):
        original_shape = frame_rgb.shape

        # 1. Face Detection
        input_blaze = preprocess_image_blazeface(frame_rgb)
        raw_regs, raw_clss = self.blazeface_model.run(input_blaze)
        faces = postprocess_blazeface_output(
            raw_regs, raw_clss, original_shape, score_threshold=MIN_DETECTION_SCORE
        )

        if not faces:
            return "No face detected"  # Or return a more structured response

        # For simplicity, process the first (or highest score) detected face
        # A more robust system might handle multiple faces with tracking or selection logic
        best_face = sorted(faces, key=lambda x: x["score"], reverse=True)[0]
        x1, y1, x2, y2 = best_face["bbox"]

        if x2 <= x1 or y2 <= y1:
            return "Invalid face ROI"

        face_roi_rgb = frame_rgb[y1:y2, x1:x2]
        if face_roi_rgb.size == 0:
            return "Empty face ROI"

        # 2. Embedding Extraction
        input_mfn = preprocess_image_mobilefacenet(face_roi_rgb)
        embedding_output = (
            self.mobilefacenet_model.run(input_mfn)[0].flatten().astype(np.float32)
        )
        embedding_normalized = embedding_output / np.linalg.norm(embedding_output)

        # 3. Recognize against FAISS
        if self.faiss_index.ntotal == 0:
            return "No Enrolled Users"

        # Search FAISS (k=1 for the single best match)
        # FAISS returns distances (D) and indices (I)
        # For IndexFlatIP (Inner Product), higher D means higher similarity (cosine similarity)
        distances, indices = self.faiss_index.search(
            embedding_normalized.reshape(1, -1), 1
        )

        similarity_score = distances[0][0]
        faiss_internal_idx = indices[0][0]

        if similarity_score >= RECOGNITION_THRESHOLD:
            if 0 <= faiss_internal_idx < len(self.user_id_map):
                user_id = self.user_id_map[faiss_internal_idx]
                user_name = get_user_name_from_id(user_id)
                # print(f"Recognized: {user_name} (ID: {user_id}), Score: {similarity_score:.4f}")
                return user_name
            else:
                print(
                    f"Error: FAISS index {faiss_internal_idx} out of bounds for user_id_map (len {len(self.user_id_map)})."
                )
                return "Map Index Error"
        else:
            # print(f"Unknown face, Max Score: {similarity_score:.4f}")
            return "Unknown"


if __name__ == "__main__":
    print("Running recognition_system.py self-test...")
    try:
        # This requires people.db to be initialized with schema for Cadet
        # Call the db init from src.db.index if needed for test
        from src.db.index import (
            db as main_db_for_test,
            Cadet as Cadet_for_test,
            Room as Room_for_test,
            SyncValidator as SV_for_test,
            CadetAttendance as CA_for_test,
        )

        main_db_for_test.connect(reuse_if_open=True)
        main_db_for_test.create_tables(
            [Cadet_for_test, Room_for_test, SV_for_test, CA_for_test], safe=True
        )
        main_db_for_test.close()
        print("people.db schema ensured for self-test.")

        system = FaceRecognitionSystem()
        print("FaceRecognitionSystem instantiated.")
        dummy_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = system.detect_and_recognize(dummy_frame)
        print(f"Recognition on dummy frame: {result}")
    except Exception as e:
        print(f"Self-test error: {e}")

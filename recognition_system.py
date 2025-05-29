import cv2
import numpy as np
import faiss
import sqlite3
import os
import time
import tensorflow as tf

# --- Configuration (MUST match enrollment_system.py) ---
BLAZEFACE_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_detection_front.tflite")
MOBILEFACENET_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mobilefacenet.tflite")
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attendance_system.db")
FAISS_INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_index.bin")
USER_ID_MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_user_id_map.npy")

EMBEDDING_DIM = 192 # Based on MobileFaceNet output
BLAZEFACE_INPUT_SIZE = (128, 128) # W, H
MOBILEFACENET_INPUT_SIZE = (112, 112) # W, H

# --- Recognition Thresholds ---
RECOGNITION_THRESHOLD = 0.65 # Cosine similarity threshold for recognition (adjust as needed)
MIN_DETECTION_SCORE = 0.7   # Minimum score for BlazeFace detection to be considered a face

# --- TensorFlow Lite Interpreter Loaders ---
class TFLiteModel:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print(f"TFLite model '{os.path.basename(model_path)}' loaded and initialized.")

    def run(self, input_data):
        input_data = input_data.astype(self.input_details[0]['dtype'])
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        outputs = []
        for output_detail in self.output_details:
            outputs.append(self.interpreter.get_tensor(output_detail['index']))
        return outputs

# Global variables for TFLite models and AnchorBoxes
blazeface_tflite = None
mobilefacenet_tflite = None
blaze_face_anchors = None

def load_tflite_models():
    global blazeface_tflite, mobilefacenet_tflite
    if blazeface_tflite is None:
        blazeface_tflite = TFLiteModel(BLAZEFACE_MODEL_PATH)
    if mobilefacenet_tflite is None:
        mobilefacenet_tflite = TFLiteModel(MOBILEFACENET_MODEL_PATH)

# --- Face Preprocessing and Post-processing Functions ---

def preprocess_image_blazeface(frame):
    img_resized = cv2.resize(frame, BLAZEFACE_INPUT_SIZE, interpolation=cv2.INTER_AREA)
    img_normalized = (img_resized.astype(np.float32) / 127.5) - 1.0
    return img_normalized[np.newaxis, :, :, :]

def preprocess_image_mobilefacenet(face_roi):
    img_resized = cv2.resize(face_roi, MOBILEFACENET_INPUT_SIZE, interpolation=cv2.INTER_AREA)
    img_normalized = (img_resized.astype(np.float32) / 127.5) - 1.0
    return img_normalized[np.newaxis, :, :, :]

class AnchorBoxes:
    def __init__(self, input_size=(128, 128), min_scale=0.1484375, max_scale=0.75):
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
        scales = [self.min_scale + (self.max_scale - self.min_scale) * i / (num_layers - 1)
                  for i in range(num_layers)]
        scales[num_layers - 1] = (scales[num_layers - 1] + 1.0) / 2.0

        for layer_idx, (fm_h, fm_w) in enumerate(self.feature_map_sizes):
            for y in range(fm_h):
                for x in range(fm_w):
                    center_x = (x + 0.5) / fm_w
                    center_y = (y + 0.5) / fm_h

                    if layer_idx < 3:
                        scale_base = scales[layer_idx]
                        scale_var_1 = scale_base
                        scale_var_2 = scale_base * 1.2
                        anchors.append([center_x, center_y, scale_var_1, scale_var_1])
                        anchors.append([center_x, center_y, scale_var_2, scale_var_2])
                    else:
                        for k in range(self.num_anchors_per_location[layer_idx]):
                            w = scales[layer_idx] * (1.0 + k * 0.0001)
                            h = scales[layer_idx] * (1.0 + k * 0.0001)
                            anchors.append([center_x, center_y, w, h])

        final_anchors = np.array(anchors, dtype=np.float32)
        if final_anchors.shape[0] != 896:
            raise ValueError(f"Anchor generation failed: Expected 896 anchors for face_detection_front, but got {final_anchors.shape[0]}.")
        return final_anchors

def _decode_boxes(raw_boxes, anchors):
    if raw_boxes.shape[0] != anchors.shape[0]:
        raise ValueError(f"Shape mismatch: raw_boxes has {raw_boxes.shape[0]} rows, but anchors has {anchors.shape[0]} rows. They must match for decoding.")

    decoded_boxes = np.zeros_like(anchors)

    clamped_dh = np.clip(raw_boxes[:, 2], -10.0, 10.0)
    clamped_dw = np.clip(raw_boxes[:, 3], -10.0, 10.0)

    decoded_boxes[:, 0] = raw_boxes[:, 0] * anchors[:, 3] + anchors[:, 1]
    decoded_boxes[:, 1] = raw_boxes[:, 1] * anchors[:, 2] + anchors[:, 0]
    decoded_boxes[:, 2] = np.exp(clamped_dh) * anchors[:, 3]
    decoded_boxes[:, 3] = np.exp(clamped_dw) * anchors[:, 2]

    boxes_yx_format = np.zeros_like(anchors)
    boxes_yx_format[:, 0] = decoded_boxes[:, 0] - decoded_boxes[:, 2] / 2
    boxes_yx_format[:, 1] = decoded_boxes[:, 1] - decoded_boxes[:, 3] / 2
    boxes_yx_format[:, 2] = decoded_boxes[:, 0] + decoded_boxes[:, 2] / 2
    boxes_yx_format[:, 3] = decoded_boxes[:, 1] + decoded_boxes[:, 3] / 2

    boxes_yx_format = np.clip(boxes_yx_format, 0.0, 1.0)

    num_landmarks = 6
    landmarks = np.zeros((raw_boxes.shape[0], num_landmarks * 2), dtype=np.float32)

    for i in range(num_landmarks):
        landmarks[:, i*2] = raw_boxes[:, 4 + i*2] * anchors[:, 2] + anchors[:, 0]
        landmarks[:, i*2 + 1] = raw_boxes[:, 4 + i*2 + 1] * anchors[:, 3] + anchors[:, 1]

    landmarks = np.clip(landmarks, 0.0, 1.0)

    return boxes_yx_format, landmarks

def non_max_suppression(boxes, scores, threshold):
    """
    Performs Non-Maximum Suppression (NMS).
    boxes: numpy array of shape (N, 4) with [ymin, xmin, ymax, xmax] (normalized)
    scores: numpy array of shape (N,)
    threshold: IoU threshold
    """
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

        # If there's only one element left, there's nothing to compare against for IoU
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
        iou = np.zeros_like(union, dtype=np.float32)
        non_zero_union_mask = union > 1e-6
        iou[non_zero_union_mask] = inter[non_zero_union_mask] / union[non_zero_union_mask]

        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]

    return keep

def postprocess_blazeface_output(regressors, classificators, original_image_shape, score_threshold=0.7, iou_threshold=0.3):
    global blaze_face_anchors
    if blaze_face_anchors is None:
        blaze_face_anchors = AnchorBoxes(input_size=BLAZEFACE_INPUT_SIZE,
                                         min_scale=0.1484375, max_scale=0.75)

    raw_regressors = regressors[0]
    raw_classificators = classificators[0]

    decoded_boxes_normalized, decoded_landmarks_normalized = _decode_boxes(raw_regressors, blaze_face_anchors.anchors)

    scores = tf.sigmoid(raw_classificators[:, 0]).numpy()

    mask = scores >= score_threshold
    filtered_boxes_normalized = decoded_boxes_normalized[mask]
    filtered_landmarks_normalized = decoded_landmarks_normalized[mask]
    filtered_scores = scores[mask]

    keep_indices = non_max_suppression(filtered_boxes_normalized, filtered_scores, iou_threshold)

    final_boxes_normalized = filtered_boxes_normalized[keep_indices]
    final_landmarks_normalized = filtered_landmarks_normalized[keep_indices]
    final_scores = filtered_scores[keep_indices]

    h_orig, w_orig, _ = original_image_shape

    detected_faces = []
    for i in range(len(final_scores)):
        score = final_scores[i]
        box_norm = final_boxes_normalized[i]
        landmarks_norm = final_landmarks_normalized[i]

        x_min = int(box_norm[1] * w_orig)
        y_min = int(box_norm[0] * h_orig)
        x_max = int(box_norm[3] * w_orig)
        y_max = int(box_norm[2] * h_orig)

        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w_orig - 1, x_max), min(h_orig - 1, y_max)

        scaled_landmarks = np.zeros_like(landmarks_norm)
        for j in range(0, len(landmarks_norm), 2):
            scaled_landmarks[j] = int(landmarks_norm[j] * w_orig)
            scaled_landmarks[j+1] = int(landmarks_norm[j+1] * h_orig)

        detected_faces.append({
            'bbox': (x_min, y_min, x_max, y_max),
            'score': score,
            'landmarks': scaled_landmarks.tolist()
        })

    return detected_faces

# --- Database & FAISS Setup ---
def get_user_name_from_id(user_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM users WHERE user_id = ?", (user_id,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else "Unknown"

def load_faiss_data():
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(USER_ID_MAP_PATH):
        try:
            index = faiss.read_index(FAISS_INDEX_PATH)
            user_id_map_array = np.load(USER_ID_MAP_PATH)
            user_id_map = user_id_map_array.tolist() if user_id_map_array.ndim > 0 else []
            print(f"FAISS index loaded with {index.ntotal} vectors.")
            print(f"User ID map loaded with {len(user_id_map)} entries.")
            return index, user_id_map
        except Exception as e:
            print(f"Error loading FAISS data: {e}. Re-initializing.")
            index = faiss.IndexFlatIP(EMBEDDING_DIM)
            user_id_map = []
            return index, user_id_map
    else:
        print("FAISS index or user ID map not found. Initializing new ones.")
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        user_id_map = []
        return index, user_id_map

# --- Real-time Recognition Logic ---
def recognize_face(embedding, faiss_index, user_id_map):
    if faiss_index.ntotal == 0:
        print("Debug: No enrolled users in FAISS index.")
        return "No Enrolled Users"

    # Search FAISS index for the closest match
    D, I = faiss_index.search(embedding.reshape(1, -1), 1) # Search for 1 nearest neighbor

    distance = D[0][0]
    matched_index_in_faiss = I[0][0]

    # Since we use IndexFlatIP with L2 normalized embeddings, this is directly cosine similarity.
    similarity = distance

    # Debug prints to help you understand the similarity score
    print(f"Debug: Nearest match similarity: {similarity:.4f}")
    print(f"Debug: Recognition threshold: {RECOGNITION_THRESHOLD:.4f}")


    if similarity >= RECOGNITION_THRESHOLD:
        matched_user_id = user_id_map[matched_index_in_faiss]
        user_name = get_user_name_from_id(matched_user_id)
        print(f"Debug: Recognized as {user_name} (ID: {matched_user_id})")
        return user_name
    else:
        print("Debug: Classification: Unknown")
        return "Unknown"

def start_recognition_stream():
    load_tflite_models() # Ensure models are loaded
    faiss_index, user_id_map = load_faiss_data() # Load FAISS data once at start

    cap = cv2.VideoCapture(0) # 0 for default webcam, change if you have multiple
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting real-time face recognition. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        original_frame_shape = frame.shape

        # 1. Face Detection
        input_blazeface = preprocess_image_blazeface(frame_rgb)
        detection_outputs = blazeface_tflite.run(input_blazeface)
        regressors = detection_outputs[0]
        classificators = detection_outputs[1]

        faces_found = postprocess_blazeface_output(
            regressors, classificators, original_frame_shape,
            score_threshold=MIN_DETECTION_SCORE
        )

        for face in faces_found:
            x1, y1, x2, y2 = face['bbox']

            # Ensure ROI is valid before cropping
            if x2 <= x1 or y2 <= y1:
                continue

            face_roi = frame_rgb[y1:y2, x1:x2]

            if face_roi.size == 0 or face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
                continue

            # 2. Embedding Extraction
            input_mobilefacenet = preprocess_image_mobilefacenet(face_roi)
            embedding_output = mobilefacenet_tflite.run(input_mobilefacenet)
            embedding = embedding_output[0].flatten().astype(np.float32)

            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)

            # 3. Face Recognition
            recognized_name = recognize_face(embedding, faiss_index, user_id_map)

            # Draw bounding box and label
            color = (0, 255, 0) if recognized_name != "Unknown" and recognized_name != "No Enrolled Users" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Position text carefully:
            # Ensure text is visible even if box is near top edge
            text_y_position = max(30, y1 - 10) # 30 pixels from top, or 10 pixels above box
            font_scale = 0.9 # You can try 1.0 or 1.2 for larger text
            thickness = 2

            cv2.putText(frame, recognized_name, (x1, text_y_position), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

        # Display the frame
        cv2.imshow('Face Recognition', frame)

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Recognition stream stopped.")


if __name__ == "__main__":
    start_recognition_stream()
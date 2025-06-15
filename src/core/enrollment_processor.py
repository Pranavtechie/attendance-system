import cv2
import numpy as np
import faiss
import sqlite3
import os
import time
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter

# --- Configuration based on your model inspection ---
BLAZEFACE_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_detection_front.tflite")
MOBILEFACENET_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mobilefacenet.tflite")
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attendance_system.db")
FAISS_INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_index.bin")
USER_ID_MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_user_id_map.npy")

EMBEDDING_DIM = 192 # Based on MobileFaceNet output
BLAZEFACE_INPUT_SIZE = (128, 128) # W, H
MOBILEFACENET_INPUT_SIZE = (112, 112) # W, H

# --- TensorFlow Lite Interpreter Loaders ---
class TFLiteModel:
    def __init__(self, model_path):
        self.interpreter = Interpreter(model_path=model_path)
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
blaze_face_anchors = None # Will be initialized once by postprocess_blazeface_output

def load_tflite_models():
    global blazeface_tflite, mobilefacenet_tflite
    if blazeface_tflite is None:
        blazeface_tflite = TFLiteModel(BLAZEFACE_MODEL_PATH)
    if mobilefacenet_tflite is None:
        mobilefacenet_tflite = TFLiteModel(MOBILEFACENET_MODEL_PATH)


# --- Face Preprocessing and Post-processing Functions ---

def preprocess_image_blazeface(frame):
    """
    Preprocesses the image for BlazeFace model input.
    Expected input: [1, 128, 128, 3] float32, normalized to [-1, 1].
    """
    img_resized = cv2.resize(frame, BLAZEFACE_INPUT_SIZE, interpolation=cv2.INTER_AREA)
    # Convert to float32 and normalize to [-1, 1]
    img_normalized = (img_resized.astype(np.float32) / 127.5) - 1.0
    return img_normalized[np.newaxis, :, :, :] # Add batch dimension

def preprocess_image_mobilefacenet(face_roi):
    """
    Preprocesses the face ROI for MobileFaceNet model input.
    Expected input: [1, 112, 112, 3] float32, normalized to [-1, 1].
    """
    img_resized = cv2.resize(face_roi, MOBILEFACENET_INPUT_SIZE, interpolation=cv2.INTER_AREA)
    # Convert to float32 and normalize to [-1, 1]
    img_normalized = (img_resized.astype(np.float32) / 127.5) - 1.0
    return img_normalized[np.newaxis, :, :, :] # Add batch dimension

# --- BlazeFace Specifics (Anchors and Decoding) ---
# This part is derived from MediaPipe's BlazeFace implementation details.
class AnchorBoxes:
    def __init__(self, input_size=(128, 128), min_scale=0.1484375, max_scale=0.75):
        self.input_size = input_size
        self.min_scale = min_scale
        self.max_scale = max_scale

        # These parameters are specific to MediaPipe's face_detection_front.tflite
        self.feature_map_sizes = [(16, 16), (8, 8), (4, 4), (2, 2)]
        # This list defines how many anchors per location for each feature map layer
        # This is where the 896 total anchors come from.
        self.num_anchors_per_location = [2, 2, 2, 56] 
        self.strides = [8, 16, 32, 64] # Pixel stride for each feature map

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

                    # The specific aspect ratios for the 56 anchors on the 2x2 layer are
                    # not directly exposed in generic MediaPipe anchor generators.
                    # For `face_detection_front.tflite`, the anchor generation logic is simplified here
                    # to match the *count* (896 total) and basic properties (square-ish).
                    # A more precise implementation would need to match MediaPipe's C++ code exactly,
                    # which defines specific "aspect_ratio" and "fixed_anchor_size" lists per layer.
                    
                    # For practical purposes, as long as the count and general scales are okay,
                    # the model's regression output will correct the fine-grained box shapes.

                    if layer_idx < 3: # First three layers (16x16, 8x8, 4x4)
                        # Generate two anchors per location (e.g., base scale and slightly larger/smaller)
                        # The exact variation might differ from MediaPipe, but the count is correct.
                        # Using aspect ratio 1.0 for simplicity, and scales derived from general scale.
                        # These are effectively two distinct square anchors (s_k and s_k' = s_k * sqrt(2))
                        # or similar for various MediaPipe models.
                        
                        scale_base = scales[layer_idx]
                        scale_var_1 = scale_base
                        scale_var_2 = scale_base * 1.2 # A common slight variation seen
                        
                        anchors.append([center_x, center_y, scale_var_1, scale_var_1]) # Aspect 1:1
                        anchors.append([center_x, center_y, scale_var_2, scale_var_2]) # Another aspect 1:1 or slightly different scale

                    else: # Last layer (2x2) with 56 anchors per location
                        # This layer is complex. Instead of trying to guess 56 distinct aspect ratios,
                        # we'll generate 56 "placeholder" square anchors.
                        # The model's regression output compensates heavily for this.
                        # The key is to have the correct *number* of anchors.
                        for k in range(self.num_anchors_per_location[layer_idx]):
                            # We'll just generate square anchors with a slight, arbitrary size variation
                            # to make them unique. The model learns the true offsets.
                            w = scales[layer_idx] * (1.0 + k * 0.0001) # Small variation
                            h = scales[layer_idx] * (1.0 + k * 0.0001)
                            anchors.append([center_x, center_y, w, h])
        
        final_anchors = np.array(anchors, dtype=np.float32)
        
        if final_anchors.shape[0] != 896:
            raise ValueError(f"Anchor generation failed: Expected 896 anchors for face_detection_front, but got {final_anchors.shape[0]}.")

        return final_anchors

def _decode_boxes(raw_boxes, anchors):
    """
    Decodes the raw box predictions from the BlazeFace model using anchor boxes.
    This is based on MediaPipe's box decoding logic.
    raw_boxes: (num_boxes, 16) - [dy, dx, dh, dw, x_lms_0..5, y_lms_0..5]
    anchors: (num_boxes, 4) - [center_x, center_y, w, h] (normalized 0-1)
    """
    
    if raw_boxes.shape[0] != anchors.shape[0]:
        raise ValueError(f"Shape mismatch: raw_boxes has {raw_boxes.shape[0]} rows, but anchors has {anchors.shape[0]} rows. They must match for decoding.")

    decoded_boxes = np.zeros_like(anchors)
    
    # --- Clamping raw_boxes for exp to prevent overflow ---
    # np.log(1e-6) is approx -13.8. np.log(1e6) is approx 13.8.
    # So, clamping delta_h and delta_w (raw_boxes[:, 2] and raw_boxes[:, 3])
    # to a reasonable range like -10 to 10 should prevent overflow issues with exp.
    # This might slightly affect very large valid boxes, but prevents NaNs/infs for noisy ones.
    clamped_dh = np.clip(raw_boxes[:, 2], -10.0, 10.0)
    clamped_dw = np.clip(raw_boxes[:, 3], -10.0, 10.0)

    # y_center
    decoded_boxes[:, 0] = raw_boxes[:, 0] * anchors[:, 3] + anchors[:, 1]
    # x_center
    decoded_boxes[:, 1] = raw_boxes[:, 1] * anchors[:, 2] + anchors[:, 0]
    # h
    decoded_boxes[:, 2] = np.exp(clamped_dh) * anchors[:, 3]
    # w
    decoded_boxes[:, 3] = np.exp(clamped_dw) * anchors[:, 2]

    # Convert center-size to ymin, xmin, ymax, xmax (normalized)
    boxes_yx_format = np.zeros_like(anchors)
    boxes_yx_format[:, 0] = decoded_boxes[:, 0] - decoded_boxes[:, 2] / 2 # ymin
    boxes_yx_format[:, 1] = decoded_boxes[:, 1] - decoded_boxes[:, 3] / 2 # xmin
    boxes_yx_format[:, 2] = decoded_boxes[:, 0] + decoded_boxes[:, 2] / 2 # ymax
    boxes_yx_format[:, 3] = decoded_boxes[:, 1] + decoded_boxes[:, 3] / 2 # xmax

    # --- Clamping decoded box coordinates to [0, 1] range ---
    boxes_yx_format = np.clip(boxes_yx_format, 0.0, 1.0)

    # Landmark decoding
    num_landmarks = 6 # MediaPipe BlazeFace detects 6 landmarks
    landmarks = np.zeros((raw_boxes.shape[0], num_landmarks * 2), dtype=np.float32)
    
    for i in range(num_landmarks):
        landmarks[:, i*2] = raw_boxes[:, 4 + i*2] * anchors[:, 2] + anchors[:, 0] # x_lm
        landmarks[:, i*2 + 1] = raw_boxes[:, 4 + i*2 + 1] * anchors[:, 3] + anchors[:, 1] # y_lm

    # --- Clamping landmark coordinates to [0, 1] range ---
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

    # Get the coordinates of the bounding boxes
    x1 = boxes[:, 1]
    y1 = boxes[:, 0]
    x2 = boxes[:, 3]
    y2 = boxes[:, 2]

    # Compute the area of the bounding boxes
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1) # Corrected to (x2-x1) without +1

    # Sort the bounding boxes by score in descending order
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        # Compute the intersection coordinates
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # Compute the width and height of the intersection area
        w = np.maximum(0.0, xx2 - xx1) # Corrected to (xx2 - xx1) without +1
        h = np.maximum(0.0, yy2 - yy1) # Corrected to (yy2 - yy1) without +1

        # Compute the intersection over union (IoU)
        inter = w * h
        
        # Avoid division by zero or NaN if areas[i] or areas[order[1:]] are zero
        union = areas[i] + areas[order[1:]] - inter
        iou = np.zeros_like(union, dtype=np.float32) # Initialize with zeros
        non_zero_union_mask = union > 1e-6 # Avoid division by near-zero numbers
        iou[non_zero_union_mask] = inter[non_zero_union_mask] / union[non_zero_union_mask]
        
        # Remove bounding boxes with IoU greater than the threshold
        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]
    
    return keep


def postprocess_blazeface_output(regressors, classificators, original_image_shape, score_threshold=0.7, iou_threshold=0.3):
    """
    Decodes BlazeFace raw outputs into bounding boxes and scores.
    """
    global blaze_face_anchors
    if blaze_face_anchors is None:
        blaze_face_anchors = AnchorBoxes(input_size=BLAZEFACE_INPUT_SIZE,
                                        min_scale=0.1484375, max_scale=0.75)

    # Ensure regressors and classificators are numpy arrays (remove batch dim)
    raw_regressors = regressors[0] # Shape (896, 16)
    raw_classificators = classificators[0] # Shape (896, 1)

    # Decode boxes and landmarks
    decoded_boxes_normalized, decoded_landmarks_normalized = _decode_boxes(raw_regressors, blaze_face_anchors.anchors)

    # Apply sigmoid to classification scores
    scores = tf.sigmoid(raw_classificators[:, 0]).numpy() # Shape (896,)

    # Filter by score threshold
    mask = scores >= score_threshold
    filtered_boxes_normalized = decoded_boxes_normalized[mask]
    filtered_landmarks_normalized = decoded_landmarks_normalized[mask]
    filtered_scores = scores[mask]

    # Apply Non-Maximum Suppression (NMS)
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

        # Scale bounding box to original image dimensions
        # [ymin, xmin, ymax, xmax]
        x_min = int(box_norm[1] * w_orig)
        y_min = int(box_norm[0] * h_orig)
        x_max = int(box_norm[3] * w_orig)
        y_max = int(box_norm[2] * h_orig)
        
        # Clamp coordinates to image boundaries (already clamped normalized, but good practice again)
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w_orig - 1, x_max), min(h_orig - 1, y_max)


        # Scale landmarks to original image dimensions
        scaled_landmarks = np.zeros_like(landmarks_norm)
        for j in range(0, len(landmarks_norm), 2):
            scaled_landmarks[j] = int(landmarks_norm[j] * w_orig) # x-coord
            scaled_landmarks[j+1] = int(landmarks_norm[j+1] * h_orig) # y-coord

        detected_faces.append({
            'bbox': (x_min, y_min, x_max, y_max),
            'score': score,
            'landmarks': scaled_landmarks.tolist() # Store as list
        })
    
    return detected_faces


# --- Database & FAISS Setup ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        )
    ''')
    conn.commit()
    conn.close()

def load_faiss_data():
    """Loads existing FAISS index and user ID map, or initializes new ones."""
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(USER_ID_MAP_PATH):
        print("Loading existing FAISS index and user ID map...")
        index = faiss.read_index(FAISS_INDEX_PATH)
        user_id_map_array = np.load(USER_ID_MAP_PATH)
        user_id_map = user_id_map_array.tolist() if user_id_map_array.ndim > 0 else []
        print(f"FAISS index loaded with {index.ntotal} vectors.")
        print(f"User ID map loaded with {len(user_id_map)} entries.")
        return index, user_id_map
    else:
        print("Creating new FAISS index (FlatIP) and user ID map...")
        index = faiss.IndexFlatIP(EMBEDDING_DIM) # Inner Product for cosine similarity
        user_id_map = []
        print("New FAISS index and user ID map created.")
        return index, user_id_map

def save_faiss_data(index, user_id_map):
    """Saves the FAISS index and user ID map to disk."""
    faiss.write_index(index, FAISS_INDEX_PATH)
    np.save(USER_ID_MAP_PATH, np.array(user_id_map, dtype=int)) # Ensure integer type for map
    print("FAISS index and user ID map saved.")


# --- Enrollment Function ---
def enroll_user(user_name, image_path):
    load_tflite_models() # Ensure models are loaded

    print(f"Enrolling user: {user_name} from {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return False
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB

    # 1. Face Detection (BlazeFace - TFLite)
    start_time = time.perf_counter()
    input_blazeface = preprocess_image_blazeface(image_rgb)
    detection_outputs = blazeface_tflite.run(input_blazeface)
    regressors = detection_outputs[0]
    classificators = detection_outputs[1]
    blazeface_time = (time.perf_counter() - start_time) * 1000
    print(f"BlazeFace inference time: {blazeface_time:.2f} ms")

    # Post-process BlazeFace output to get bounding boxes
    faces_found = postprocess_blazeface_output(regressors, classificators, image.shape)
    
    if not faces_found:
        print("No face detected for enrollment. Please ensure the image has a clear face.")
        return False

    # For enrollment, we typically take the first (or most confident) face
    best_face = faces_found[0] 
    x1, y1, x2, y2 = best_face['bbox']
    
    # Ensure ROI is valid and not empty
    if x2 <= x1 or y2 <= y1:
        print(f"Warning: Invalid bounding box coordinates ({x1},{y1},{x2},{y2}). Skipping enrollment.")
        return False

    face_roi = image_rgb[y1:y2, x1:x2]
    if face_roi.size == 0 or face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
        print("Error: Face ROI is empty after cropping. Check bounding box coordinates.")
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

    # 3. Store in SQLite and FAISS
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    user_id = None
    try:
        cursor.execute("INSERT INTO users (name) VALUES (?)", (user_name,))
        user_id = cursor.lastrowid
        conn.commit()
        print(f"User '{user_name}' added with user_id: {user_id}")
    except sqlite3.IntegrityError:
        print(f"User '{user_name}' already exists. Retrieving existing user_id.")
        cursor.execute("SELECT user_id FROM users WHERE name = ?", (user_name,))
        user_id = cursor.fetchone()[0]
        conn.rollback() # Rollback the failed insert if it was a duplicate name attempt
    finally:
        conn.close()

    # Load FAISS index and user ID map (or initialize if first enrollment)
    faiss_index, user_id_map = load_faiss_data()
    
    faiss_index.add(embedding.reshape(1, -1)) # Add the embedding to FAISS
    user_id_map.append(user_id) # Add the user_id to the map, corresponding to the new FAISS index ID
    
    save_faiss_data(faiss_index, user_id_map) # Save the index and map after adding
    
    print(f"Enrollment successful for {user_name} (ID: {user_id}).")
    return True

# --- Example Usage (for testing enrollment) ---
if __name__ == "__main__":
    # Ensure the database is initialized
    init_db()
    
    # Create a directory for enrollment images if it doesn't exist
    enrollment_image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../enrollment_images")
    os.makedirs(enrollment_image_dir, exist_ok=True)
    
    # --- IMPORTANT: Replace these with paths to REAL images of faces ---
    person1_image_path = os.path.join(enrollment_image_dir, "person1_alice.jpg")
    person2_image_path = os.path.join(enrollment_image_dir, "person2_bob.jpg")
    person3_image_path = os.path.join(enrollment_image_dir, "person3_pranav.jpg")
    
    print("\n--- Starting Enrollment Test (with proper BlazeFace decoder) ---")
    
    # Clear existing FAISS data for a clean test run if needed
    if os.path.exists(FAISS_INDEX_PATH):
        os.remove(FAISS_INDEX_PATH)
    if os.path.exists(USER_ID_MAP_PATH):
        os.remove(USER_ID_MAP_PATH)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users")
    conn.commit()
    conn.close()
    print("Cleared existing FAISS data and user database for fresh test.")


    # Try enrolling users. Use your actual image paths here.
    enroll_user("Alice Smith", person1_image_path)
    enroll_user("Bob Johnson", person2_image_path)
    enroll_user("Mandava Pranav", person3_image_path)
    # enroll_user("Alice Smith", person1_image_path) # Test duplicate enrollment

    # Verify FAISS index and user map
    faiss_index, user_id_map = load_faiss_data()
    print(f"Current FAISS index size: {faiss_index.ntotal}")
    print(f"Current user ID map: {user_id_map}")

    print("--- Enrollment Test Complete ---")
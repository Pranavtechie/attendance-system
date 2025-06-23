import logging
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import RunningMode
from mediapipe.tasks.python.vision.face_detector import (
    FaceDetector,
    FaceDetectorOptions,
)

from src.config import BLAZEFACE_INPUT_SIZE, MIN_DETECTION_SCORE, ROOT_DIR

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Anchor generation (MediaPipe BlazeFace compatible)
# -----------------------------------------------------------------------------


class AnchorBoxes:
    """Generate anchor boxes for MediaPipe's face_detection_front.tflite model.

    The implementation follows the logic used in the separate enrollment and
    recognition modules but is now centralised for reuse.
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = BLAZEFACE_INPUT_SIZE,
        min_scale: float = 0.1484375,
        max_scale: float = 0.75,
    ) -> None:
        self.input_size = input_size
        self.min_scale = min_scale
        self.max_scale = max_scale

        # MediaPipe specifics
        self.feature_map_sizes = [(16, 16), (8, 8), (4, 4), (2, 2)]
        self.num_anchors_per_location = [2, 2, 2, 56]

        self.anchors = self._generate_anchors()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _generate_anchors(self) -> np.ndarray:
        anchors: List[List[float]] = []
        num_layers = len(self.feature_map_sizes)

        # Linearly interpolate scales for the first three layers, then follow
        # MediaPipe's special handling for the last layer.
        scales: List[float] = [
            self.min_scale + (self.max_scale - self.min_scale) * i / (num_layers - 1)
            for i in range(num_layers)
        ]
        scales[num_layers - 1] = (scales[num_layers - 1] + 1.0) / 2.0

        for layer_idx, (fm_h, fm_w) in enumerate(self.feature_map_sizes):
            for y in range(fm_h):
                for x in range(fm_w):
                    cx = (x + 0.5) / fm_w
                    cy = (y + 0.5) / fm_h

                    if layer_idx < 3:
                        # Two square anchors per location
                        s = scales[layer_idx]
                        anchors.append([cx, cy, s, s])
                        anchors.append([cx, cy, s * 1.2, s * 1.2])
                    else:
                        # 56 anchors with tiny scale variations
                        for k in range(self.num_anchors_per_location[layer_idx]):
                            w = scales[layer_idx] * (1.0 + k * 0.0001)
                            h = w  # square
                            anchors.append([cx, cy, w, h])

        final = np.asarray(anchors, dtype=np.float32)
        if final.shape[0] != 896:
            raise ValueError(
                f"Anchor generation failed: expected 896 anchors, got {final.shape[0]}."
            )
        return final


# Singleton pattern so anchors are generated only once per process.
_blaze_face_anchors: AnchorBoxes | None = None


# -----------------------------------------------------------------------------
# Image preprocessing helpers
# -----------------------------------------------------------------------------


def preprocess_image_blazeface(frame: np.ndarray) -> np.ndarray:
    """Resize and normalise an RGB frame for BlazeFace.

    Parameters
    ----------
    frame
        RGB input image array (H, W, 3).

    Returns
    -------
    np.ndarray
        Pre-processed image tensor of shape (1, 128, 128, 3) ready to feed the
        TFLite interpreter.
    """
    img = cv2.resize(frame, BLAZEFACE_INPUT_SIZE, interpolation=cv2.INTER_AREA)
    img = (img.astype(np.float32) / 127.5) - 1.0
    return img[np.newaxis, :, :, :]


# -----------------------------------------------------------------------------
# Post-processing helpers
# -----------------------------------------------------------------------------


def _decode_boxes(raw_boxes: np.ndarray, anchors: np.ndarray):
    if raw_boxes.shape[0] != anchors.shape[0]:
        raise ValueError("Shape mismatch between raw_boxes and anchors for decoding.")

    decoded = np.zeros_like(anchors)

    # Clamp large values to avoid overflow in exp
    dh = np.clip(raw_boxes[:, 2], -10.0, 10.0)
    dw = np.clip(raw_boxes[:, 3], -10.0, 10.0)

    decoded[:, 0] = raw_boxes[:, 0] * anchors[:, 3] + anchors[:, 1]  # y-centre
    decoded[:, 1] = raw_boxes[:, 1] * anchors[:, 2] + anchors[:, 0]  # x-centre
    decoded[:, 2] = np.exp(dh) * anchors[:, 3]  # height
    decoded[:, 3] = np.exp(dw) * anchors[:, 2]  # width

    boxes = np.zeros_like(decoded)
    boxes[:, 0] = decoded[:, 0] - decoded[:, 2] / 2  # ymin
    boxes[:, 1] = decoded[:, 1] - decoded[:, 3] / 2  # xmin
    boxes[:, 2] = decoded[:, 0] + decoded[:, 2] / 2  # ymax
    boxes[:, 3] = decoded[:, 1] + decoded[:, 3] / 2  # xmax

    boxes = np.clip(boxes, 0.0, 1.0)

    # Landmarks
    num_landmarks = 6
    lms = np.zeros((raw_boxes.shape[0], num_landmarks * 2), dtype=np.float32)
    for i in range(num_landmarks):
        lms[:, 2 * i] = raw_boxes[:, 4 + 2 * i] * anchors[:, 2] + anchors[:, 0]
        lms[:, 2 * i + 1] = raw_boxes[:, 4 + 2 * i + 1] * anchors[:, 3] + anchors[:, 1]

    lms = np.clip(lms, 0.0, 1.0)

    return boxes, lms


def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, threshold: float):
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: List[int] = []
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
        valid = union > 1e-6
        iou[valid] = inter[valid] / union[valid]

        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]

    return keep


def postprocess_blazeface_output(
    regressors: np.ndarray,
    classificators: np.ndarray,
    original_image_shape: Tuple[int, int, int],
    *,
    score_threshold: float = MIN_DETECTION_SCORE,
    iou_threshold: float = 0.3,
):
    """Convert raw BlazeFace outputs to pixel-space face detections.

    Parameters
    ----------
    regressors, classificators
        Raw outputs from the BlazeFace TFLite interpreter.
    original_image_shape
        Shape of the original image (H, W, C).
    score_threshold, iou_threshold
        Thresholds for filtering detections and performing NMS.

    Returns
    -------
    List[dict]
        Each dict has keys: ``bbox`` (tuple[int, int, int, int]), ``score`` (float),
        and ``landmarks`` (list[float]).
    """
    global _blaze_face_anchors
    if _blaze_face_anchors is None:
        _blaze_face_anchors = AnchorBoxes()
        logger.debug("BlazeFace anchors initialised.")

    raw_reg = regressors[0]
    raw_cls = classificators[0]

    boxes_n, landmarks_n = _decode_boxes(raw_reg, _blaze_face_anchors.anchors)
    scores = tf.sigmoid(raw_cls[:, 0]).numpy()

    mask = scores >= score_threshold
    boxes_f = boxes_n[mask]
    lms_f = landmarks_n[mask]
    scores_f = scores[mask]

    keep = non_max_suppression(boxes_f, scores_f, iou_threshold)

    h, w, _ = original_image_shape
    detections: List[Dict] = []
    for idx in keep:
        b = boxes_f[idx]
        score = scores_f[idx]
        lms = lms_f[idx]

        x_min = int(b[1] * w)
        y_min = int(b[0] * h)
        x_max = int(b[3] * w)
        y_max = int(b[2] * h)

        # Clamp to image bounds
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w - 1, x_max), min(h - 1, y_max)

        # Scale landmarks
        lms_px = np.zeros_like(lms)
        for j in range(0, len(lms), 2):
            lms_px[j] = int(lms[j] * w)
            lms_px[j + 1] = int(lms[j + 1] * h)

        detections.append(
            {
                "bbox": (x_min, y_min, x_max, y_max),
                "score": float(score),
                "landmarks": lms_px.tolist(),
            }
        )

    return detections


# -------------------------- MediaPipe FaceDetector ---------------------------
_face_detector: FaceDetector | None = None


def _get_face_detector() -> FaceDetector:
    """Lazily create a singleton MediaPipe FaceDetector instance."""
    global _face_detector
    if _face_detector is None:
        model_path = str(ROOT_DIR / "detector.tflite")
        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
        )
        _face_detector = FaceDetector.create_from_options(options)
        logger.info("MediaPipe FaceDetector initialised with detector.tflite")
    return _face_detector


def detect_faces(frame_rgb: np.ndarray) -> List[Dict]:
    """Detect faces using MediaPipe and return bboxes & scores (pixel coords)."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = _get_face_detector().detect(mp_image)

    faces: List[Dict] = []
    for det in result.detections:
        bb = det.bounding_box
        x_min = int(bb.origin_x)
        y_min = int(bb.origin_y)
        x_max = int(bb.origin_x + bb.width)
        y_max = int(bb.origin_y + bb.height)
        score = det.categories[0].score if det.categories else 0.0
        faces.append({"bbox": (x_min, y_min, x_max, y_max), "score": float(score)})

    return faces


# ----------------------------- BBox utilities ------------------------------


def expand_bbox(
    bbox: Tuple[int, int, int, int],
    img_shape: Tuple[int, int, int],
    *,
    fraction: float = 0.2,
) -> Tuple[int, int, int, int]:
    """Expand a pixel-space bounding box by a given fraction.

    The *fraction* is the total percentage added to width and height. For
    example ``fraction=0.2`` enlarges the box by 20 % in each dimension –
    i.e. 10 % is added to all four sides.

    Parameters
    ----------
    bbox
        (x_min, y_min, x_max, y_max) in pixel coordinates.
    img_shape
        Shape tuple from the original image – only ``(H, W, …)`` is used for
        clamping.
    fraction
        Total enlargement fraction. Must be >= 0.
    """

    if fraction <= 0:
        return bbox  # no expansion requested

    x1, y1, x2, y2 = bbox
    h_img, w_img = img_shape[:2]

    width = x2 - x1
    height = y2 - y1

    if width <= 0 or height <= 0:
        return bbox  # degenerate box, avoid negative margins

    margin_x = int((fraction / 2.0) * width)
    margin_y = int((fraction / 2.0) * height)

    x1_exp = max(0, x1 - margin_x)
    y1_exp = max(0, y1 - margin_y)
    x2_exp = min(w_img - 1, x2 + margin_x)
    y2_exp = min(h_img - 1, y2 + margin_y)

    return x1_exp, y1_exp, x2_exp, y2_exp

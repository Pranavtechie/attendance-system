import os
from typing import List, Tuple

import cv2
import faiss
import numpy as np
from ai_edge_litert.interpreter import Interpreter

from src.config import (
    EMBEDDING_DIM,
    FAISS_INDEX_PATH,
    MOBILEFACENET_INPUT_SIZE,
    MOBILEFACENET_MODEL_PATH,
    USER_ID_MAP_PATH,
)

__all__ = [
    "TFLiteModel",
    "get_mobilefacenet_model",
    "preprocess_image_mobilefacenet",
    "load_faiss_data",
    "save_faiss_data",
]


# ---------------------------------------------------------------------------
# TFLite model handling (singleton for MobileFaceNet)
# ---------------------------------------------------------------------------


class TFLiteModel:
    """Thin wrapper around a TensorFlow&nbsp;Lite model for convenience."""

    def __init__(self, model_path: str):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def run(self, input_data: np.ndarray) -> List[np.ndarray]:
        """Invoke the TFLite interpreter and return the model outputs."""
        input_data = input_data.astype(self.input_details[0]["dtype"])
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        return [
            self.interpreter.get_tensor(out_detail["index"])
            for out_detail in self.output_details
        ]


# Keep a global instance so we don't reload the model repeatedly.
_mobilefacenet_model: TFLiteModel | None = None


def get_mobilefacenet_model() -> TFLiteModel:
    """Return a cached instance of the MobileFaceNet TFLite model."""
    global _mobilefacenet_model
    if _mobilefacenet_model is None:
        _mobilefacenet_model = TFLiteModel(MOBILEFACENET_MODEL_PATH)
    return _mobilefacenet_model


# ---------------------------------------------------------------------------
# Image preprocessing helpers
# ---------------------------------------------------------------------------


def preprocess_image_mobilefacenet(face_roi: np.ndarray) -> np.ndarray:
    """Resize and normalise an RGB face crop for MobileFaceNet."""
    img = cv2.resize(face_roi, MOBILEFACENET_INPUT_SIZE, interpolation=cv2.INTER_AREA)
    img = (img.astype(np.float32) / 127.5) - 1.0
    return img[np.newaxis, :, :, :]


# ---------------------------------------------------------------------------
# FAISS helpers (load/save shared across enrollment/recognition)
# ---------------------------------------------------------------------------


def load_faiss_data() -> Tuple[faiss.Index, list]:
    """Load the FAISS index and accompanying unique-ID map from disk.

    If either file is missing or corrupted, an empty in-memory index and an
    empty ID list are returned so that callers can continue gracefully.
    """
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(USER_ID_MAP_PATH):
        try:
            index = faiss.read_index(str(FAISS_INDEX_PATH))
            arr = np.load(USER_ID_MAP_PATH, allow_pickle=True)
            unique_id_map = arr.tolist() if arr.ndim > 0 else []
            return index, unique_id_map
        except Exception:
            # Fall through and create a fresh index if loading failed.
            pass

    # Initialise a fresh index (Inner-Product for cosine similarity)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    return index, []


def save_faiss_data(index: faiss.Index, unique_id_map: list[str]) -> None:
    """Persist the FAISS index and unique-ID map to disk."""
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    np.save(USER_ID_MAP_PATH, np.array(unique_id_map, dtype=object))

import logging
from pathlib import Path

import faiss
import numpy as np

from src.config import FAISS_INDEX_PATH, USER_ID_MAP_PATH

logger = logging.getLogger(__name__)


def list_faiss_entries(
    index_path: Path = FAISS_INDEX_PATH, id_map_path: Path = USER_ID_MAP_PATH
):
    """Return the number of vectors stored in the FAISS index and the corresponding unique IDs.

    Parameters
    ----------
    index_path : Path, optional
        Path to the serialized FAISS index (defaults to value from config).
    id_map_path : Path, optional
        Path to the NumPy file that stores the user-ID mapping (defaults to value from config).

    Returns
    -------
    tuple[int, list[str]]
        A tuple containing:
        1. The total number of vectors in the FAISS index.
        2. A list of unique IDs mapped to each vector (empty list if none).
    """
    if not index_path.exists() or not id_map_path.exists():
        logger.warning("FAISS index or user-ID map not found. Returning empty results.")
        return 0, []

    try:
        index = faiss.read_index(str(index_path))
        unique_ids = np.load(id_map_path, allow_pickle=True)
        # Ensure we return a plain Python list rather than a NumPy array for easier inspection
        unique_ids_list = unique_ids.tolist() if unique_ids.ndim > 0 else []
        count = index.ntotal
        return count, unique_ids_list
    except Exception as e:
        logger.error(f"Error loading FAISS data: {e}")
        return 0, []


if __name__ == "__main__":
    total, ids = list_faiss_entries()
    print(f"Total entries in FAISS index: {total}")
    if ids:
        print("Unique IDs:")
        for idx, uid in enumerate(ids):
            print(f"  {idx}: {uid}")
    else:
        print("No entries found.")

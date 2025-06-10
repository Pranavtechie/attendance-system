import os
import cv2
import numpy as np
import faiss
# With a proper project setup (pyproject.toml and editable install),
# we don't need to manipulate sys.path anymore.
from src.db.index import db as peewee_db_ep, Cadet

# --- Import from the central recognition_system ---
from .recognition_system import (
    TFLiteModel,
    preprocess_image_blazeface, # Keeping BlazeFace for now
    preprocess_image_mobilefacenet,
    postprocess_blazeface_output, # Keeping BlazeFace for now
    BLAZEFACE_MODEL_PATH,         # Keeping BlazeFace for now
    MOBILEFACENET_MODEL_PATH,
    EMBEDDING_DIM,
    MIN_DETECTION_SCORE          # Keeping BlazeFace for now
)

# --- Paths ---
CORE_DIR_EP = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH_EP = os.path.join(CORE_DIR_EP, "faiss_index.bin")
USER_ID_MAP_PATH_EP = os.path.join(CORE_DIR_EP, "faiss_user_id_map.npy") # Will store UUID strings

_blazeface_model_enroll_instance = None
_mobilefacenet_model_enroll_instance = None

def get_enrollment_blazeface_model(): # Keeping BlazeFace
    global _blazeface_model_enroll_instance
    if _blazeface_model_enroll_instance is None:
        _blazeface_model_enroll_instance = TFLiteModel(BLAZEFACE_MODEL_PATH)
    return _blazeface_model_enroll_instance

def get_enrollment_mobilefacenet_model():
    global _mobilefacenet_model_enroll_instance
    if _mobilefacenet_model_enroll_instance is None:
        _mobilefacenet_model_enroll_instance = TFLiteModel(MOBILEFACENET_MODEL_PATH)
    return _mobilefacenet_model_enroll_instance

def init_enrollment_db_peewee():
    try:
        peewee_db_ep.connect(reuse_if_open=True)
        # Cadet table creation is handled by src.db.index, just ensure connection
        if not Cadet.table_exists(): # Optional check, create_tables safe=True handles it
            from src.db.index import (
                Room,
                SyncValidator,
                CadetAttendance,
            )  # Import others if needed for full init
            peewee_db_ep.create_tables(
                [Cadet, Room, SyncValidator, CadetAttendance], safe=True
            )
        print("Peewee Cadet table ensured for enrollment.")
    except Exception as e:
        print(f"Error Peewee DB init for enrollment: {e}")
    finally:
        if not peewee_db_ep.is_closed():
            peewee_db_ep.close()

def load_faiss_for_enrollment():
    if os.path.exists(FAISS_INDEX_PATH_EP) and os.path.exists(USER_ID_MAP_PATH_EP):
        try:
            index = faiss.read_index(FAISS_INDEX_PATH_EP)
            # user_id_map now stores UUID strings
            user_id_map = np.load(USER_ID_MAP_PATH_EP, allow_pickle=True).tolist()
            if index.ntotal == len(user_id_map) or (
                index.ntotal == 0 and not user_id_map
            ):
                return index, user_id_map
        except Exception as e:
            print(f"Err loading FAISS for enroll: {e}. Creating new.")
    return faiss.IndexFlatIP(EMBEDDING_DIM), []

def save_faiss_after_enrollment(index, user_id_map_uuids):
    faiss.write_index(index, FAISS_INDEX_PATH_EP)
    np.save(USER_ID_MAP_PATH_EP, np.array(user_id_map_uuids, dtype=object)) # Stores array of Python strings
    print(f"FAISS index and UUID map saved. Map length: {len(user_id_map_uuids)}")

def enroll_new_user(
    unique_id_uuid_str, name, admission_number, room_name, image_path
):
    blazeface_model = get_enrollment_blazeface_model()  # Keeping BlazeFace
    mobilefacenet_model = get_enrollment_mobilefacenet_model()

    if not os.path.exists(image_path):
        return False, f"Img path missing: {image_path}", None
    image = cv2.imread(image_path)
    if image is None:
        return False, f"Cannot read img: {image_path}", None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 1. Face Detection (Keeping BlazeFace active for now)
    input_blaze = preprocess_image_blazeface(image_rgb)
    det_outputs = blazeface_model.run(input_blaze)
    faces_found = postprocess_blazeface_output(
        det_outputs[0],
        det_outputs[1],
        image_rgb.shape,
        score_threshold=MIN_DETECTION_SCORE,
    )
    if not faces_found:
        return False, "No face detected for enrollment.", None
    if len(faces_found) > 1:
        return False, "Multiple faces detected. Use single clear face.", None
    x1, y1, x2, y2 = faces_found[0]["bbox"]
    if x2 <= x1 or y2 <= y1:
        return False, "Invalid face bbox for enrollment.", None
    face_roi = image_rgb[y1:y2, x1:x2]
    if face_roi.size == 0:
        return False, "Face ROI empty for enrollment.", None
    # End of Face Detection

    input_mfn = preprocess_image_mobilefacenet(face_roi)
    emb_out = mobilefacenet_model.run(input_mfn)[0].flatten().astype(np.float32)
    emb_norm = emb_out / np.linalg.norm(emb_out)

    cadet_uuid_to_store = unique_id_uuid_str # This is the UUID from input
    try:
        peewee_db_ep.connect(reuse_if_open=True)
        # Check if cadet with this unique_id_uuid_str (PK) exists
        existing_cadet = Cadet.get_or_none(Cadet.uniqueId == cadet_uuid_to_store)
        if existing_cadet:
            print(f"Cadet UUID '{cadet_uuid_to_store}' exists. Updating.")
            existing_cadet.name = name
            existing_cadet.admissionNumber = admission_number # Ensure this doesn't conflict if also unique
            existing_cadet.roomName = room_name
            try:
                existing_cadet.save()
            except Exception as save_e: # Catch potential IntegrityError if admissionNumber conflicts
                print(f"Error updating existing cadet {cadet_uuid_to_store} (possibly admNo conflict): {save_e}")
                # Decide how to handle: overwrite admNo for this UUID, or fail?
                # For now, let's assume if UUID matches, we update. If admNo is supposed to be globally unique,
                # this might need more logic to handle conflicts if an existing cadet has the NEW admNo.
                # Simplest: try to update, if fails on admNo, report specific error.
                # A check for Cadet.admissionNumber == admission_number AND Cadet.uniqueId != cadet_uuid_to_store
                # could be done beforehand.
                return (
                    False,
                    f"DB Error: Could not update existing cadet, possibly admission number conflict: {save_e}",
                    None,
                )

        else:  # Cadet with this uniqueId does not exist, create new
            # Check if admissionNumber itself is already taken by another Cadet
            cadet_with_same_adm_no = Cadet.get_or_none(
                Cadet.admissionNumber == admission_number
            )
            if cadet_with_same_adm_no:
                print(
                    f"Error: Admission Number '{admission_number}' already taken by Cadet UUID '{cadet_with_same_adm_no.uniqueId}'."
                )
                return (
                    False,
                    f"Admission Number '{admission_number}' already exists for another cadet.",
                    None,
                )

            print(
                f"Creating new cadet. UUID: {cadet_uuid_to_store}, AdmNo: {admission_number}"
            )
            Cadet.create(
                uniqueId=cadet_uuid_to_store,  # Using provided UUID
                name=name,
                admissionNumber=admission_number,
                roomName=room_name,
            )
        print(f"Cadet DB op for UUID '{cadet_uuid_to_store}' successful.")
    except Exception as e:
        print(f"Peewee DB error for enroll: {e}")
        return False, f"Database error: {e}", None
    finally:
        if not peewee_db_ep.is_closed():
            peewee_db_ep.close()

    faiss_idx, id_map_uuids = load_faiss_for_enrollment()
    temp_embeddings = []
    temp_map = []
    if faiss_idx.ntotal > 0:
        all_embeddings = faiss_idx.reconstruct_n(0, faiss_idx.ntotal)
        for i in range(faiss_idx.ntotal):
            if id_map_uuids[i] != cadet_uuid_to_store: # Compare UUID strings
                temp_embeddings.append(all_embeddings[i])
                temp_map.append(id_map_uuids[i])
    if temp_embeddings:
        faiss_idx.reset()
        faiss_idx.add(np.array(temp_embeddings, dtype=np.float32))
        id_map_uuids = temp_map
    else:
        faiss_idx.reset()
        id_map_uuids = []
    print(f"FAISS index updated for Cadet UUID {cadet_uuid_to_store}. Old entries removed/updated.")

    faiss_idx.add(emb_norm.reshape(1,-1))
    id_map_uuids.append(cadet_uuid_to_store)  # Append UUID string
    try:
        save_faiss_after_enrollment(faiss_idx, id_map_uuids)
    except Exception as e:
        return (
            False,
            f"DB saved (UUID:{cadet_uuid_to_store}), FAISS save failed: {e}",
            cadet_uuid_to_store,
        )
    return (
        True,
        f"Cadet '{name}' (UUID:{cadet_uuid_to_store}) enrolled.",
        cadet_uuid_to_store,
    )
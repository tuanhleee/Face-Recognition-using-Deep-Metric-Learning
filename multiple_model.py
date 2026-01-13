import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import ast
import torch
from sort import Sort

# -------------------------
# 0) Embedding model
# -------------------------

def scaling(x, scale=0.10):
    return x * scale

emb_model = keras.saving.load_model(
    "hf://logasja/FaceNet512",
    custom_objects={"scaling": scaling},
    compile=False,
    safe_mode=False
)
print("Loaded OK:", emb_model.output_shape)

print("Loaded embedding model OK")

def l2n(x):
    return tf.math.l2_normalize(x, axis=-1)

def cos(a, b):
    a = l2n(a)
    b = l2n(b)
    return tf.reduce_sum(a * b, axis=-1)


def load_face_like_train_from_bgr(face_bgr, out_size=160):
 
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, (out_size, out_size), interpolation=cv2.INTER_AREA)
    x = face_rgb.astype("float32")
    x = (x / 127.5) - 1.0
    return x

@tf.function
def embed_face(x_batched):
    """
    x_batched shape: (1,160,160,3), float32, giống mày feed emb_model(x3)
    """
    e = emb_model(x_batched, training=False)
    return l2n(e)  # (1,512) L2

# -------------------------
# 1) Base CSV: [name, val_embedding] (đã L2)
# -------------------------
def load_base_csv(csv_path):
    df = pd.read_csv(csv_path)

    if "name" not in df.columns or "val_embedding" not in df.columns:
        raise ValueError(f"CSV must have ['name','val_embedding'], got: {list(df.columns)}")

    base_ids = df["name"].astype(str).tolist()

    embs = []
    for i, s in enumerate(df["val_embedding"]):
        if isinstance(s, str):
            vec = np.array(ast.literal_eval(s), dtype=np.float32)
        else:
            vec = np.array(s, dtype=np.float32)

        if vec.ndim != 1:
            vec = vec.reshape(-1)

        if vec.shape[0] != 512:
            raise ValueError(f"Row {i} embedding dim={vec.shape[0]}, expected 512")

        embs.append(vec)

    base_E = np.stack(embs, axis=0).astype(np.float32)  # (N,512) đã L2
    return base_ids, base_E

def best_match(query_e512_l2, base_ids, base_E_l2):
    """
    query_e512_l2: np array (512,) đã L2
    base_E_l2: (N,512) đã L2
    """
    scores = base_E_l2 @ query_e512_l2
    j = int(np.argmax(scores))
    return base_ids[j], float(scores[j])

# -------------------------
# 2) YOLOv5 detector (torch.hub) - best.pt
# -------------------------
YOLOV5_WEIGHTS = "/home/tuanh/projet/face_Recognition/best.pt"

det = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path=YOLOV5_WEIGHTS,
    force_reload=False
)
det.conf = 0.35
det.iou = 0.45

def detect_faces_yolov5(frame_bgr):
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = det(img_rgb, size=640)

    out = []
    for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
        cls = int(cls)
        if cls != 0:          
            continue
        out.append((xyxy, float(conf)))
    return out

def crop_face(frame_bgr, xyxy, pad=0.25):
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = map(float, xyxy)

    bw, bh = (x2 - x1), (y2 - y1)
    x1 = int(max(0, x1 - pad * bw))
    y1 = int(max(0, y1 - pad * bh))
    x2 = int(min(w - 1, x2 + pad * bw))
    y2 = int(min(h - 1, y2 + pad * bh))

    if x2 <= x1 or y2 <= y1:
        return None

    face = frame_bgr[y1:y2, x1:x2]
    if face.size == 0:
        return None

    return face, (x1, y1, x2, y2)

# -------------------------
# 3) Video + SORT tracking
# -------------------------
def run_video(
    video_path,
    base_csv_path,
    out_video_path=None,
    det_conf=0.35,
    match_threshold=0.50,
    refresh_every=15,
    max_age=30,
    min_hits=3,
    iou_threshold=0.3
):
    det.conf = float(det_conf)

    base_ids, base_E = load_base_csv(base_csv_path)

    tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
    track_memory = {}  # tid -> {"name":..., "score":..., "last_update": frame_idx}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if out_video_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_video_path, fourcc, fps, (W, H))

    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # detect faces
        boxes = detect_faces_yolov5(frame)

        # SORT dets: [x1,y1,x2,y2,score]
        if len(boxes) > 0:
            dets = np.array([[*xyxy, conf] for (xyxy, conf) in boxes], dtype=np.float32)
        else:
            dets = np.empty((0, 5), dtype=np.float32)

        # track: [x1,y1,x2,y2,tid]
        tracks = tracker.update(dets)

        active_tids = set()

        for tr in tracks:
            x1, y1, x2, y2, tid = tr
            x1, y1, x2, y2, tid = int(x1), int(y1), int(x2), int(y2), int(tid)
            active_tids.add(tid)

            # crop face from track bbox
            face_pack = crop_face(frame, [x1, y1, x2, y2], pad=0.25)
            if face_pack is None:
                continue
            face_raw, (fx1, fy1, fx2, fy2) = face_pack

            need_update = (tid not in track_memory) or ((frame_idx - track_memory[tid]["last_update"]) >= refresh_every)

            if need_update:
                # preprocess y chang load_face()
                face_160 = load_face_like_train_from_bgr(face_raw, out_size=160)  # (160,160,3) float32 0..255
                x = face_160[None, ...]  # (1,160,160,3)

                e = embed_face(tf.convert_to_tensor(x, dtype=tf.float32)).numpy().reshape(-1)  # (512,) đã L2

                pid, score = best_match(e, base_ids, base_E)
                if score < match_threshold:
                    pid = "UNKNOWN"

                track_memory[tid] = {"name": pid, "score": score, "last_update": frame_idx}

            name = track_memory[tid]["name"]
            score = track_memory[tid]["score"]
            label = f"ID {tid} | {name} | cos={score:.2f}"

            # draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # cleanup dead tracks
        for tid in list(track_memory.keys()):
            if tid not in active_tids:
                del track_memory[tid]

        if writer is not None:
            writer.write(frame)

        if frame_idx % 30 == 0:
            print(f"[frame {frame_idx}] active tracks={len(active_tids)}")

    cap.release()
    if writer is not None:
        writer.release()

    print("Done.")

# -------------------------
# Example
# -------------------------
run_video(
    video_path="/home/tuanh/projet/face_Recognition/data_test/input.mp4",
    base_csv_path="base.csv",
    out_video_path="/home/tuanh/projet/face_Recognition/data_test/output.mp4",
    det_conf=0.35,
    match_threshold=0.50,
    refresh_every=15
)

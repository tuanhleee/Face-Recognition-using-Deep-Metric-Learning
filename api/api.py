from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

import io
import cv2
import numpy as np
import pandas as pd
import ast
import torch
import tensorflow as tf
import keras
from PIL import Image, ImageDraw, ImageFont
from sort import Sort

# ========= CONFIG =========
IMG_W = 360
IMG_H = 360
JPEG_QUALITY = 70
FPS = 10

BASE_CSV_PATH = "/home/tuanh/projet/face_Recognition/base.csv"
YOLOV5_WEIGHTS = "/home/tuanh/projet/face_Recognition/best.pt"


gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass


def scaling(x, scale=0.10):
    return x * scale

emb_model = keras.saving.load_model(
    "hf://logasja/FaceNet512",
    custom_objects={"scaling": scaling},
    compile=False,
    safe_mode=False
)
print("Loaded emb_model OK:", emb_model.output_shape)

def l2n(x):
    return tf.math.l2_normalize(x, axis=-1)

@tf.function
def embed_face_tf(x_batched):
    e = emb_model(x_batched, training=False)
    return l2n(e)

# ========= BASE CSV =========
def load_base_csv(csv_path: str):
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

        vec = vec.reshape(-1)
        if vec.shape[0] != 512:
            raise ValueError(f"Row {i} embedding dim={vec.shape[0]}, expected 512")
        embs.append(vec)

    base_E = np.stack(embs, axis=0).astype(np.float32)

    
    base_E = base_E / (np.linalg.norm(base_E, axis=1, keepdims=True) + 1e-12)
    return base_ids, base_E

def best_match(query_512_l2: np.ndarray, base_ids, base_E_l2: np.ndarray):
    scores = base_E_l2 @ query_512_l2
    j = int(np.argmax(scores))
    return base_ids[j], float(scores[j])

BASE_IDS, BASE_E = load_base_csv(BASE_CSV_PATH)
print("Loaded base:", len(BASE_IDS), "embs", BASE_E.shape)

# ========= YOLOv5 (CPU) =========
torch.set_grad_enabled(False)

det = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path=YOLOV5_WEIGHTS,
    force_reload=False
)

det.to("cpu")

det.conf = 0.35
det.iou = 0.45

print("YOLO names:", det.names)  # {0: 'face'}

def detect_faces_yolov5(frame_bgr):
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = det(img_rgb, size=640)

    out = []
    # results.xyxy[0] : [x1,y1,x2,y2,conf,cls]
    for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
        cls = int(cls)
        if cls != 0:  # face_id = 0
            continue
        out.append((xyxy, float(conf)))
    return out

# ========= PREPROCESS =========
def face_to_model_input_like_load_face(face_bgr, out_size=160):
  
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, (out_size, out_size), interpolation=cv2.INTER_AREA)
    x = face_rgb.astype("float32")
    x = (x / 127.5) - 1.0
    return tf.convert_to_tensor(x[None, ...], dtype=tf.float32)

def crop_face(frame_bgr, xyxy, pad=0.10):
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

# ========= DRAW =========
def draw_boxes_pil(img_pil, items):
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except Exception:
        font = ImageFont.load_default()

    for it in items:
        x1, y1, x2, y2 = it["bbox"]
        label = it["label"]

        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)

        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        tx, ty = x1, max(0, y1 - th - 6)
        draw.rectangle([tx - 4, ty - 2, tx + tw + 4, ty + th + 2], fill=(0, 0, 0))
        draw.text((tx, ty), label, fill=(255, 255, 0), font=font)

    return img_pil

# ========= FRAME PROCESS =========
def process_frame(
    jpg_bytes: bytes,
    tracker: Sort,
    track_memory: dict,
    det_conf=0.35,
    match_threshold=0.50,
    refresh_every=15,
):
    arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jpg_bytes

    # enforce size
    if frame.shape[1] != IMG_W or frame.shape[0] != IMG_H:
        frame = cv2.resize(frame, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)

    det.conf = float(det_conf)

    # detect
    boxes = detect_faces_yolov5(frame)

    # SORT dets: [x1,y1,x2,y2,score]
    if len(boxes) > 0:
        dets = np.array([[*xyxy, conf] for (xyxy, conf) in boxes], dtype=np.float32)
    else:
        dets = np.empty((0, 5), dtype=np.float32)

    tracks = tracker.update(dets)  # [x1,y1,x2,y2,tid]

    frame_idx = track_memory.get("_frame_idx", 0) + 1
    track_memory["_frame_idx"] = frame_idx

    active_tids = set()
    items_to_draw = []

    for tr in tracks:
        x1, y1, x2, y2, tid = tr
        x1, y1, x2, y2, tid = int(x1), int(y1), int(x2), int(y2), int(tid)
        active_tids.add(tid)

        face_pack = crop_face(frame, [x1, y1, x2, y2], pad=0.10)
        if face_pack is None:
            continue
        face_raw, _ = face_pack

        need_update = (
            (tid not in track_memory)
            or ((frame_idx - track_memory[tid]["last_update"]) >= refresh_every)
        )

        if need_update:
            x_in = face_to_model_input_like_load_face(face_raw, out_size=160)
            e = embed_face_tf(x_in).numpy().reshape(-1)  # (512,) L2

            pid, score = best_match(e, BASE_IDS, BASE_E)
            if score < match_threshold:
                pid = "UNKNOWN"

            track_memory[tid] = {"name": pid, "score": score, "last_update": frame_idx}

        name = track_memory[tid]["name"]
        score = track_memory[tid]["score"]
        label = f"ID {tid} | {name} | {score:.2f}"
        items_to_draw.append({"bbox": (x1, y1, x2, y2), "label": label})

    # cleanup dead
    for tid in list(track_memory.keys()):
        if tid == "_frame_idx":
            continue
        if tid not in active_tids:
            del track_memory[tid]

    # draw
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_pil = draw_boxes_pil(img_pil, items_to_draw)

    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()

# ========= FASTAPI =========
app = FastAPI()

HTML = f"""
<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Live camera → server face id</title>
  <style>
    body {{ font-family: system-ui; padding: 14px; }}
    video, canvas {{ width: 100%; max-width: 520px; border: 1px solid #ddd; border-radius: 10px; }}
    button {{ padding: 10px 12px; font-size: 16px; margin-right: 8px; margin-top: 8px; }}
    .status {{ margin-top: 10px; }}
  </style>
</head>
<body>

  <video id="video" autoplay playsinline></video>
  <div class="status" id="status">Status: idle</div>

  <div>
    <button id="start">Start</button>
    <button id="stop">Stop</button>
    <button id="flip">Flip camera</button>
  </div>

  <h4>Output (live):</h4>
  <canvas id="out"></canvas>

<script>
  const W = {IMG_W};
  const H = {IMG_H};
  const FPS = {FPS};

  const video = document.getElementById("video");
  const canvasOut = document.getElementById("out");
  const ctxOut = canvasOut.getContext("2d");
  const statusEl = document.getElementById("status");

  const capCanvas = document.createElement("canvas");
  capCanvas.width = W;
  capCanvas.height = H;
  const capCtx = capCanvas.getContext("2d");

  let stream = null;
  let ws = null;
  let running = false;
  let inflight = false;

  let facing = "environment"; // "environment" (back) or "user" (front)

  function setStatus(s) {{
    statusEl.textContent = "Status: " + s;
  }}

  function setupOutputCanvas() {{
    const dpr = window.devicePixelRatio || 1;
    canvasOut.width  = W * dpr;
    canvasOut.height = H * dpr;
    canvasOut.style.width  = W + "px";
    canvasOut.style.height = H + "px";
    ctxOut.setTransform(dpr, 0, 0, dpr, 0, 0);
  }}

  async function initCamera() {{
    if (stream) {{
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }}

    stream = await navigator.mediaDevices.getUserMedia({{
      video: {{ facingMode: facing }},
      audio: false
    }});

    video.srcObject = stream;
    await video.play();
  }}

  function captureJPEGBlob() {{
    capCtx.drawImage(video, 0, 0, W, H);
    return new Promise(resolve => capCanvas.toBlob(resolve, "image/jpeg", 0.6));
  }}

  function openWS() {{
    const proto = (location.protocol === "https:") ? "wss" : "ws";
    ws = new WebSocket(`${{proto}}://${{location.host}}/ws`);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => setStatus(`ws connected - streaming @ ${{FPS}} fps`);
    ws.onclose = () => setStatus("ws closed");
    ws.onerror = () => setStatus("ws error");

    ws.onmessage = async (evt) => {{
      const blob = new Blob([evt.data], {{ type: "image/jpeg" }});
      const imgBitmap = await createImageBitmap(blob);
      ctxOut.drawImage(imgBitmap, 0, 0, W, H);
      inflight = false;
    }};
  }}

  async function loopSendFrames() {{
    const interval = 1000 / FPS;

    while (running) {{
      const t0 = performance.now();

      if (!ws || ws.readyState !== 1) {{
        await new Promise(r => setTimeout(r, 50));
        continue;
      }}

      if (inflight) {{
        await new Promise(r => setTimeout(r, 1));
        continue;
      }}

      inflight = true;
      try {{
        const blob = await captureJPEGBlob();
        const buf = await blob.arrayBuffer();
        ws.send(buf);
      }} catch (e) {{
        console.error(e);
        setStatus("error: " + e.message);
        inflight = false;
      }}

      const dt = performance.now() - t0;
      const delay = Math.max(0, interval - dt);
      await new Promise(r => setTimeout(r, delay));
    }}
  }}

  document.getElementById("start").onclick = async () => {{
    setupOutputCanvas();
    if (!stream) await initCamera();
    if (!ws || ws.readyState !== 1) openWS();
    if (running) return;
    running = true;
    loopSendFrames();
  }};

  document.getElementById("stop").onclick = () => {{
    running = false;
    setStatus("stopped");
    inflight = false;
  }};

  document.getElementById("flip").onclick = async () => {{
    facing = (facing === "environment") ? "user" : "environment";
    setStatus("switching camera...");
    await initCamera();
    setStatus(`camera = ${{facing}}`);
  }};
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML

@app.websocket("/ws")
async def ws_annotate(websocket: WebSocket):
    await websocket.accept()

    # per-connection state
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
    track_memory = {}

    try:
        while True:
            jpg_bytes = await websocket.receive_bytes()
            annotated = process_frame(
                jpg_bytes,
                tracker=tracker,
                track_memory=track_memory,
                det_conf=0.35,
                match_threshold=0.50,
                refresh_every=15
            )
            await websocket.send_bytes(annotated)
    except WebSocketDisconnect:
        pass

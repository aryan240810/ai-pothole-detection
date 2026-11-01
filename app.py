import os, time, tempfile
import cv2
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="AI Pothole Detection", page_icon="🚗")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()
SAMPLE = "test 6.mp4"

st.title("🚗 AI Pothole Detection – The Outliers")
file = st.file_uploader("Upload road video", type=["mp4", "avi", "mov"])
run_sample = st.button("▶ Run sample demo video")

def risk_score(box, w, h, conf):
    x1, y1, x2, y2 = box
    size = min(((x2 - x1) * (y2 - y1)) / (w * h) * 10, 1)
    closeness = y2 / h
    return conf * (0.6 * closeness + 0.4 * size)

video_path = None
if file:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(file.read())
    tmp.close()
    video_path = tmp.name
elif run_sample:
    if os.path.exists(SAMPLE):
        video_path = SAMPLE
    else:
        st.error("sample_video.mp4 not found in the repository.")

if video_path:
    cap = cv2.VideoCapture(video_path)
    frame_box, info = st.empty(), st.empty()
    i = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        i += 1
        if i % 3 != 0:
            continue

        start = time.time()
        h, w = frame.shape[:2]
        if w > 640:                                   # smaller = faster on server
            frame = cv2.resize(frame, (640, int(h * 640 / w)))
            h, w = frame.shape[:2]

        road_h = int(h * 0.72)                        # ignore the dashboard
        road = frame[:road_h]
        result = model(road, conf=0.25, imgsz=640, verbose=False)[0]

        worst, count = 0, 0
        for b in result.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            conf = float(b.conf[0])
            bw, bh = x2 - x1, y2 - y1
            if bw > w * 0.5 or bw * bh > w * road_h * 0.15:
                continue                              # too big = false alarm
            if y2 < road_h * 0.35:
                continue                              # too far up = horizon
            r = risk_score((x1, y1, x2, y2), w, road_h, conf)
            worst, count = max(worst, r), count + 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"pothole {conf:.2f} risk {r:.2f}",
                        (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2)

        action = ("🔴 SLOW DOWN" if worst > 0.25
                  else "🟡 CAUTION" if worst > 0.10 else "🟢 SAFE")
        fps = 1 / max(time.time() - start, 0.001)
        frame_box.image(frame, channels="BGR")
        info.markdown(f"**Action:** {action} | **Potholes:** {count} | "
                      f"**Risk:** {worst:.2f} | **FPS:** {fps:.1f}")
    cap.release()

import streamlit as st, cv2, tempfile, time
from ultralytics import YOLO

model = YOLO("best.pt")
st.title("🚗 AI Pothole Detection – The Outliers")
file = st.file_uploader("Upload road video", type=["mp4", "avi", "mov"])

def risk_score(box, w, h, conf):
    x1, y1, x2, y2 = box
    size = min(((x2 - x1) * (y2 - y1)) / (w * h) * 10, 1)  # bigger = worse
    closeness = y2 / h                                      # lower in frame = closer
    return conf * (0.6 * closeness + 0.4 * size)

if file:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(file.read())
    cap = cv2.VideoCapture(tmp.name)
    frame_box, info = st.empty(), st.empty()
    i=0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if i % 3 != 0:
            continue
        start = time.time()
        h, w = frame.shape[:2]
        result = model(frame, conf=0.4, imgsz=416, verbose=False)[0]

        worst, count = 0, 0
        for b in result.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            if y2 < h * 0.45:   # skip anything in the top 45% of the image
             continue
            conf = float(b.conf[0])
            r = risk_score((x1, y1, x2, y2), w, h, conf)
            worst, count = max(worst, r), count + 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"pothole {conf:.2f} risk {r:.2f}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        action = "🔴 SLOW DOWN" if worst > 0.25 else "🟡 CAUTION" if worst > 0.1 else "🟢 SAFE"
        fps = 1 / (time.time() - start)
        frame_box.image(frame, channels="BGR")
        info.markdown(f"**Action:** {action} | **Potholes:** {count} | "
                      f"**Risk:** {worst:.2f} | **FPS:** {fps:.1f}")
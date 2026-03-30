import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import tempfile
import os

MODEL_PATH = "runs/detect/runs/helmet_detector/weights/best.pt"
CLASSES    = ["With Helmet", "Without Helmet"]
COLORS     = {"With Helmet": (0,255,0), "Without Helmet": (0,0,255)}

st.set_page_config(page_title="Helmet Compliance Detector", page_icon="🪖", layout="wide")
st.title("🪖 Helmet Compliance Detector")
st.markdown("Upload an image to check helmet compliance.")

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)

uploaded = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded:
    img   = Image.open(uploaded).convert("RGB")
    frame = np.array(img)[:,:,::-1].copy()
    results = model(frame, verbose=False)

    helmet_found  = False
    no_helmet_found = False

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue
            cls_id = int(box.cls[0])
            label  = CLASSES[cls_id]
            color  = COLORS[label]
            if label == "With Helmet":    helmet_found = True
            if label == "Without Helmet": no_helmet_found = True
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            text = f"{label} {conf:.2f}"
            (tw,th),_ = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
            cv2.rectangle(frame,(x1,y1-th-8),(x1+tw,y1),color,-1)
            cv2.putText(frame,text,(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original Image", use_column_width=True)
    with col2:
        st.image(frame[:,:,::-1], caption="Detection Result", use_column_width=True)

    if no_helmet_found:
        st.error("⚠️ NON-COMPLIANT — Worker without helmet detected!")
    elif helmet_found:
        st.success("✅ COMPLIANT — All workers wearing helmets!")
    else:
        st.info("ℹ️ No workers detected.")
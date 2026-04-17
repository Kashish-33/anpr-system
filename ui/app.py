import streamlit as st
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Page config
st.set_page_config(
    page_title="ANPR System",
    page_icon="🚗",
    layout="centered"
)

# Load model — cache karo taaki baar baar load na ho
# Kyun: YOLO aur EasyOCR load hone mein time lagta hai
@st.cache_resource
def load_models():
    model = YOLO('models/best.pt')
    reader = easyocr.Reader(['en'])
    return model, reader

model, reader = load_models()

# Helper functions
def preprocess_plate(img):
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def fix_plate(text):
    text = text.upper().replace(' ', '')
    fixes = {'O':'0','I':'1','Z':'2','S':'5','B':'8','T':'1','N':'M'}
    result = ''
    for i, c in enumerate(text):
        if i < 2 or 4 <= i < 6:
            result += c
        else:
            result += fixes.get(c, c) if c.isalpha() else c
    return result

def run_anpr(image_path):
    image = cv2.imread(image_path)
    results = model(image_path, conf=0.5)
    plates = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        crop = image[y1:y2, x1:x2]
        processed = preprocess_plate(crop)
        ocr_out = reader.readtext(processed,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        raw_text = ' '.join([t for _, t, _ in ocr_out])
        fixed_text = fix_plate(raw_text)
        # Annotate image
        cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(image, fixed_text, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        plates.append({
            'bbox': (x1,y1,x2,y2),
            'detection_conf': round(conf, 2),
            'raw_ocr': raw_text,
            'plate_text': fixed_text
        })
    return image, plates

# ---- UI ----
st.title(" ANPR System")
st.markdown("Automatic Number Plate Recognition using YOLOv8 + EasyOCR")
st.divider()

uploaded_file = st.file_uploader(
    "Upload car image",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file:
    # Temp file mein save karo
    # Kyun: OpenCV file path chahta hai, bytes nahi
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    st.image(uploaded_file, caption="Input Image", use_column_width=True)

    with st.spinner("Detecting plates..."):
        annotated_img, plates = run_anpr(tmp_path)

    os.unlink(tmp_path)  # Temp file delete karo

    st.divider()

    if plates:
        st.success(f" {len(plates)} plate(s) detected!")

        # Annotated image dikhao
        annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, caption="Detected Plates", use_column_width=True)

        # Results table
        st.subheader("Results")
        for i, p in enumerate(plates):
            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"Plate {i+1}", p['plate_text'])
            with col2:
                st.metric("Detection Confidence", p['detection_conf'])
    else:
        st.error(" No plates detected. Try another image!")
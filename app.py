from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.pipeline import load_artifacts, predict_bgr_image


ARTIFACTS_DIR = Path("artifacts")

st.set_page_config(page_title="AT&T Face Recognition", layout="wide")
st.title("AT&T Face Recognition with OpenCV + Backpropagation")
st.write(
    "Upload a face image and the model will predict the closest AT&T subject. "
    "Run `python train.py` first so the saved model is available."
)

if not (ARTIFACTS_DIR / "model.pkl").exists():
    st.warning("No trained model found in artifacts/. Train the model before launching the app.")
    st.stop()

artifacts = load_artifacts(ARTIFACTS_DIR)

uploaded_file = st.file_uploader("Choose a face image", type=["png", "jpg", "jpeg", "bmp", "pgm"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")
    rgb_image = np.array(pil_image)
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    predicted_label, confidence, probabilities = predict_bgr_image(bgr_image, artifacts)

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.image(pil_image, caption="Input image", use_container_width=True)
    with col_right:
        st.metric("Predicted subject", predicted_label)
        st.metric("Confidence", f"{confidence:.2%}")

        top_indices = np.argsort(probabilities)[::-1][:5]
        top_rows = []
        for index in top_indices:
            class_name = artifacts.label_encoder.inverse_transform([index])[0]
            top_rows.append(
                {
                    "subject": class_name,
                    "probability": float(probabilities[index]),
                }
            )
        st.write("Top 5 predictions")
        st.dataframe(top_rows, use_container_width=True)

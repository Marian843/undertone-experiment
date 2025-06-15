import streamlit as st
import pandas as pd
import cv2
import numpy as np
from colorthief import ColorThief
from skimage import color
import tempfile
import os

# Styling
st.markdown("""
    <style>
        html, body {
            font-family: 'Quicksand', sans-serif;
            background-color: white;
        }
        h1 {
            font-size: 3rem;
            color: #ff69b4;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        h2 {
            font-size: 1.4rem;
            color: #d63384;
            font-weight: 500;
            margin-top: 0;
        }
        .stButton>button {
            background-color: #ffb6c1;
            color: white;
            border-radius: 10px;
            font-size: 16px;
            border: none;
            padding: 0.4rem 1.2rem;
        }
        .stButton>button:hover {
            background-color: #ff99aa;
        }
        .result-box {
            background-color: #fff0f5;
            border-left: 6px solid #ff69b4;
            border-radius: 10px;
            padding: 1.5rem;
            margin-top: 1.5rem;
        }
        .result-box p {
            font-size: 17px;
            margin: 0.6rem 0;
            color: #c71585;
        }
        .result-highlight {
            font-weight: 700;
            font-size: 20px;
            color: #d63384;
        }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Quicksand:wght@400;600&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Title section
st.markdown("<h1>Undertone Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h3>Simple UI for testing the capability of existing skin tone detection systems.</h2>", unsafe_allow_html=True)

# Load the CSV with actual labels
labels_df = pd.read_csv("undertone_labels.csv")

# File upload
uploaded_file = st.file_uploader("Upload a UTKFace image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Save uploaded image temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_file.write(uploaded_file.read())
    img_path = temp_file.name

    # Match image name to ground truth
    matching_row = labels_df[labels_df["image_name"] == uploaded_file.name]

    if matching_row.empty:
        st.error("This image name is not found in your undertone_labels.csv.")
    else:
        image = cv2.imread(img_path)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_skin = (0, 20, 70)
        upper_skin = (20, 255, 255)
        skin_mask = cv2.inRange(image_hsv, lower_skin, upper_skin)
        skin = cv2.bitwise_and(image, image, mask=skin_mask)
        cv2.imwrite(img_path, skin)

        color_thief = ColorThief(img_path)
        dominant_rgb = color_thief.get_color(quality=1)

        # Convert RGB to LAB
        rgb_array = np.array(dominant_rgb) / 255.0
        rgb_array = rgb_array.reshape((1, 1, 3))
        lab = color.rgb2lab(rgb_array)
        l, a, b = lab[0, 0]

        # Rule-based logic
       if a > 15 and b < 10:
            predicted = "cool"
        elif b > 15 and a < 10:
            predicted = "warm"
        elif 10 <= a <= 15 and 10 <= b <= 15:
            predicted = "neutral"
        else:
            predicted = "unclassified"


        actual = matching_row["ground_truth_undertone"].values[0]
        correct = predicted == actual

        # Cute result panel
        st.markdown(f"""
            <div class="result-box">
                <p><span class="result-highlight">Predicted Undertone:</span> {predicted.upper()}</p>
                <p><span class="result-highlight">Ground Truth:</span> {actual.upper()}</p>
                <p><span class="result-highlight">Match:</span> {"Yes" if correct else "No"}</p>
                <p><strong>Dominant RGB:</strong> {dominant_rgb}</p>
                <p><strong>CIELAB:</strong> L={l:.2f}, a={a:.2f}, b={b:.2f}</p>
            </div>
        """, unsafe_allow_html=True)

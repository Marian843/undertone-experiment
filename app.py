import streamlit as st
import numpy as np
import cv2
from colorthief import ColorThief
from skimage import color
import tempfile
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Initialize session state for history
if "history" not in st.session_state:
    st.session_state.history = []

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fedoroka&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Fedoroka', sans-serif;
        background-color: #ffe6f0;
    }
    .stButton>button {
        background-color: #ff99cc;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    .result-box {
        background-color: #fff0f5;
        border-radius: 10px;
        padding: 1em;
        margin-top: 1em;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Rule-Based CIELAB Analysis in Undertone Skin Detection")

uploaded_file = st.file_uploader("Upload a facial image", type=["jpg", "jpeg", "png"])
gt_label = st.selectbox("Select your actual (ground truth) undertone:", ["Cool", "Neutral", "Warm"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    analyze = st.button("Analyze Undertone")

    if analyze:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name

        image = cv2.imread(temp_path)
        if image is None:
            st.error("Image could not be read.")
        else:
            # Skin segmentation using HSV
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_skin = (0, 20, 70)
            upper_skin = (20, 255, 255)
            mask = cv2.inRange(image_hsv, lower_skin, upper_skin)
            skin = cv2.bitwise_and(image, image, mask=mask)

            # Save segmented skin to temp file
            cv2.imwrite(temp_path, skin)

            try:
                color_thief = ColorThief(temp_path)
                dominant_rgb = color_thief.get_color(quality=1)
                rgb_array = np.array(dominant_rgb) / 255.0
                lab = color.rgb2lab(rgb_array.reshape((1, 1, 3)))
                _, a, b = lab[0, 0]

                # Rule-based prediction
                if a > 15 and b < 20:
                    pred = "Cool"
                elif b > 20 and a < 15:
                    pred = "Warm"
                elif 10 <= a <= 20 and 15 <= b <= 25:
                    pred = "Neutral"
                else:
                    pred = "Unclassified"

                # Save to session state
                st.session_state.history.append({
                    "actual": gt_label,
                    "predicted": pred
                })

                # Result box
                st.markdown("""
                <div class="result-box">
                    <h4>Dominant RGB: <code>{}</code></h4>
                    <h4>CIELAB (a, b): <code>({:.2f}, {:.2f})</code></h4>
                    <h4>Predicted Undertone: <code>{}</code></h4>
                    <h4>Ground Truth Match: <code>{}</code></h4>
                </div>
                """.format(dominant_rgb, a, b, pred, 'Yes' if pred.lower() == gt_label.lower() else 'No'), unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Color extraction failed: {e}")

        os.remove(temp_path)

# Show summary metrics and visual confusion matrix if enough history exists
if len(st.session_state.history) >= 3:
    st.markdown("## Summary Metrics (All Images This Session)")

    labels = ["cool", "warm", "neutral", "unclassified"]
    actuals = [x["actual"].lower() for x in st.session_state.history]
    preds = [x["predicted"].lower() for x in st.session_state.history]

    class_metrics = {}
    for label in labels:
        TP = sum(1 for a, p in zip(actuals, preds) if a == label and p == label)
        FP = sum(1 for a, p in zip(actuals, preds) if a != label and p == label)
        FN = sum(1 for a, p in zip(actuals, preds) if a == label and p != label)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        class_metrics[label] = (precision, recall, f1)

    accuracy = sum(1 for a, p in zip(actuals, preds) if a == p) / len(actuals)

    st.markdown("<div class='result-box'><strong>Overall Accuracy:</strong> {:.0f}%</div>".format(accuracy * 100), unsafe_allow_html=True)

    st.markdown("### Per-Class Metrics:")
    for label in labels:
        p, r, f = class_metrics[label]
        st.markdown(f"**{label.capitalize()}** — Precision: {p:.0%}, Recall: {r:.0%}, F1 Score: {f:.0%}")

    # Confusion Matrix
    cm = confusion_matrix(actuals, preds, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=sns.light_palette("#ff99cc", as_cmap=True),
        cbar=False,
        linewidths=1,
        linecolor="white",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=12, color="#cc0066")
    ax.set_ylabel("Actual", fontsize=12, color="#cc0066")
    ax.set_title("Confusion Matrix", fontsize=14, color="#cc0066")
    plt.xticks(color="#cc0066")
    plt.yticks(color="#cc0066")
    st.pyplot(fig)

# Option to reset
if st.button("Reset Session"):
    st.session_state.history = []
    st.success("Session has been reset.")

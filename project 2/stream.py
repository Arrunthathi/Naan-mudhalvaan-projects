# tumor_detection_app_from_folder.py
import streamlit as st
import os
import random
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Brain Tumor Auto Detector")
st.title("🧠 Brain Tumor Auto Detection & Severity Estimation")

# Update this to your dataset path
BASE_PATH = r"C:\Users\Arunthathi\Desktop\aadhi\project 2\Training"
CATEGORIES = ["notumor", "pituitary"]
IMG_SIZE = 256

# --- Dummy segmentation logic (replace with your real model) ---
def dummy_segment_tumor(image_np):
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    center = (IMG_SIZE // 2, IMG_SIZE // 2)
    radius = IMG_SIZE // 6
    cv2.circle(mask, center, radius, 255, -1)
    return mask

def calculate_tumor_percentage(mask):
    tumor_pixels = np.sum(mask > 0)
    total_pixels = mask.size
    percent = (tumor_pixels / total_pixels) * 100
    return percent

def classify_danger(percent):
    if percent < 2:
        return "Not Dangerous", "🟢"
    elif percent < 8:
        return "Moderately Dangerous", "🟡"
    else:
        return "Highly Dangerous", "🔴"

# --- Function to load one random image ---
def get_random_image():
    selected_class = random.choice(CATEGORIES)
    folder = os.path.join(BASE_PATH, selected_class)
    files = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not files:
        return None, None
    chosen_file = random.choice(files)
    image_path = os.path.join(folder, chosen_file)
    image = Image.open(image_path).convert("RGB")
    return image, selected_class

# --- Process one random MRI from the folders ---
st.subheader("📂 Random MRI Selection from Dataset")

if st.button("🎯 Detect Tumor from Random Image"):
    image, label = get_random_image()
    
    if image is None:
        st.error("No images found in the folders.")
    else:
        image_resized = image.resize((IMG_SIZE, IMG_SIZE))
        image_np = np.array(image_resized)

        st.image(image_np, caption=f"Selected Image ({label})", use_column_width=True)

        # Segment and analyze
        mask = dummy_segment_tumor(image_np)
        tumor_percent = calculate_tumor_percentage(mask)
        severity, icon = classify_danger(tumor_percent)

        st.subheader("🧪 Detection Result")
        st.write(f"**Class Label:** {label}")
        st.write(f"**Affected Area:** {tumor_percent:.2f}%")
        st.write(f"**Severity:** {severity} {icon}")

        # Show overlay
        overlay = image_np.copy()
        overlay[mask > 0] = [255, 0, 0]
        st.image(overlay, caption="Detected Tumor Region", use_column_width=True)

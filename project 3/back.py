import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Virtual Background Replacement")

st.markdown("""
Upload a photo of yourself or your family, and replace the background with a new image —
like how virtual backgrounds work in video calls!
""")

# Upload original image
main_img_file = st.file_uploader("fly", type=["jpg", "jpeg", "png"])

# Upload new virtual background
bg_img_file = st.file_uploader("Trees", type=["jpg", "jpeg", "png"])

def replace_background(foreground_img, background_img):
    foreground_img = cv2.resize(foreground_img, (640, 480))
    background_img = cv2.resize(background_img, (640, 480))

    # Convert to HSV for masking background (assuming blue sky or light background)
    hsv = cv2.cvtColor(foreground_img, cv2.COLOR_BGR2HSV)
    lower_sky = np.array([90, 20, 80])
    upper_sky = np.array([135, 255, 255])
    mask = cv2.inRange(hsv, lower_sky, upper_sky)

    # Clean the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.medianBlur(mask, 5)

    # Invert to keep subject
    inv_mask = cv2.bitwise_not(mask)

    # Apply masks
    fg = cv2.bitwise_and(foreground_img, foreground_img, mask=inv_mask)
    bg = cv2.bitwise_and(background_img, background_img, mask=mask)

    # Combine both
    result = cv2.add(fg, bg)
    return result

# Process if both images are uploaded
if main_img_file and bg_img_file:
    # Read and decode images
    main_img_bytes = np.asarray(bytearray(main_img_file.read()), dtype=np.uint8)
    bg_img_bytes = np.asarray(bytearray(bg_img_file.read()), dtype=np.uint8)

    main_img = cv2.imdecode(main_img_bytes, 1)
    bg_img = cv2.imdecode(bg_img_bytes, 1)

    output = replace_background(main_img, bg_img)

    st.subheader("Virtual Background Applied")
    st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), channels="RGB")

    # Save and download
    result_img = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    st.download_button("Download Result", result_img.tobytes(), file_name="virtual_background.jpg", mime="image/jpeg")

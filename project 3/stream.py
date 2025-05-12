import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ------------------ Title and Intro ------------------
st.title("🧍 Virtual Background Replacement")
st.markdown("""
This app lets you **remove and replace the background** from your uploaded image . Great for creating fun or professional portraits!

👉 Upload your photo with a background (preferably with blue sky or solid color)  
👉 Upload a new background image  
👉 Get a composite image with the background replaced!
""")

# ------------------ Upload Section ------------------
st.header("📷 Step 1: Upload Your Photo")
main_img_file = st.file_uploader("fly (with background)", type=["jpg", "jpeg", "png"])

st.header("🌄 Step 2: Upload New Background")
bg_img_file = st.file_uploader("Tree", type=["jpg", "jpeg", "png"])

# ------------------ Background Replacement Function ------------------
def replace_background(foreground_img, background_img):
    """Replaces sky/solid background in foreground image with another background image."""
    # Resize images to match
    foreground_img = cv2.resize(foreground_img, (640, 480))
    background_img = cv2.resize(background_img, (640, 480))

    # Convert to HSV color space for color-based segmentation
    hsv = cv2.cvtColor(foreground_img, cv2.COLOR_BGR2HSV)

    # Define color range for blue sky or similar backgrounds
    lower_sky = np.array([90, 20, 80])   # Lower HSV bound for blue
    upper_sky = np.array([135, 255, 255])  # Upper HSV bound

    # Create mask for background
    mask = cv2.inRange(hsv, lower_sky, upper_sky)

    # Clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.medianBlur(mask, 5)

    # Invert mask to get foreground
    inv_mask = cv2.bitwise_not(mask)

    # Extract subject and background regions
    fg = cv2.bitwise_and(foreground_img, foreground_img, mask=inv_mask)
    bg = cv2.bitwise_and(background_img, background_img, mask=mask)

    # Combine them
    final = cv2.add(fg, bg)
    return final

# ------------------ Processing and Display ------------------
if main_img_file and bg_img_file:
    # Convert uploaded files to OpenCV images
    main_bytes = np.asarray(bytearray(main_img_file.read()), dtype=np.uint8)
    bg_bytes = np.asarray(bytearray(bg_img_file.read()), dtype=np.uint8)

    main_img = cv2.imdecode(main_bytes, 1)
    bg_img = cv2.imdecode(bg_bytes, 1)

    # Process the images
    st.info("🔄 Processing image. Please wait...")
    result = replace_background(main_img, bg_img)

    # Show the result
    st.subheader("✅ Result: Background Replaced")
    st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Final Output", use_column_width=True)

    # Download button
    output_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    st.download_button("⬇️ Download Result", output_image.tobytes(), file_name="virtual_background.jpg", mime="image/jpeg")

elif main_img_file or bg_img_file:
    st.warning("Please upload **both** images to proceed.")

# Footer
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit and OpenCV")

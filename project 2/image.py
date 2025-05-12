import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Path to the pituitary tumor image folder
tumor_folder = r'C:\Users\Arunthathi\Desktop\aadhi\project 2\Training\pituitary'
image_files = [f for f in os.listdir(tumor_folder) if f.endswith(('.png', '.jpg', '.jpeg'))][:3]  # select 3 images

for i, filename in enumerate(image_files):
    path = os.path.join(tumor_folder, filename)
    
    # Load the image in grayscale
    img = cv2.imread(path, 0)

    # Preprocess: blur to reduce noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Threshold to isolate bright regions (tumor)
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)

    # Find contours (potential tumor borders)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert grayscale to color to highlight contours in color
    output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Draw contours in red
    cv2.drawContours(output, contours, -1, (0, 0, 255), 2)

    # Show the original and highlighted image
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")

    plt.subplot(1, 2, 2)
    plt.imshow(output)
    plt.title("Tumor Highlighted")
    plt.axis('off')
    plt.show()

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from PIL import Image
import torch
import sys

# Assuming detect_dual.py has a run_detection function
from detect_dual import run as yolo_run_detection


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
new_env = os.environ.copy()

def add_logo(logo_path, size=(200, 150)):
    logo = Image.open(logo_path)
    logo = logo.resize(size)
    st.image(logo, use_column_width=False)

def run_detection(image_path):
    # Directly call the function from detect_dual.py
    yolo_run_detection(
        source=image_path,
        imgsz=(640, 640),
        device="cpu",
        weights="models/detect/yolov9tr.pt",
        name="yolov9_c_640_detect",
        exist_ok=True)
    #Find the output image
    output_dir = "runs/detect/yolov9_c_640_detect"
    output_image = os.path.join(output_dir, os.path.basename(image_path))
    return output_image
    
    

def main():
    st.title("YOLO9tr Object Detection")

    # Add the research center logo at the top of the app
    add_logo("logo_ai.jpg")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_path = "temp_image.jpg"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    else:
        image_path = "United_States_000502.jpg"  # Default image

    st.image(image_path, caption="Image for Detection", use_column_width=True)

    if st.button("Run Detection"):
        with st.spinner("Running detection..."):
            output_image = run_detection(image_path)
        
        # Display the output image
        st.image(output_image, caption="Detection Result", use_column_width=True)

if __name__ == "__main__":
    main()

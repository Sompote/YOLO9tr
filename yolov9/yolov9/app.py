import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import subprocess
import os

def run_detection(uploaded_file):
    # Save the uploaded file temporarily
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Run the detection command
    command = [
        "python", "detect_dual.py",
        "--source", "temp_image.jpg",
        "--img", "640",
        "--device", "cpu",
        "--weights", "models/detect/yolov9tr.pt",
        "--name", "yolov9_c_640_detect",
        "--exist" "-ok"
    ]
    
    subprocess.run(command, check=True)
    
    # Find the output image
    output_dir = "runs/detect/yolov9_c_640_detect"
    output_image = os.path.join(output_dir, os.path.basename("temp_image.jpg"))
    
    return output_image

def main():
    st.title("YOLOv9 Object Detection")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Run Detection"):
            with st.spinner("Running detection..."):
                output_image = run_detection(uploaded_file)
            
            # Display the output image
            st.image(output_image, caption="Detection Result", use_column_width=True)
            
            # Optional: Display the image using matplotlib
            fig, ax = plt.subplots()
            img = mpimg.imread(output_image)
            ax.imshow(img)
            ax.axis('off')
            st.pyplot(fig)

if __name__ == "__main__":
    main()
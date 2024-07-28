import gradio as gr
import subprocess
import os
from PIL import Image
import torch

def add_logo(img):
    logo = Image.open('logoAI.png')
    logo = logo.resize((200, 150))
    img_with_logo = Image.new('RGB', (img.width, img.height + logo.height))
    img_with_logo.paste(logo, (0, 0))
    img_with_logo.paste(img, (0, logo.height))
    return img_with_logo

def run_detection(image):
    # Save the uploaded image temporarily
    image_path = "temp_image.jpg"
    image.save(image_path)

    env = os.environ.copy()
    env['PYTHONPATH'] = '/mount/src/yolo9tr/'
    
    # Run the detection command
    command = [
        "python", "detect_dual.py",
        "--source", image_path,
        "--img", "640",
        "--device", "cpu",
        "--weights", "models/detect/yolov9tr.pt",
        "--name", "yolov9_c_640_detect",
        "--exist-ok"
    ]
    subprocess.run(command, check=True, env=os.environ)

    # Find the output image
    output_dir = "runs/detect/yolov9_c_640_detect"
    output_image = os.path.join(output_dir, os.path.basename(image_path))
    
    # Add logo to the output image
    output_with_logo = add_logo(Image.open(output_image))
    
    return output_with_logo

def main():
    input_image = gr.Image(type="pil", label="Upload an image")
    output_image = gr.Image(type="pil", label="Detection Result")

    iface = gr.Interface(
        fn=run_detection,
        inputs=input_image,
        outputs=output_image,
        title="YOLO9tr Object Detection",
        description="Upload an image to perform object detection using YOLO9tr.",
        examples=[["United_States_000502.jpg"]]
    )

    iface.launch()

if __name__ == "__main__":
    main()
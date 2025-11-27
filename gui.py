import gradio as gr
import cv2
from lpr import detect_and_recognize_license_plate

def process(image):
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    text, plate_img = detect_and_recognize_license_plate(image_bgr)

    if isinstance(plate_img, str):
        return None, text

    plate_display = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
    return plate_display, text

gr.Interface(
    fn=process,
    inputs=gr.Image(type="numpy"),
    outputs=[gr.Image(type="numpy"), "text"],
    title="Indian License Plate Recognition",
    description="Upload a vehicle image to detect and recognize the license plate."
).launch()

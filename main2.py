import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os

# Streamlit app
st.title("AI-Powered: Road sign Clasiification & General Objects  Detection")

st.write("Upload an image to detect road signs using YOLO.")

# Sidebar: Select detection type
st.sidebar.title("Detection Settings")
detection_type = st.sidebar.selectbox(
    "Choose Detection Type",
    ["Traffic Sign Classification", "General Object Detection"]
)

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Load YOLO models
traffic_sign_model_path = r'D:\\Sem 7\\Final Project\\Udacityresult\\Road sign detection result\\weights\\best.pt'
general_object_model_path = r'C:\\Users\\barot nandni\\Downloads\\best (1).pt'

# Ensure models exist
if not os.path.exists(traffic_sign_model_path) or not os.path.exists(general_object_model_path):
    st.error("Model file not found. Check the paths to the model files.")
    st.stop()

traffic_model = YOLO(traffic_sign_model_path)
general_model = YOLO(general_object_model_path)

# Process uploaded image
if uploaded_file is not None:
    # Read and process image
    image = np.array(Image.open(uploaded_file).convert("RGB"))

    # Display the original image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform detection based on selected type
    if detection_type == "Traffic Sign Classification":
        st.subheader("Detecting Traffic Signs...")
        results = traffic_model(image)

        # Draw bounding boxes on the image
        annotated_image = results[0].plot()  # YOLOv8 visualization
        st.image(annotated_image, caption="Traffic Sign Detection Results", use_column_width=True)

        # Display detected classes
        st.subheader("Detected Classes:")
        detected_classes = []
        for box in results[0].boxes:  # Iterate through bounding boxes
            class_id = int(box.cls.cpu().numpy())  # Extract class ID
            class_name = traffic_model.names[class_id]  # Get class name
            detected_classes.append(class_name)

        # Display each class on a new line
        for class_name in set(detected_classes):
            st.write(f"- {class_name}")

    elif detection_type == "General Object Detection":
        st.subheader("Detecting General Objects...")
        results = general_model(image)

        # Draw bounding boxes on the image
        annotated_image = results[0].plot()  # YOLOv8 visualization
        st.image(annotated_image, caption="Object Detection Results", use_column_width=True)

        # Display detected classes
        st.subheader("Detected Classes:")
        detected_classes = []
        for box in results[0].boxes:  # Iterate through bounding boxes
            class_id = int(box.cls.cpu().numpy())  # Extract class ID
            class_name = general_model.names[class_id]  # Get class name
            detected_classes.append(class_name)

        # Display each class on a new line
        for class_name in set(detected_classes):
            st.write(f"- {class_name}")

st.sidebar.info(
    """
    Explore cutting-edge AI capabilities with our dual-purpose detection app! Harnessing the power of YOLOv8 models:
    - **Road Sign Classification**: Optimized for multi-class Classification with YOLOv8m, perfect for intricate traffic scenarios.
    - **General Object Detection**: Leveraging YOLOv8n for swift and accurate identification of objects like cars, pedestrians, and more.

    Upload an image, analyze the results, and experience the power of AI-driven vision technology for autonomous systems and beyond.
    """
)
# Footer
st.sidebar.info("Streamlit app for road sign Classification and object detection using YOLO.")

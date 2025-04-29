import streamlit as st
import cv2
import numpy as np
from PIL import Image
from clahe_processing import apply_clahe
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('corrosion_detection_model.h5')

st.title("Corrosion Detection System")

option = st.radio("Select Input Method:", 
                  ("Upload Image", "IP Camera"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        # Process with CLAHE
        processed = apply_clahe(image_np)
        
        # Display original and processed
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        with col2:
            st.image(processed, caption="Processed Image", use_column_width=True)
        
        # Prepare for prediction
        resized = cv2.resize(processed, (128, 128))
        input_tensor = np.expand_dims(resized, axis=0) / 255.0
        
        # Predict
        prediction = model.predict(input_tensor)
        corrosion_prob = prediction[0][0]
        result = "Corroded" if corrosion_prob > 0.5 else "Not Corroded"
        
        st.success(f"Prediction: {result} (Confidence: {corrosion_prob:.2f})")

else:  # IP Camera option
    st.warning("Make sure your mobile is connected to the same network")
    ip_address = st.text_input("Enter your mobile IP address:", "192.168.x.x")
    
    if st.button("Start IP Camera Stream"):
        cap = cv2.VideoCapture(f"http://192.168.1.3:8080/video")
        frame_placeholder = st.empty()
        stop_button = st.button("Stop Stream")
        
        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to connect to camera")
                break
                
            # Process frame
            processed = apply_clahe(frame)
            resized = cv2.resize(processed, (128, 128))
            input_tensor = np.expand_dims(resized, axis=0) / 255.0
            
            # Predict
            prediction = model.predict(input_tensor)
            corrosion_prob = prediction[0][0]
            result = "Not Corroded" if corrosion_prob > 0.5 else "Corroded"
            
            # Annotate frame
            cv2.putText(frame, f"Status: {result} ({corrosion_prob:.2f})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display in Streamlit
            frame_placeholder.image(frame, channels="BGR")
            
            if stop_button:
                break
                
        cap.release()

# Add to your streamlit_app.py
st.sidebar.header("Debugging Options")
show_processing = st.sidebar.checkbox("Show processing steps")
adjust_clahe = st.sidebar.checkbox("Adjust CLAHE parameters")

if adjust_clahe:
    clip_limit = st.sidebar.slider("CLAHE Clip Limit", 0.5, 4.0, 2.0, 0.5)
    grid_size = st.sidebar.slider("CLAHE Grid Size", 4, 32, 8, 4)
else:
    clip_limit, grid_size = 2.0, 8

# Modify your processing call
processed = apply_clahe(image_np, clip_limit=clip_limit, grid_size=(grid_size, grid_size))
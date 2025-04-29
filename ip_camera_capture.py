import cv2
import numpy as np
from clahe_processing import apply_clahe  # We'll create this next

def process_ip_camera(model):
    url = "http://192.168.1.3:8080/video"
    cap = cv2.VideoCapture(url)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
            
        # Preprocess
        processed = apply_clahe(frame)
        
        # Resize to model input size (128x128 based on your model summary)
        resized = cv2.resize(processed, (128, 128))
        input_tensor = np.expand_dims(resized, axis=0) / 255.0
        
        # Predict
        prediction = model.predict(input_tensor)
        corrosion_prob = prediction[0][0]
        result = "Corroded" if corrosion_prob > 0.5 else "Not Corroded"
        
        # Display
        cv2.putText(frame, f"Status: {result} ({corrosion_prob:.2f})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Corrosion Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
import cv2
import numpy as np

def apply_clahe(image, clip_limit=2.0, grid_size=(8,8)):
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    cl = clahe.apply(l)
    
    # Merge channels and convert back to BGR
    processed_lab = cv2.merge((cl, a, b))
    processed_bgr = cv2.cvtColor(processed_lab, cv2.COLOR_LAB2BGR)
    
    return processed_bgr
# Add to your processing pipeline
def debug_visualization(image, processed):
    cv2.imshow("Original", image)
    cv2.imshow("After CLAHE", processed)
    cv2.imshow("Resized for Model", cv2.resize(processed, (128, 128)))
    cv2.waitKey(100)  # Display for 100ms

# Check if converting to grayscale improves results
def apply_clahe_grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)
import cv2
import numpy as np
import os

def test_with_validation_images(model, test_folder="images"):
    """Test model with known validation images"""
    test_images = {
        "corroded_1.jpg": 1,
        "corroded_2.jpg": 1,
        "clean_1.jpg": 0,
        "clean_2.jpg": 0
    }
    
    results = {}
    for img_path, true_label in test_images.items():
        full_path = os.path.join(test_folder, img_path)
        if not os.path.exists(full_path):
            continue
            
        image = cv2.imread(full_path)
        if image is None:
            continue
            
        processed = apply_clahe(image)
        resized = cv2.resize(processed, (128, 128))
        input_tensor = np.expand_dims(resized, axis=0) / 255.0
        pred = model.predict(input_tensor)[0][0]
        
        results[img_path] = {
            "prediction": "Corroded" if pred > 0.5 else "Clean",
            "confidence": pred,
            "correct": (pred > 0.5) == true_label
        }
    
    return results

def verify_model_health(model):
    """Comprehensive model verification"""
    print("\n=== Model Health Check ===")
    
    # 1. Test with validation images
    val_results = test_with_validation_images(model)
    
    # 2. Check input/output shapes
    test_input = np.random.rand(1, 128, 128, 3)
    test_output = model.predict(test_input)
    print(f"\nInput/Output Shapes:")
    print(f"Expected input: (None, 128, 128, 3), Actual: {test_input.shape}")
    print(f"Expected output: (None, 1), Actual: {test_output.shape}")
    
    # 3. Verify predictions are within bounds
    print("\nPrediction Range Check:")
    print(f"All predictions should be 0-1. Test result: {test_output[0][0]:.4f}")
    
    return val_results
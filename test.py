import cv2

cap = cv2.VideoCapture(2)  # Try 0, 1, 2

if not cap.isOpened():
    print("Camera not found")
else:
    print("Camera working!")
    ret, frame = cap.read()
    cv2.imwrite("test.jpg", frame)

cap.release()

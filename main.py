import cv2
import time
from ultralytics import YOLO
import pygame

pygame.mixer.init()
sound = pygame.mixer.Sound("warning sound.mp3")

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Set up the camera
camera = cv2.VideoCapture(0)  # 0 for default camera

while True:
    # Capture frame-by-frame
    ret, frame = camera.read()

    # Perform object detection on the frame
    results = model.predict(frame, show=True)  # show=True to display the results

    for box in results[0].boxes:
        class_id = int(box.cls)  # Get class ID
        class_label = results[0].names[class_id]  # Get class label from class ID
        if (class_label=='dog'):
            sound.play()

    # Wait for 3 seconds before capturing the next image
    time.sleep(3)

    if cv2.waitKey(1) & 0xFF == 32:  # Check for space bar press (ASCII code 32)
        print("Space bar was pressed! Stopping the camera...")
        break

# Release the camera
camera.release()
cv2.destroyAllWindows()


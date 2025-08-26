import numpy as np
import cv2 as cv
import time
import os
import urllib.request

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, filename)
            print(f"Successfully downloaded {filename}")
            return True
        except Exception as e:
            print(f"Error downloading {filename}: {str(e)}")
            return False
    return True

# Model files and URLs
prototxt_url = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/daef68a6c2f5fbb8c88404266aa28180646d17e0/MobileNetSSD_deploy.prototxt"
model_url = "https://github.com/chuanqi305/MobileNet-SSD/raw/daef68a6c2f5fbb8c88404266aa28180646d17e0/MobileNetSSD_deploy.caffemodel"
prototxt_file = "MobileNetSSD_deploy.prototxt"
model_file = "MobileNetSSD_deploy.caffemodel"

if not download_file(prototxt_url, prototxt_file) or not download_file(model_url, model_file):
    print("Failed to download model files. Please check your internet connection.")
    exit()

# Initialize class labels
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Confidence threshold
conf_threshold = 0.5

print("Loading model...")
try:
    net = cv.dnn.readNetFromCaffe(prototxt_file, model_file)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    print("Please make sure the model files are in the correct directory.")
    exit()

# Initialize webcam
cap = cv.VideoCapture(0)

# Set camera resolution
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# FPS calculation
prev_time = time.time()

if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Press 'q' to exit")

try:
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Can't receive frame. Exiting...")
            break

        # Get frame dimensions
        (h, w) = frame.shape[:2]
        
        # Preprocess the frame for object detection
        blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        
        # Pass the blob through the network
        net.setInput(blob)
        detections = net.forward()

        # Loop over the detections
        for i in range(detections.shape[2]):
            # Extract the confidence (probability) associated with the prediction
            confidence = detections[0, 0, i, 2]
            
            # Filter out weak detections
            if confidence > conf_threshold:
                # Extract the index of the class label
                idx = int(detections[0, 0, i, 1])
                class_name = CLASSES[idx]
                
                # Compute the bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure bounding boxes are within frame dimensions
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                
                # Draw the prediction on the frame
                label = "{}: {:.2f}%".format(class_name, confidence * 100)
                cv.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
                
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv.putText(frame, label, (startX, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Calculate and display FPS
        # curr_time = time.time()
        # fps = 1 / (curr_time - prev_time)
        # prev_time = curr_time
        # cv.putText(frame, f"FPS: {int(fps)}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the frame with detections
        cv.imshow('Object Detection', frame)
        
        # Break on 'q'
        if cv.waitKey(1) == ord('q'):
            break

finally:
    cap.release()
    cv.destroyAllWindows()
import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf

# Load YOLO model for person detection
yolo_model = YOLO("yolov8n.pt")  # Ensure you have the YOLO model weights

# Load the pre-trained emotion detection model
emotion_model = tf.keras.models.load_model("emotion_model.h5")

# Behavior/emotion classes
behavior_classes = ['Boredom', 'Sleepy', 'Confusion', 'Yawning', 'Frustrated', 'Engagement']

def preprocess_frame(frame, target_size=(32, 32)):
    """
    Preprocess a single video frame.
    Resize, normalize, and ensure correct input shape.
    """
    frame_resized = cv2.resize(frame, target_size)  # Resize to target size
    frame_normalized = frame_resized / 255.0  # Normalize pixel values to [0, 1]
    return np.expand_dims(frame_normalized, axis=0)  # Add batch dimension (1, 32, 32, 3)

def analyze_real_time_behavior():
    """
    Analyze real-time behavior using YOLO for person detection and emotion_model for classification.
    """
    cap = cv2.VideoCapture(0)  # Use webcam input

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if no frame is captured

        # Use YOLO to detect persons
        results = yolo_model.predict(frame, conf=0.5, verbose=False)
        detections = results[0].boxes.xyxy.cpu().numpy()  # Extract bounding boxes

        for box in detections:
            x1, y1, x2, y2 = map(int, box[:4])  # Extract coordinates
            person_frame = frame[y1:y2, x1:x2]  # Crop the person region

            if person_frame.size == 0:
                continue

            preprocessed_frame = preprocess_frame(person_frame)

            # Use emotion_model to predict the behavior
            predictions = emotion_model.predict(preprocessed_frame, verbose=0)
            predicted_class = behavior_classes[np.argmax(predictions)]

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, predicted_class, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the frame with bounding boxes
        cv2.imshow("Real-Time Behavior Analysis", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the real-time behavior analysis
analyze_real_time_behavior()

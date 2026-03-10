import cv2
import numpy as np
import tensorflow as tf
from collections import Counter

# Load the pre-trained model
model = tf.keras.models.load_model("emotion_model.h5")

# Updated behavior classes
behavior_classes = ['Boredom', 'Sleepy', 'Confusion', 'Yawning', 'Frustrated', 'Engagement']

def preprocess_frame(frame, target_size=(32, 32)):
    """
    Preprocess a single video frame.
    Resize, normalize, and ensure correct input shape.
    """
    frame_resized = cv2.resize(frame, target_size)  # Resize to (32, 32)
    frame_normalized = frame_resized / 255.0  # Normalize pixel values to [0, 1]
    return np.expand_dims(frame_normalized, axis=0)  # Add batch dimension (1, 32, 32, 3)

def analyze_real_time():
    """
    Analyze behavior in real-time using the webcam.
    """
    cap = cv2.VideoCapture(0)  # Open the webcam (0 is usually the default camera)

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    behavior_counts = Counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Preprocess the frame
        preprocessed_frame = preprocess_frame(frame)

        # Predict behavior for the frame
        predictions = model.predict(preprocessed_frame, verbose=0)
        predicted_class = behavior_classes[np.argmax(predictions)]

        # Update behavior counts
        behavior_counts[predicted_class] += 1

        # Display the result on the frame
        cv2.putText(frame, f"Behavior: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (0, 0), (300, 50), (0, 0, 0), -1)  # Semi-transparent background
        cv2.putText(frame, f"Behavior: {predicted_class}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the frame
        cv2.imshow("Real-Time Behavior Analysis", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

# Run the real-time behavior analysis
analyze_real_time()

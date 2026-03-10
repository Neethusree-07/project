import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
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

def analyze_video(video_path):
    """
    Analyze the video and classify behaviors frame by frame.
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    behavior_counts = Counter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if video ends

        frame_count += 1
        preprocessed_frame = preprocess_frame(frame)

        # Predict behavior for the frame
        predictions = model.predict(preprocessed_frame, verbose=0)
        predicted_class = behavior_classes[np.argmax(predictions)]
        behavior_counts[predicted_class] += 1

    cap.release()
    return behavior_counts, frame_count

def plot_behaviors(behavior_counts, frame_count):
    """
    Plot the detected behaviors as a bar chart.
    """
    # Normalize counts to percentages
    behaviors = list(behavior_counts.keys())
    counts = list(behavior_counts.values())
    percentages = [(count / frame_count) * 100 for count in counts]

    plt.figure(figsize=(10, 6))
    plt.bar(behaviors, percentages, color='skyblue')
    plt.xlabel("Behaviors")
    plt.ylabel("Percentage of Frames (%)")
    plt.title("Behavior Distribution in Video")
    plt.show()

# Input: Path to the video file
video_path = r"C:\\Users\\Saray\\OneDrive\\Pictures\\Camera Roll\\Testing_Clip_1.mp4"

# Analyze video and get behavior counts
behavior_counts, frame_count = analyze_video(video_path)

# Plot the results
plot_behaviors(behavior_counts, frame_count)

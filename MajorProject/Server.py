import cv2
import tempfile
import os
import numpy as np
import tensorflow as tf
from collections import Counter
from ultralytics import YOLO
from flask import Flask, render_template, Response, request, redirect, url_for, session, jsonify
from werkzeug.utils import secure_filename
from sklearn.metrics import *
model = tf.keras.models.load_model("emotion_model.h5")
yolo_model = YOLO("yolov8n.pt")  # Ensure you have the YOLO model weights

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
behavior_classes = ['Boredom', 'Sleepy', 'Confusion', 'Yawning', 'Frustrated', 'Engagement']


def preprocess_frame(frame, target_size=(32, 32)):
    frame_resized = cv2.resize(frame, target_size)
    frame_normalized = frame_resized / 255.0
    return np.expand_dims(frame_normalized, axis=0)

def generate_frames():
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    behavior_counts = Counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        preprocessed_frame = preprocess_frame(frame)
        predictions = model.predict(preprocessed_frame, verbose=0)
        predicted_class = behavior_classes[np.argmax(predictions)]
        behavior_counts[predicted_class] += 1
       
        
        # Display prediction on the frame
        cv2.putText(frame, f"Behavior: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (0, 0), (300, 50), (0, 0, 0), -1)
        cv2.putText(frame, f"Behavior: {predicted_class}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

def generate_video_frames():
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    behavior_counts = Counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

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
            predictions = model.predict(preprocessed_frame, verbose=0)
            predicted_class = behavior_classes[np.argmax(predictions)]

            # Simulate true labels (for real-world, you'd have these as ground truth)
           

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, predicted_class, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the frame with bounding boxes
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

app = Flask(__name__)
app.secret_key = 'SARAYU@05'  # Change this to a strong secret key
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template("home.html", logged_in=session.get("user_id") is not None)

@app.route('/login', methods=['POST'])
def login():
    user_id = request.form.get("userid")
    password = request.form.get("password")
    
    if user_id =='sarayu' and password=='spars':
        session['user_id'] = user_id
        return redirect(url_for('home'))
    else:
        return "Invalid credentials. <a href='/'>Try again</a>"

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('home'))

@app.route('/individual-testing')
def individual_testing():
    if 'user_id' not in session:
        return redirect(url_for('home'))
    return render_template('individual_testing.html')

@app.route('/live-testing')
def live_testing():
    if 'user_id' not in session:
        return redirect(url_for('home'))
    return render_template('live_testing.html')

@app.route('/multiple-video-testing')
def multiple_video_testing():
    if 'user_id' not in session:
        return redirect(url_for('home'))
    return render_template('multiple_video_testing.html')
@app.route('/video_group_feed')
def video_group_feed():
    return Response(generate_video_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    analyzed_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        preprocessed_frame = preprocess_frame(frame)
        predictions = model.predict(preprocessed_frame, verbose=0)
        predicted_class = behavior_classes[np.argmax(predictions)]
        
        # Draw predicted class on frame
        cv2.putText(frame, predicted_class, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        analyzed_frames.append(frame)
    
    cap.release()
    return analyzed_frames

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return redirect(request.url)
    
    file = request.files['video']
    if file.filename == '':
        return redirect(request.url)
    
    # Create a temporary file to save the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        file.save(temp_video.name)

        # Return the MJPEG stream for the video
        return Response(analyze_video_and_stream(temp_video.name),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

def analyze_video_and_stream(video_path):
    """
    Read the video, process each frame, and yield it as a MJPEG stream.
    """
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        preprocessed_frame = preprocess_frame(frame)
        predictions = model.predict(preprocessed_frame, verbose=0)
        predicted_class = behavior_classes[np.argmax(predictions)]
        
        # Draw predicted class on frame
        cv2.putText(frame, predicted_class, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Convert frame to JPEG format
        _, jpeg = cv2.imencode('.jpg', frame)
        
        # Yield the frame as an MJPEG stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
    
    cap.release()

app.run()

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import mediapipe as mp
import tempfile
import cv2
import numpy as np

from flask import Flask, render_template, Response, request
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load models
age_model = load_model('age_model_50epochs.h5')
gender_model = load_model('gender_model_50epochs.h5')
emotion_model = load_model('emotion_model_10epochs.h5')

# Initialize MediaPipe components
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Function to map scalar to emotion
def map_scalar_to_emotion(scalar_value):
    if scalar_value < 1.0:
        return 'Angry'
    elif scalar_value < 2.0:
        return 'Disgust'
    elif scalar_value < 3.0:
        return 'Fear'
    elif scalar_value < 4.0:
        return 'Happy'
    elif scalar_value < 5.0:
        return 'Sad'
    elif scalar_value < 6.0:
        return 'Surprise'
    else:
        return 'Neutral'

# Function to recognize gestures
def recognize_gesture(landmarks):
    if (landmarks[mp_holistic.HandLandmark.THUMB_TIP].y < landmarks[mp_holistic.HandLandmark.WRIST].y and
        landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp_holistic.HandLandmark.THUMB_TIP].y):
        return "Hand Waving"
    elif landmarks[mp_holistic.HandLandmark.WRIST].visibility < 0.9:
        return "Thumbs Up"
    elif (landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y > landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].y and
          landmarks[mp_holistic.HandLandmark.THUMB_TIP].y > landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y):
        return "Peace Sign"
    elif landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP].visibility > 0.9:
        return "Pointing"
    else:
        visible_fingers = sum(1 for lm in landmarks if lm.visibility > 0.9)
        return f"{visible_fingers} Finger(s)"

# Function to recognize posture
def recognize_posture(pose_landmarks):
    hip_landmarks = [pose_landmarks.landmark[i] for i in [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value]]
    return "Sitting" if all(hip.y > 0.7 for hip in hip_landmarks) else "Standing"

# Video processing function
def generate_frames(cap):
    with mp_pose.Pose() as pose, mp_holistic.Holistic() as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame for processing
            frame = cv2.resize(frame, (640, 480))

            # Face detection and model predictions
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.1, 5)

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (224, 224)) / 255.0
                face_input = np.expand_dims(face_resized, axis=0)

                # Predictions
                age = int(age_model.predict(face_input)[0][0])
                gender = "Male" if gender_model.predict(face_input)[0][0] > 0.5 else "Female"
                emotion_scalar = emotion_model.predict(face_input)[0][0]
                emotion = map_scalar_to_emotion(emotion_scalar)

                # Draw rectangle and text
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f'Age: {age}', (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame, f'Gender: {gender}', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame, f'Emotion: {emotion}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Pose and gesture detection
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(frame_rgb)
            holistic_results = holistic.process(frame_rgb)

            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if holistic_results.right_hand_landmarks:
                gesture = recognize_gesture(holistic_results.right_hand_landmarks.landmark)
                cv2.putText(frame, gesture, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if pose_results.pose_landmarks:
                posture = recognize_posture(pose_results.pose_landmarks)
                cv2.putText(frame, posture, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Encode frame to display it in the browser
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed', methods=['GET', 'POST'])
def video_feed():
    source = request.args.get('source', 'webcam')

    # If source is video and a file is uploaded
    if source == 'video' and request.method == 'POST' and 'video_file' in request.files:
        video_file = request.files['video_file']
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_file.read())
        tfile.close()  # Close the file to ensure it can be accessed properly
        cap = cv2.VideoCapture(tfile.name)
    else:
        cap = cv2.VideoCapture(0)  # Start the webcam feed

    return Response(generate_frames(cap),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)

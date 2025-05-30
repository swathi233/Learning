import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
import mediapipe as mp
import threading
import queue

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

# Function to capture frames from the video source
def capture_frames(cap, frames_queue):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frames_queue.full():
            frames_queue.get()  # Remove the oldest frame if the queue is full
        frames_queue.put(frame)

# Streamlit app
st.title("Real-Time Age, Gender, Emotion, Pose and Gesture Recognition")

# Checkbox for running webcam
run_webcam = st.checkbox("Run Webcam")
FRAME_WINDOW = st.image([])

cap = None

if cap is None and run_webcam:
    cap = cv2.VideoCapture(0)

if cap is not None:
    frames_queue = queue.Queue(maxsize=10)
    thread = threading.Thread(target=capture_frames, args=(cap, frames_queue))
    thread.start()

    while True:
        if not frames_queue.empty():
            frame = frames_queue.get()
            
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
            pose_results = mp_pose.Pose().process(frame_rgb)
            holistic_results = mp_holistic.Holistic().process(frame_rgb)

            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if holistic_results.right_hand_landmarks:
                gesture = recognize_gesture(holistic_results.right_hand_landmarks.landmark)
                cv2.putText(frame, gesture, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if pose_results.pose_landmarks:
                posture = recognize_posture(pose_results.pose_landmarks)
                cv2.putText(frame, posture, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the frame
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
else:
    st.warning("Click 'Run Webcam' to start.")

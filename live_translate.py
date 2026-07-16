import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import joblib
import subprocess
from collections import deque

# 1. Initialization
model = tf.keras.models.load_model('asl_model.h5')
le = joblib.load('label_encoder.pkl')
last_spoken = ""

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
prediction_history = deque(maxlen=10) 

# 2. Start Camera
cap = cv2.VideoCapture(0)
print("Starting live translation... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            data = np.array(landmarks).reshape(1, 63)
            prediction = model.predict(data, verbose=0)
            
            if np.max(prediction) > 0.8:
                predicted_label = le.inverse_transform([np.argmax(prediction)])[0]
                prediction_history.append(predicted_label)
        
        if prediction_history:
            most_common_label = max(set(prediction_history), key=prediction_history.count)
            cv2.putText(frame, f"Sign: {most_common_label}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # This triggers the Windows system voice to speak the letter
            if most_common_label != last_spoken:
                # We use 'powershell' to call the built-in Windows text-to-speech engine
                # It says the letter clearly (e.g., "A", "B", "C")
                cmd = f'powershell -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{most_common_label}\')"'
                subprocess.Popen(cmd, shell=True)
                last_spoken = most_common_label
    else:
        prediction_history.clear()
        last_spoken = "" 
        cv2.putText(frame, "No hand detected", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('ASL Real-Time Translator', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
import os

from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import subprocess
from collections import deque

app = Flask(__name__)

# ----------------------------
# Load Model
# ----------------------------

model = tf.keras.models.load_model("model/hand_gesture_model.keras")
labels = np.load("model/label_encoder.npy", allow_pickle=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

gesture_history = deque(maxlen=10)

# ----------------------------
# Normalization
# ----------------------------

def normalize(points):
    pts = np.array(points).reshape(21,2)
    pts = pts - pts[0]
    m = np.max(np.abs(pts))
    if m != 0:
        pts = pts / m
    return pts.flatten()

# ----------------------------
# Routes
# ----------------------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/gesture")
def gesture():
    return render_template("gesture.html")

@app.route("/control")
def control():
    return render_template("control.html")

@app.route("/sign")
def sign():
    from sign_language import run_sign_language
    run_sign_language()
    return render_template("sign.html")

# ----------------------------
# Camera Stream
# ----------------------------

def generate_frames():

    cap = cv2.VideoCapture(0)

    while True:

        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame,1)
        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        text="No Hand"

        if result.multi_hand_landmarks:

            hand=result.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(frame,hand,mp_hands.HAND_CONNECTIONS)

            points=[]

            for lm in hand.landmark:
                points.append(lm.x)
                points.append(lm.y)

            if len(points)==42:

                prediction=model.predict(
                    np.array([normalize(points)]),
                    verbose=0
                )

                confidence=np.max(prediction)

                if confidence>0.6:

                    gesture=labels[np.argmax(prediction)]

                    text=f"{gesture} ({confidence:.2f})"

                    gesture_history.appendleft(gesture)

        cv2.putText(frame,text,(20,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,255,0),2)

        ret,buffer=cv2.imencode('.jpg',frame)
        frame=buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame')

# ----------------------------
# Gesture Control
# ----------------------------

process=None

@app.route("/start_control")
def start_control():

    global process

    if process is None:
        process = subprocess.Popen(["../venv/Scripts/python.exe", "gesture_control.py"])

    return "Gesture Control Started"

@app.route("/stop_control")
def stop_control():

    global process

    if process:
        process.terminate()
        process=None

    return "Gesture Control Stopped"

# ----------------------------

@app.route("/gesture_history")
def get_gesture_history():
    return jsonify(list(gesture_history))

# ----------------------------

if __name__== "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
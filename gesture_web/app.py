import os
from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import subprocess
from collections import deque

current_prediction = "None"
gesture_history = []
app = Flask(__name__)

# Load Model
model = tf.keras.models.load_model("model/hand_gesture_model.keras")
labels = np.load("model/label_encoder.npy", allow_pickle=True)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6)
mp_draw = mp.solutions.drawing_utils
gesture_history = deque(maxlen=10)

# Normalization
def normalize(points):
    pts = np.array(points).reshape(21,2)
    pts = pts - pts[0]
    m = np.max(np.abs(pts))
    if m != 0:
        pts = pts / m
    return pts.flatten()

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/gesture")
def gesture():
    return render_template("gesture.html")

@app.route("/control")
def control():
    return render_template("control.html")

@app.route("/mouse")
def mouse():
    return render_template("mouse.html")

@app.route("/sign")
def sign():
    return render_template("sign.html")

@app.route("/presentation")
def presentation():
    return render_template("presentation.html")


# Camera Stream
def generate_frames():
    global current_prediction
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
                if confidence > 0.6:
                    gesture = labels[np.argmax(prediction)]
                    text = f"{gesture} ({confidence:.2f})"
                    current_prediction = gesture
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

# Gesture Control
control_process = None

@app.route("/start_control")
def start_control():
    global control_process
    if control_process is None:
        python_path = os.path.abspath(
            "../.venv/Scripts/python.exe"
        )
        script_path = os.path.abspath(
            "gesture_control.py"
        )
        control_process = subprocess.Popen(
            [python_path, script_path]
        )
    return "Gesture Control Started"

@app.route("/stop_control")
def stop_control():
    global control_process
    if control_process:
        control_process.terminate()
        control_process = None
    return "Gesture Control Stopped"



mouse_process = None

@app.route("/start_mouse")
def start_mouse():
    global mouse_process
    if mouse_process is None:
        python_path = os.path.abspath(
            "../.venv/Scripts/python.exe"
        )
        script_path = os.path.abspath(
            "run_virtual_mouse.py"
        )
        mouse_process = subprocess.Popen(
            [python_path, script_path]
        )
    return "Virtual Mouse Started"


@app.route("/stop_mouse")
def stop_mouse():
    global mouse_process
    if mouse_process:
        mouse_process.terminate()
        mouse_process = None
    return "Virtual Mouse Stopped"



# Sign Language
sign_process = None

@app.route("/start_sign")
def start_sign():
    global sign_process
    if sign_process is None:
        python_path = os.path.abspath(
            "../.venv/Scripts/python.exe"
        )
        script_path = os.path.abspath(
            "run_sign.py"
        )
        sign_process = subprocess.Popen(
            [python_path, script_path]
        )
    return "Sign Language Started"

@app.route("/stop_sign")
def stop_sign():
    global sign_process
    if sign_process:
        sign_process.terminate()
        sign_process = None
    return "Sign Language Stopped"


presentation_process = None

@app.route("/start_presentation")
def start_presentation():
    global presentation_process
    if presentation_process is None:
        python_path = os.path.abspath(
            "../.venv/Scripts/python.exe"
        )
        script_path = os.path.abspath(
            "run_presentation.py"
        )
        presentation_process = subprocess.Popen(
            [python_path, script_path]
        )
    return "Presentation Tool Started"

@app.route("/stop_presentation")
def stop_presentation():
    global presentation_process
    if presentation_process:
        presentation_process.terminate()
        presentation_process = None
    return "Presentation Tool Stopped"



@app.route("/gesture_history")
def get_gesture_history():
    return jsonify(list(gesture_history))

@app.route("/current_prediction")
def current_prediction_api():
    return jsonify(
        {
            "prediction":
                current_prediction,
            "count":
                len(gesture_history)
        }
    )


@app.route("/gesture_history")
def history():
    return jsonify(
        gesture_history[-20:]
    )

if __name__== "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

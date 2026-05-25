import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pyautogui
import time
from collections import deque

pyautogui.FAILSAFE = False

print("Gesture Control Started")

model = tf.keras.models.load_model("model/hand_gesture_model.keras")
labels = np.load("model/label_encoder.npy", allow_pickle=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6)

gesture_buffer = deque(maxlen=5)

last_action_time = 0
cooldown = 1


def normalize(points):
    pts = np.array(points).reshape(21, 2)
    pts = pts - pts[0]
    m = np.max(np.abs(pts))
    if m != 0:
        pts = pts / m
    return pts.flatten()


def perform_action(gesture):

    global last_action_time

    if time.time() - last_action_time < cooldown:
        return

    print("Detected:", gesture)

    if gesture == "play":
        pyautogui.press("space")

    elif gesture == "pause":
        pyautogui.press("space")

    elif gesture == "stop":
        pyautogui.press("s")

    elif gesture == "next":
        pyautogui.press("n")

    elif gesture == "previous":
        pyautogui.press("p")

    elif gesture == "volume_up":
        pyautogui.hotkey("ctrl", "up")

    elif gesture == "volume_down":
        pyautogui.hotkey("ctrl", "down")

    last_action_time = time.time()


cam = cv2.VideoCapture(0)

# Create preview window once
cv2.namedWindow("Gesture Control", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Gesture Control", 240,180)
cv2.moveWindow("Gesture Control", 1200,650)
cv2.setWindowProperty("Gesture Control", cv2.WND_PROP_TOPMOST, 1)

print("Gesture Control Running...")

while True:

    ret, frame = cam.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        hand = result.multi_hand_landmarks[0]

        points = []

        for lm in hand.landmark:
            points.append(lm.x)
            points.append(lm.y)

        if len(points) == 42:

            prediction = model.predict(
                np.array([normalize(points)]),
                verbose=0
            )

            confidence = np.max(prediction)

            if confidence > 0.6:

                gesture = labels[np.argmax(prediction)]

                gesture_buffer.append(gesture)

                stable_gesture = max(set(gesture_buffer), key=gesture_buffer.count)

                # Show detected gesture on camera
                cv2.putText(frame, stable_gesture, (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

                perform_action(stable_gesture)

    # ---------------- MINI CAMERA PREVIEW ----------------

    mini_frame = cv2.resize(frame, (220, 160))

    cv2.imshow("Gesture Control", mini_frame)

    # Move the window to bottom-right (adjust if needed)
    cv2.moveWindow("Gesture Control", 1200, 650)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

    # -------- MINI CAMERA PREVIEW --------

    mini_frame = cv2.resize(frame, (240,180))

    cv2.imshow("Gesture Control", mini_frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

time.sleep(0.01)

cam.release()
cv2.destroyAllWindows()
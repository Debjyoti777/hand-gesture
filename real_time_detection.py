import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Load trained model and labels
model = tf.keras.models.load_model("hand_gesture_model.keras")
labels = np.load("label_encoder.npy", allow_pickle=True)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

# Normalization function (same as training)
def normalize(points):
    pts = np.array(points).reshape(21, 2)
    pts = pts - pts[0]  # wrist as origin
    m = np.max(np.abs(pts))
    if m != 0:
        pts = pts / m
    return pts.flatten()

# Start camera
cam = cv2.VideoCapture(0)

print("🎥 Real-time Gesture Detection Running... Press 'q' to quit")

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    text = "..."

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
                text = labels[np.argmax(prediction)]

        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

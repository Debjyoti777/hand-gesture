import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Load trained dynamic model
model = tf.keras.models.load_model(
    "gesture_web/model/dynamic_sign_model.h5"
)

labels = np.load(
    "gesture_web/model/dynamic_sign_labels.npy",
    allow_pickle=True
)

# MediaPipe setup
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils


def normalize(points):

    pts = np.array(points).reshape(21, 2)

    pts = pts - pts[0]

    m = np.max(np.abs(pts))

    if m != 0:
        pts = pts / m

    return pts.flatten()


def run_sign_language():

    cap = cv2.VideoCapture(0)

    sequence = []

    text = "Detecting..."

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        text = "No Hand"

        if result.multi_hand_landmarks:

            frame_features = []

            # process BOTH hands
            for hand in result.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS
                )

                points = []

                for lm in hand.landmark:

                    points.append(lm.x)
                    points.append(lm.y)

                if len(points) == 42:

                    frame_features.extend(
                        normalize(points)
                    )

            # BOTH hands required
            if len(frame_features) == 84:

                sequence.append(frame_features)

                # keep last 20 frames
                if len(sequence) > 20:
                    sequence.pop(0)

                # predict when sequence ready
                if len(sequence) == 20:

                    prediction = model.predict(
                        np.array([sequence]),
                        verbose=0
                    )

                    confidence = np.max(prediction)

                    if confidence > 0.5:

                        gesture = labels[
                            np.argmax(prediction)
                        ]

                        text = gesture

        cv2.putText(
            frame,
            text,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Sign Language", frame)

        # ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()

    cv2.destroyAllWindows()
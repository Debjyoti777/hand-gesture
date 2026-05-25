import os
import cv2
import numpy as np
import mediapipe as mp

dataset_folder = "../filtered_wlasl"
sequence_length = 20

mp_hands = mp.solutions.hands

# ✅ allow TWO hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

data = []
labels = []


def normalize(points):

    pts = np.array(points).reshape(21, 2)

    pts = pts - pts[0]

    m = np.max(np.abs(pts))

    if m != 0:
        pts = pts / m

    return pts.flatten()


for label in os.listdir(dataset_folder):

    label_path = os.path.join(dataset_folder, label)

    if not os.path.isdir(label_path):
        continue

    print("Processing:", label)

    for video in os.listdir(label_path):

        video_path = os.path.join(label_path, video)

        cap = cv2.VideoCapture(video_path)

        sequence = []

        while len(sequence) < sequence_length:

            ret, frame = cap.read()

            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = hands.process(rgb)

            # ✅ multi-hand processing
            if result.multi_hand_landmarks:

                frame_features = []

                for hand in result.multi_hand_landmarks:

                    points = []

                    for lm in hand.landmark:

                        points.append(lm.x)
                        points.append(lm.y)

                    if len(points) == 42:

                        normalized = normalize(points)

                        frame_features.extend(normalized)

                # ✅ ensure BOTH hands exist
                if len(frame_features) == 84:

                    sequence.append(frame_features)

        cap.release()

        # ✅ only save complete sequences
        if len(sequence) == sequence_length:

            data.append(sequence)
            labels.append(label)

hands.close()

np.save("../dynamic_sign_data.npy", {
    "data": data,
    "labels": labels
})

print("Dynamic dataset saved!")
import os
import cv2
import numpy as np
import mediapipe as mp

dataset_folder = "../sign_frames"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

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

    for img_name in os.listdir(label_path):

        img_path = os.path.join(label_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:

            hand = result.multi_hand_landmarks[0]

            points = []

            for lm in hand.landmark:
                points.append(lm.x)
                points.append(lm.y)

            if len(points) == 42:
                data.append(normalize(points))
                labels.append(label)

hands.close()

np.save("../sign_landmarks_data.npy", {
    "data": data,
    "labels": labels
})

print("Sign landmark data saved!")
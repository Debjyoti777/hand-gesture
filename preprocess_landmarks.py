import os
import cv2
import numpy as np
import mediapipe as mp

dataset_folder = "dataset"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

all_landmarks = []
all_labels = []

print("Starting landmark preprocessing...")


def normalize_landmarks(landmark_list):
    points = np.array(landmark_list)
    points = points.reshape(21, 2)

    wrist = points[0]
    points = points - wrist

    max_value = np.max(np.abs(points))

    if max_value != 0:
        points = points / max_value

    return points.flatten().tolist()


for gesture_name in os.listdir(dataset_folder):
    gesture_path = os.path.join(dataset_folder, gesture_name)

    if not os.path.isdir(gesture_path):
        continue

    print("Processing gesture:", gesture_name)

    for image_name in os.listdir(gesture_path):
        image_path = os.path.join(gesture_path, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_image)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]

            temp_landmarks = []

            for lm in hand.landmark:
                temp_landmarks.append(lm.x)
                temp_landmarks.append(lm.y)

            normalized = normalize_landmarks(temp_landmarks)

            all_landmarks.append(normalized)
            all_labels.append(gesture_name)

np.save("landmarks_data.npy", {
    "data": all_landmarks,
    "labels": all_labels
})
print("Finished! Landmark file saved.")
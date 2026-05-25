import cv2
import os

# input folder (sorted videos)
video_root = "../sorted_wlasl"

# output folder (frames dataset)
frame_root = "../sign_frames"

os.makedirs(frame_root, exist_ok=True)

print("Starting frame extraction...")

for word in os.listdir(video_root):

    word_folder = os.path.join(video_root, word)

    if not os.path.isdir(word_folder):
        continue

    print("Processing word:", word)

    output_word_folder = os.path.join(frame_root, word)
    os.makedirs(output_word_folder, exist_ok=True)

    for video in os.listdir(word_folder):

        video_path = os.path.join(word_folder, video)

        cap = cv2.VideoCapture(video_path)

        frame_count = 0
        saved_count = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            # save every 5th frame
            if frame_count % 5 == 0:

                filename = f"{video}_{saved_count}.jpg"

                save_path = os.path.join(output_word_folder, filename)

                cv2.imwrite(save_path, frame)

                saved_count += 1

            frame_count += 1

        cap.release()

print("Frame extraction complete!")
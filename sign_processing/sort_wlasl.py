import json
import os
import shutil

json_path = "../wlasl_dataset/WLASL_v0.3.json"
video_folder = "../wlasl_dataset/videos"
output_folder = "../sorted_wlasl"

os.makedirs(output_folder, exist_ok=True)

with open(json_path, "r") as f:
    data = json.load(f)

for entry in data:

    word = entry["gloss"].lower()

    for instance in entry["instances"]:

        video_id = instance["video_id"]
        video_name = video_id + ".mp4"

        source = os.path.join(video_folder, video_name)

        if os.path.exists(source):

            word_folder = os.path.join(output_folder, word)
            os.makedirs(word_folder, exist_ok=True)

            destination = os.path.join(word_folder, video_name)

            shutil.copy(source, destination)

print("Dataset organized successfully")
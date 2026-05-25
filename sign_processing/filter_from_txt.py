import os
import shutil

source = "../sorted_wlasl"
target = "../filtered_wlasl"

# read words from txt file
with open("words.txt", "r") as f:
    words = [line.strip().lower() for line in f.readlines()]

os.makedirs(target, exist_ok=True)

for word in words:

    src_path = os.path.join(source, word)
    dst_path = os.path.join(target, word)

    if os.path.exists(src_path):
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        print(f"Copied: {word}")
    else:
        print(f"Not found: {word}")

print("Done filtering!")
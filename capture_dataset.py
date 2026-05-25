import cv2
import os
import time

print("Program started")

gesture_name = "hello"
main_folder = "dataset"
full_path = os.path.join(main_folder, gesture_name)

os.makedirs(full_path, exist_ok=True)

print("Saving images to:", full_path)

cam = cv2.VideoCapture(0)

time.sleep(2)  # let camera warm up

if not cam.isOpened():
    print("Camera could not be opened")
    input("Press Enter to exit")
    exit()

print("Camera opened successfully")

picture_counter = 0

print("Press 's' to SAVE a picture.")
print("Press 'q' to QUIT.")

while True:
    ret, frame = cam.read()

    if not ret:
        print("Frame not received from camera")
        break

    frame = cv2.flip(frame, 1)

    text = f"{gesture_name} | Pic #{picture_counter}"
    cv2.putText(frame, text, (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow("Gesture Recording Window", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        filename = os.path.join(full_path, f"{picture_counter}.jpg")
        cv2.imwrite(filename, frame)
        print("Saved:", filename)
        picture_counter += 1

    elif key == ord('q'):
        print("Exiting program")
        break

cam.release()
cv2.destroyAllWindows()
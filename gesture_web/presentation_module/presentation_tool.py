import cv2
import mediapipe as mp
import numpy as np

# -------------------------
# Mediapipe Setup
# -------------------------

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# -------------------------
# Variables
# -------------------------

canvas = None
prev_x, prev_y = 0, 0

current_color = (255, 0, 255)   # Purple default
menu_open = False

pinch_threshold = 35

# -------------------------
# Camera
# -------------------------

cap = cv2.VideoCapture(0)

cv2.namedWindow(
    "Virtual Air Canvas",
    cv2.WND_PROP_FULLSCREEN
)

cv2.setWindowProperty(
    "Virtual Air Canvas",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

# -------------------------
# Main Loop
# -------------------------

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    result = hands.process(rgb)

    # Prevent NameError when no hand exists
    index_up = False
    middle_up = False
    pinch_gesture = False

    if result.multi_hand_landmarks:

        handLms = result.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame,
            handLms,
            mp_hands.HAND_CONNECTIONS
        )

        # -------------------------
        # Landmark Positions
        # -------------------------

        index_x = int(
            handLms.landmark[8].x *
            frame.shape[1]
        )

        index_y = int(
            handLms.landmark[8].y *
            frame.shape[0]
        )

        thumb_x = int(
            handLms.landmark[4].x *
            frame.shape[1]
        )

        thumb_y = int(
            handLms.landmark[4].y *
            frame.shape[0]
        )

        index_up = (
            handLms.landmark[8].y <
            handLms.landmark[6].y
        )

        middle_up = (
            handLms.landmark[12].y <
            handLms.landmark[10].y
        )

        ring_folded = (
            handLms.landmark[16].y >
            handLms.landmark[14].y
        )

        pinky_folded = (
            handLms.landmark[20].y >
            handLms.landmark[18].y
        )

        # -------------------------
        # Pinch Gesture
        # -------------------------

        pinch_distance = np.sqrt(
            (index_x - thumb_x) ** 2 +
            (index_y - thumb_y) ** 2
        )

        pinch_gesture = (
            pinch_distance < pinch_threshold and
            ring_folded and
            pinky_folded
        )

        if pinch_gesture:
            menu_open = True

        # -------------------------
        # Color Menu
        # -------------------------

        if menu_open:

            colors = [
                ((255, 0, 0), 50),     # Blue
                ((0, 255, 0), 170),    # Green
                ((0, 0, 255), 290),    # Red
                ((0, 0, 0), 410)       # Black
            ]

            for color, xpos in colors:

                cv2.rectangle(
                    frame,
                    (xpos, 20),
                    (xpos + 90, 100),
                    color,
                    -1
                )

            # Eraser button
            cv2.rectangle(
                frame,
                (530, 20),
                (670, 100),
                (255,255,255),
                -1
            )

            cv2.putText(
                frame,
                "ERASE",
                (550,70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,0,255),
                2
            )

            # Select color using pinch
            if pinch_gesture and 20 < index_y < 100:

                if 50 < index_x < 140:
                    current_color = (255,0,0)
                    menu_open = False

                elif 170 < index_x < 260:
                    current_color = (0,255,0)
                    menu_open = False

                elif 290 < index_x < 380:
                    current_color = (0,0,255)
                    menu_open = False

                elif 410 < index_x < 500:
                    current_color = (0,0,0)
                    menu_open = False

                elif 530 < index_x < 670:
                    current_color = "eraser"
                    menu_open = False

        # -------------------------
        # Pointer Mode
        # -------------------------

        elif index_up and middle_up:

            cv2.circle(
                frame,
                (index_x, index_y),
                12,
                (0,255,255),
                -1
            )

            prev_x, prev_y = 0, 0

        # -------------------------
        # Draw Mode
        # -------------------------

        elif index_up:

            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = index_x, index_y

            thickness = (
                40
                if current_color == "eraser"
                else 8
            )

            color = (
                (0,0,0)
                if current_color == "eraser"
                else current_color
            )

            cv2.line(
                canvas,
                (prev_x, prev_y),
                (index_x, index_y),
                color,
                thickness
            )

            cv2.circle(
                frame,
                (index_x, index_y),
                10,
                color,
                -1
            )

            prev_x, prev_y = index_x, index_y

        else:
            prev_x, prev_y = 0, 0

    # -------------------------
    # Overlay Canvas
    # -------------------------

    frame = cv2.add(frame, canvas)

    cv2.putText(
        frame,
        "Index Finger = Draw",
        (20, frame.shape[0]-90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        "Pinch Thumb+Index = Open Color Menu",
        (20, frame.shape[0]-55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        "Press Q to Quit",
        (20, frame.shape[0]-20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2
    )

    cv2.imshow(
        "Virtual Air Canvas",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

# -------------------------
# Cleanup
# -------------------------

cap.release()
cv2.destroyAllWindows()
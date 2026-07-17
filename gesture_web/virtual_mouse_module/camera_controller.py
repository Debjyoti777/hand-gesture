import cv2
import time

from virtual_mouse_module.hand_tracker import HandTracker
from virtual_mouse_module.frame_processor import FrameProcessor
from virtual_mouse_module.gesture_detector import GestureDetector
from virtual_mouse_module.virtual_mouse import VirtualMouse
from virtual_mouse_module.frame_saver import FrameSaver
from virtual_mouse_module.video_recorder import VideoRecorder


class CameraController:

    def __init__(self):
        # Create Objects
        self.processor = FrameProcessor()
        self.tracker = HandTracker("model/hand_landmarker.task")
        self.detector = GestureDetector()
        self.mouse = VirtualMouse()
        self.saver = FrameSaver()

        # Webcam
        self.cap = cv2.VideoCapture(0)
        self.width = int(self.cap.get(3))
        self.height = int(self.cap.get(4))

        # Video Recorder
        self.recorder = VideoRecorder(
            "saved_videos/output1.mp4",
            self.width,
            self.height
        )

        # Variables
        self.dragging = False
        self.last_click_time = 0
        self.click_delay = 0.5
        self.last_scroll_time = 0
        self.scroll_delay = 0.3
        self.prev_time = time.time()

    def run(self):
        while True:
            success, frame = self.cap.read()
            if not success:
                break

            # 1. Resize and Flip FIRST
            frame = self.processor.resize_frame(frame)
            frame = self.processor.flip_frame(frame)

            # 2. DETECT BEFORE BLURRING (MediaPipe needs sharp edges for accuracy)
            result = self.tracker.detect(frame)

            # 3. Apply blur purely for visual output (Optional)
            frame = self.processor.blur_frame(frame)

            self.recorder.write_frame(frame)
            self.saver.save_frame(frame)

            h, w, _ = frame.shape
            current_action = "Tracking..."

            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]

                # Draw All Landmarks
                for landmark in hand_landmarks:
                    lx = int(landmark.x * w)
                    ly = int(landmark.y * h)
                    cv2.circle(frame, (lx, ly), 4, (0, 255, 0), -1)

                # Cursor Movement
                index_tip = hand_landmarks[8]
                screen_x = int(index_tip.x * self.mouse.screen_width)
                screen_y = int(index_tip.y * self.mouse.screen_height)
                self.mouse.move_cursor(screen_x, screen_y)

                # Highlight index finger
                x = int(index_tip.x * w)
                y = int(index_tip.y * h)
                cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)

                current_time = time.time()

                # ====================================
                # GESTURE LOGIC
                # ====================================
                if self.detector.detect_click(hand_landmarks):
                    if current_time - self.last_click_time > self.click_delay:
                        self.mouse.left_click()
                        current_action = "LEFT CLICK"
                        self.last_click_time = current_time

                elif self.detector.detect_right_click(hand_landmarks):
                    if current_time - self.last_click_time > self.click_delay:
                        self.mouse.right_click()
                        current_action = "RIGHT CLICK"
                        self.last_click_time = current_time

                elif self.detector.detect_drag(hand_landmarks):
                    if not self.dragging:
                        self.mouse.drag_start()
                        self.dragging = True
                    current_action = "DRAGGING"

                elif self.detector.detect_scroll(hand_landmarks):
                    if current_time - self.last_scroll_time > self.scroll_delay:
                        if y < h // 2:
                            self.mouse.scroll_up()
                            current_action = "SCROLL UP"
                        else:
                            self.mouse.scroll_down()
                            current_action = "SCROLL DOWN"
                        self.last_scroll_time = current_time

                else:
                    if self.dragging:
                        self.mouse.drag_stop()
                        self.dragging = False

            # ====================================
            # UI OVERLAY & FPS
            # ====================================
            current_frame_time = time.time()
            fps = 1 / (current_frame_time - self.prev_time)
            self.prev_time = current_frame_time

            # Draw a dark background rectangle for text readability
            cv2.rectangle(frame, (10, 10), (350, 90), (0, 0, 0), -1)

            cv2.putText(frame, f"FPS: {int(fps)}", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Action: {current_action}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Display
            cv2.imshow("Virtual Mouse", frame)

            # ESC Key Exit
            if cv2.waitKey(1) & 0xFF == 27:
                break

        # Cleanup
        self.cap.release()
        self.recorder.release()
        cv2.destroyAllWindows()
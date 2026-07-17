import cv2
from pathlib import Path


class FrameSaver:

    def __init__(
            self,
            save_interval=10):

        self.frame_count = 0

        self.save_interval = save_interval

        # Create path to captured_frames folder
        self.output_folder = Path("captured_frames")

        # Create folder if it doesn't exist
        self.output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

    def save_frame(self, frame):

        self.frame_count += 1

        # Save every nth frame
        if self.frame_count % self.save_interval == 0:

            filename = (
                self.output_folder /
                f"frame_{self.frame_count}.jpg"
            )

            cv2.imwrite(
                str(filename),
                frame
            )
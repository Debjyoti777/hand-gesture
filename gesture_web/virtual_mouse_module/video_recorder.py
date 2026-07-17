import cv2


class VideoRecorder:

    def __init__(
            self,
            filename,
            width,
            height):

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        self.writer = cv2.VideoWriter(
            filename,
            fourcc,
            20,
            (width, height)
        )

        print("VideoWriter opened:", self.writer.isOpened())

    def write_frame(
            self,
            frame):

        self.writer.write(frame)

    def release(self):

        self.writer.release()
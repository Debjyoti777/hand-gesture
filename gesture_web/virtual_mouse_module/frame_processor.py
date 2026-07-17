import cv2


class FrameProcessor:

    def resize_frame(self, frame,
                     width=640,
                     height=480):

        return cv2.resize(
            frame,
            (width, height)
        )

    def flip_frame(self, frame):

        return cv2.flip(
            frame,
            1
        )

    def blur_frame(self, frame):

        return cv2.GaussianBlur(
            frame,
            (5, 5),
            0
        )

    def convert_rgb(self, frame):

        return cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )
# @PURPOSE:
# @AUTHOR: Aoxiang Zhang
# @TIME: 06:37 2025/4/30 UTC+8
import cv2 as cv

GREEN = (0, 0, 255)
RED = (0, 0, 225)
BLUE = (225, 0, 0)


class ImgProcessor:
    def __init__(self, img):
        self.img = img

    def draw_keypoint(self, x, y, radius=2, color=BLUE, thickness=-1):
        cv.circle(self.img, (int(x), int(y)), radius, color, thickness)

    def draw_line(self, x1, y1, x2, y2, color=BLUE, thickness=1):
        cv.line(self.img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

    def draw_box(self, x_left, x_right, y_top, y_bottom, color=BLUE, thickness=1):
        cv.rectangle(self.img, (int(x_left), int(y_top)), (int(x_right), int(y_bottom)), color, thickness)

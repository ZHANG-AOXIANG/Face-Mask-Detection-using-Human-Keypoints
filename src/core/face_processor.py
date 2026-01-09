import cv2 as cv
import numpy as np
import math


class FaceProcessor:
    def __init__(self):
        pass

    def process(self, image, face_keypoints):
        """
        根据坐标点将人脸旋转至双眼水平，并修复过度旋转 Bug。
        face_keypoints: [[lx, ly], [rx, ry], [nx, ny], [rex, rey]]
        """
        left_eye = np.array(face_keypoints[0])
        right_eye = np.array(face_keypoints[1])

        # 1. 计算原始旋转角度
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]

        # atan2 返回弧度 (-pi, pi)
        angle_rad = math.atan2(dy, dx)
        angle = math.degrees(angle_rad)

        # 2. 核心修复：防止过度旋转 (Angle Normalization)
        # 人脸对齐的目标是让双眼水平，如果角度超过 90 度，说明发生了镜像或检测点翻转
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180

        # 3. 图像填充 (保持你代码中的安全边距逻辑)
        padded_img, new_left_eye, _, _ = self._make_square(image, left_eye)

        # 4. 执行旋转 (使用修复后的角度)
        h, w = padded_img.shape[:2]
        # 注意：OpenCV 的旋转方向是逆时针为正，需根据坐标系核对
        M = cv.getRotationMatrix2D(new_left_eye, angle, 1.0)
        rotated_img = cv.warpAffine(padded_img, M, (w, h))

        # 5. 动态裁剪 (基于眼睛间距)
        eye_dist = math.sqrt(dx ** 2 + dy ** 2)
        y_top, y_bottom, x_left, x_right = self._get_cut_point(new_left_eye, eye_dist)

        return self._cut_image(rotated_img, y_top, y_bottom, x_left, x_right)

    def _make_square(self, image, anchor_point):
        """实现安全的边界填充，防止旋转切角"""
        h, w = image.shape[:2]
        x, y = int(anchor_point[0]), int(anchor_point[1])
        max_side = max(x, y, w - x, h - y)
        pad = int(max_side * 1.5)

        x_left, x_right = pad - x, pad - (w - x)
        y_top, y_bottom = pad - y, pad - (h - y)

        square_image = cv.copyMakeBorder(image, y_top, y_bottom, x_left, x_right,
                                         cv.BORDER_CONSTANT, value=[0, 0, 0])
        return square_image, (pad, pad), x_left, y_top

    def _get_cut_point(self, point, distance):
        """定义裁剪区域坐标"""
        y_top = point[1] - distance * 1.5
        y_bottom = point[1] + distance * 1.5
        x_left = point[0] - distance * 1.5
        x_right = point[0] + distance * 1.5
        return y_top, y_bottom, x_left, x_right

    def _cut_image(self, image, y_top, y_bottom, x_left, x_right):
        """执行裁剪并防止溢出"""
        y1, y2 = max(0, int(y_top)), min(image.shape[0], int(y_bottom))
        x1, x2 = max(0, int(x_left)), min(image.shape[1], int(x_right))
        return image[y1:y2, x1:x2]
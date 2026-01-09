from src.utils.parser import KeypointsParser
from src.utils.img_tool import ImgProcessor


GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)


class KeypointsDrawer(ImgProcessor, KeypointsParser):

    def __init__(self, img, keypoints_data):
        ImgProcessor.__init__(self, img)
        KeypointsParser.__init__(self, keypoints_data)

    def draw_nose(self, person_score_target=0.6, keypoint_score_target=0.65):
        for id in range(self.get_total_person()):
            if (self.get_person_score(id) > person_score_target
                    and self.get_nose_score(id) > keypoint_score_target):
                self.draw_keypoint(self.get_nose_x(id),
                                   self.get_nose_y(id), color=RED)

    def draw_right_eye(self, person_score_target=0.6, keypoint_score_target=0.65):
        for id in range(self.get_total_person()):
            if (self.get_person_score(id) > person_score_target
                    and self.get_right_eye_score(id) > keypoint_score_target):
                self.draw_keypoint(self.get_right_eye_x(id),
                                   self.get_right_eye_y(id), color=BLUE)

    def draw_left_eye(self, person_score_target=0.6, keypoint_score_target=0.65):
        for id in range(self.get_total_person()):
            if (self.get_person_score(id) > person_score_target
                    and self.get_left_eye_score(id) > keypoint_score_target):
                self.draw_keypoint(self.get_left_eye_x(id),
                                   self.get_left_eye_y(id), color=GREEN)

    def draw_right_ear(self, person_score_target=0.6, keypoint_score_target=0.65):
        for id in range(self.get_total_person()):
            if (self.get_person_score(id) > person_score_target
                    and self.get_right_ear_score(id) > keypoint_score_target):
                self.draw_keypoint(self.get_right_ear_x(id),
                                   self.get_right_ear_y(id))

    def draw_left_ear(self, person_score_target=0.6, keypoint_score_target=0.65):
        for id in range(self.get_total_person()):
            if (self.get_person_score(id) > person_score_target
                    and self.get_left_ear_score(id) > keypoint_score_target):
                self.draw_keypoint(self.get_left_ear_x(id),
                                   self.get_left_ear_y(id))

    def draw_right_shoulder(self, person_score_target=0.6, keypoint_score_target=0.65):
        for id in range(self.get_total_person()):
            if (self.get_person_score(id) > person_score_target
                    and self.get_right_shoulder_score(id) > keypoint_score_target):
                self.draw_keypoint(self.get_right_shoulder_x(id),
                                   self.get_right_shoulder_y(id))

    def draw_left_shoulder(self, person_score_target=0.6, keypoint_score_target=0.65):
        for id in range(self.get_total_person()):
            if (self.get_person_score(id) > person_score_target
                    and self.get_left_shoulder_score(id) > keypoint_score_target):
                self.draw_keypoint(self.get_left_shoulder_x(id),
                                   self.get_left_shoulder_y(id))

    def draw_right_elbow(self, person_score_target=0.6, keypoint_score_target=0.65):
        for id in range(self.get_total_person()):
            if (self.get_person_score(id) > person_score_target
                    and self.get_right_elbow_score(id) > keypoint_score_target):
                self.draw_keypoint(self.get_right_elbow_x(id),
                                   self.get_right_elbow_y(id))

    def draw_left_elbow(self, person_score_target=0.6, keypoint_score_target=0.65):
        for id in range(self.get_total_person()):
            if (self.get_person_score(id) > person_score_target
                    and self.get_left_elbow_score(id) > keypoint_score_target):
                self.draw_keypoint(self.get_left_elbow_x(id),
                                   self.get_left_elbow_y(id))

    def draw_right_wrist(self, person_score_target=0.6, keypoint_score_target=0.65):
        for id in range(self.get_total_person()):
            if (self.get_person_score(id) > person_score_target
                    and self.get_right_wrist_score(id) > keypoint_score_target):
                self.draw_keypoint(self.get_right_wrist_x(id),
                                   self.get_right_wrist_y(id))

    def draw_left_wrist(self, person_score_target=0.6, keypoint_score_target=0.65):
        for id in range(self.get_total_person()):
            if (self.get_person_score(id) > person_score_target
                    and self.get_left_wrist_score(id) > keypoint_score_target):
                self.draw_keypoint(self.get_left_wrist_x(id),
                                   self.get_left_wrist_y(id))

    def draw_left_hip(self, person_score_target=0.6, keypoint_score_target=0.65):
        for id in range(self.get_total_person()):
            if (self.get_person_score(id) > person_score_target
                    and self.get_left_hip_score(id) > keypoint_score_target):
                self.draw_keypoint(self.get_left_hip_x(id),
                                   self.get_left_hip_y(id))

    def draw_right_hip(self, person_score_target=0.6, keypoint_score_target=0.65):
        for id in range(self.get_total_person()):
            if (self.get_person_score(id) > person_score_target
                    and self.get_right_hip_score(id) > keypoint_score_target):
                self.draw_keypoint(self.get_right_hip_x(id),
                                   self.get_right_hip_y(id))

    def draw_left_knee(self, person_score_target=0.6, keypoint_score_target=0.65):
        for id in range(self.get_total_person()):
            if (self.get_person_score(id) > person_score_target
                    and self.get_left_knee_score(id) > keypoint_score_target):
                self.draw_keypoint(self.get_left_knee_x(id),
                                   self.get_left_knee_y(id))

    def draw_right_knee(self, person_score_target=0.6, keypoint_score_target=0.65):
        for id in range(self.get_total_person()):
            if (self.get_person_score(id) > person_score_target
                    and self.get_right_knee_score(id) > keypoint_score_target):
                self.draw_keypoint(self.get_right_knee_x(id),
                                   self.get_right_knee_y(id))

    def draw_left_ankle(self, person_score_target=0.6, keypoint_score_target=0.65):
        for id in range(self.get_total_person()):
            if (self.get_person_score(id) > person_score_target
                    and self.get_left_ankle_score(id) > keypoint_score_target):
                self.draw_keypoint(self.get_left_ankle_x(id),
                                   self.get_left_ankle_y(id))

    def draw_right_ankle(self, person_score_target=0.6, keypoint_score_target=0.65):
        for id in range(self.get_total_person()):
            if (self.get_person_score(id) > person_score_target
                    and self.get_right_ankle_score(id) > keypoint_score_target):
                self.draw_keypoint(self.get_right_ankle_x(id),
                                   self.get_right_ankle_y(id))

    def connect_left_nose_eye(self, person_score_target=0.6, keypoint_score_target=0.65):
        for id in range(self.get_total_person()):
            if (self.get_person_score(id) > person_score_target
                    and self.get_nose_score(id) > keypoint_score_target
                    and self.get_left_eye_score(id) > keypoint_score_target):
                self.draw_line(self.get_nose_x(id),
                               self.get_nose_y(id),
                               self.get_left_eye_x(id),
                               self.get_left_eye_y(id), color=(0, 225, 0))

    def connect_right_nose_eye(self, person_score_target=0.6, keypoint_score_target=0.65):
        for id in range(self.get_total_person()):
            if (self.get_person_score(id) > person_score_target
                    and self.get_nose_score(id) > keypoint_score_target
                    and self.get_right_eye_score(id) > keypoint_score_target):
                self.draw_line(self.get_nose_x(id),
                               self.get_nose_y(id),
                               self.get_right_eye_x(id),
                               self.get_right_eye_y(id), color=GREEN)

    def connect_right_eye_ear(self, person_score_target=0.6, keypoint_score_target=0.65):
        for id in range(self.get_total_person()):
            if (self.get_person_score(id) > person_score_target
                    and self.get_right_eye_score(id) > keypoint_score_target
                    and self.get_right_ear_score(id) > keypoint_score_target):
                self.draw_line(self.get_right_eye_x(id),
                               self.get_right_eye_y(id),
                               self.get_right_ear_x(id),
                               self.get_right_ear_y(id), color=GREEN)

    def connect_left_eye_ear(self, person_score_target=0.6, keypoint_score_target=0.65):
        for id in range(self.get_total_person()):
            if (self.get_person_score(id) > person_score_target
                    and self.get_left_eye_score(id) > keypoint_score_target
                    and self.get_left_ear_score(id) > keypoint_score_target):
                self.draw_line(self.get_left_eye_x(id),
                               self.get_left_eye_y(id),
                               self.get_left_ear_x(id),
                               self.get_left_ear_y(id), color=GREEN)

    def connect_right_shoulder_elbow(self, person_score_target=0.6, keypoint_score_target=0.65):
        for id in range(self.get_total_person()):
            if (self.get_person_score(id) > person_score_target
                    and self.get_right_shoulder_score(id) > keypoint_score_target
                    and self.get_right_elbow_score(id) > keypoint_score_target):
                self.draw_line(self.get_right_shoulder_x(id),
                               self.get_right_shoulder_y(id),
                               self.get_right_elbow_x(id),
                               self.get_right_elbow_y(id), color=GREEN)

    def connect_left_shoulder_elbow(self, person_score_target=0.6, keypoint_score_target=0.65):
        for id in range(self.get_total_person()):
            if (self.get_person_score(id) > person_score_target
                    and self.get_left_shoulder_score(id) > keypoint_score_target
                    and self.get_left_elbow_score(id) > keypoint_score_target):
                self.draw_line(self.get_left_shoulder_x(id),
                               self.get_left_shoulder_y(id),
                               self.get_left_elbow_x(id),
                               self.get_left_elbow_y(id), color=GREEN)

    def connect_right_elbow_wrist(self, person_score_target=0.6, keypoint_score_target=0.65):
        for id in range(self.get_total_person()):
            if (self.get_person_score(id) > person_score_target
                    and self.get_right_elbow_score(id) > keypoint_score_target
                    and self.get_right_wrist_score(id) > keypoint_score_target):
                self.draw_line(self.get_right_elbow_x(id),
                               self.get_right_elbow_y(id),
                               self.get_right_wrist_x(id),
                               self.get_right_wrist_y(id), color=GREEN)

    def connect_left_elbow_wrist(self, person_score_target=0.6, keypoint_score_target=0.65):
        for id in range(self.get_total_person()):
            if (self.get_person_score(id) > person_score_target
                    and self.get_left_wrist_score(id) > keypoint_score_target
                    and self.get_left_elbow_score(id) > keypoint_score_target):
                self.draw_line(self.get_left_elbow_x(id),
                               self.get_left_elbow_y(id),
                               self.get_left_wrist_x(id),
                               self.get_left_wrist_y(id), color=GREEN)

    def connect_right_hip_knee(self, person_score_target=0.6, keypoint_score_target=0.65):
        for id in range(self.get_total_person()):
            if (self.get_person_score(id) > person_score_target
                    and self.get_right_hip_score(id) > keypoint_score_target
                    and self.get_right_knee_score(id) > keypoint_score_target):
                self.draw_line(self.get_right_hip_x(id),
                               self.get_right_hip_y(id),
                               self.get_right_knee_x(id),
                               self.get_right_knee_y(id), color=GREEN)

    def connect_left_hip_knee(self, person_score_target=0.6, keypoint_score_target=0.65):
        for id in range(self.get_total_person()):
            if (self.get_person_score(id) > person_score_target
                    and self.get_left_hip_score(id) > keypoint_score_target
                    and self.get_left_knee_score(id) > keypoint_score_target):
                self.draw_line(self.get_left_hip_x(id),
                               self.get_left_hip_y(id),
                               self.get_left_knee_x(id),
                               self.get_left_knee_y(id), color=GREEN)

    def connect_right_knee_ankle(self, person_score_target=0.6, keypoint_score_target=0.65):
        for id in range(self.get_total_person()):
            if (self.get_person_score(id) > person_score_target
                    and self.get_right_knee_score(id) > keypoint_score_target
                    and self.get_right_ankle_score(id) > keypoint_score_target):
                self.draw_line(self.get_right_knee_x(id),
                               self.get_right_knee_y(id),
                               self.get_right_ankle_x(id),
                               self.get_right_ankle_y(id), color=GREEN)

    def connect_left_knee_ankle(self, person_score_target=0.6, keypoint_score_target=0.65):
        for id in range(self.get_total_person()):
            if (self.get_person_score(id) > person_score_target
                    and self.get_left_knee_score(id) > keypoint_score_target
                    and self.get_left_ankle_score(id) > keypoint_score_target):
                self.draw_line(self.get_left_knee_x(id),
                               self.get_left_knee_y(id),
                               self.get_left_ankle_x(id),
                               self.get_left_ankle_y(id), color=GREEN)

    def draw_person_post(self, person_score_target=0.6, keypoint_score_target=0.65):
        SKELETON_CONNECTIONS = [
            ('nose', 'left_eye', (255, 0, 0)),
            ('nose', 'right_eye', (255, 0, 0)),
            ('left_eye', 'left_ear', (255, 0, 0)),
            ('right_eye', 'right_ear', (255, 0, 0)),
            ('left_shoulder', 'left_elbow', (0, 255, 0)),
            ('left_elbow', 'left_wrist', (0, 255, 0)),
            ('right_shoulder', 'right_elbow', (0, 255, 0)),
            ('right_elbow', 'right_wrist', (0, 255, 0)),
            ('left_shoulder', 'right_shoulder', (0, 255, 255)),
            ('left_shoulder', 'left_hip', (0, 255, 255)),
            ('right_shoulder', 'right_hip', (0, 255, 255)),
            ('left_hip', 'right_hip', (0, 255, 255)),
            ('left_hip', 'left_knee', (0, 0, 255)),
            ('left_knee', 'left_ankle', (0, 0, 255)),
            ('right_hip', 'right_knee', (0, 0, 255)),
            ('right_knee', 'right_ankle', (0, 0, 255)),
        ]
        for id in range(self.get_total_person()):
            if self.get_person_score(id) < person_score_target:
                continue
            for part_a, part_b, color in SKELETON_CONNECTIONS:
                if (self._get_kp_coord(id, part_a, 2) > keypoint_score_target and
                        self._get_kp_coord(id, part_b, 2) > keypoint_score_target):
                    x1, y1 = self._get_kp_coord(id, part_a, 0), self._get_kp_coord(id, part_a, 1)
                    x2, y2 = self._get_kp_coord(id, part_b, 0), self._get_kp_coord(id, part_b, 1)

                    self.draw_line(x1, y1, x2, y2, color=color, thickness=2)

    def draw_all_keypoints(self, person_score_target=0.6, keypoint_score_target=0.65):
        for id in range(self.get_total_person()):
            if self.get_person_score(id) < person_score_target:
                continue
            parts = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                     'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                     'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                     'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
            for part in parts:
                score = self._get_kp_coord(id, part, 2)
                if score > keypoint_score_target:
                    x = self._get_kp_coord(id, part, 0)
                    y = self._get_kp_coord(id, part, 1)
                    self.draw_keypoint(x, y)

    def draw_cut_box(self, person_score_target=0.6, keypoint_score_target=0.65):
        for id in range(self.get_total_person()):
            if (self.get_person_score(id) < person_score_target
                    and self.get_nose_score(id) > keypoint_score_target
                    and self.get_left_eye_score(id) > keypoint_score_target
                    and self.get_right_eye_score(id) > keypoint_score_target):
                x_left, x_right, y_top, y_bottom = self.get_cut_box_point(id)
                self.draw_box(x_left, x_right, y_top, y_bottom)

    def detection_box_visualizer(self, test_keyinfor):
        for n in range(len(test_keyinfor)):
            if test_keyinfor[n][-1]:
                self.draw_box(self.img,
                              test_keyinfor[n][0],
                              test_keyinfor[n][2],
                              test_keyinfor[n][1],
                              test_keyinfor[n][3],
                              color=GREEN)
            else:
                self.draw_box(self.img,
                              test_keyinfor[n][0],
                              test_keyinfor[n][2],
                              test_keyinfor[n][1],
                              test_keyinfor[n][3],
                              color=RED)


class DetectVisualizer(ImgProcessor):
    def __init__(self, img, test_keyinfor):
        ImgProcessor.__init__(self, img)
        self.test_keyinfor = test_keyinfor

    def detection_box_visualizer(self, test_keyinfor):
        for n in range(len(test_keyinfor)):
            if test_keyinfor[n][-1]:
                self.draw_box(
                    test_keyinfor[n][0],
                    test_keyinfor[n][1],
                    test_keyinfor[n][2],
                    test_keyinfor[n][3],
                    color=GREEN)
            else:
                self.draw_box(
                    test_keyinfor[n][0],
                    test_keyinfor[n][1],
                    test_keyinfor[n][2],
                    test_keyinfor[n][3],
                    color=RED)

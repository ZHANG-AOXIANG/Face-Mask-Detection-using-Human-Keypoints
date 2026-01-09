import json
import os
import math
import numpy as np


class KeypointsParser:
    def __init__(self, person_keypoints_data):
        self.data = person_keypoints_data

    def _get_kp_coord(self, person_id, part, index):
        return float(self.data['person_list'][person_id]['key_points'][part][index])

    def get_nose_x(self, person_id):
        return self._get_kp_coord(person_id, 'nose', 0)

    def get_nose_y(self, person_id):
        return self._get_kp_coord(person_id, 'nose', 1)

    def get_nose_score(self, person_id):
        return self._get_kp_coord(person_id, 'nose', 2)

    def get_left_eye_x(self, person_id):
        return self._get_kp_coord(person_id, 'left_eye', 0)

    def get_left_eye_y(self, person_id):
        return self._get_kp_coord(person_id, 'left_eye', 1)

    def get_left_eye_score(self, person_id):
        return self._get_kp_coord(person_id, 'left_eye', 2)

    def get_right_eye_x(self, person_id):
        return self._get_kp_coord(person_id, 'right_eye', 0)

    def get_right_eye_y(self, person_id):
        return self._get_kp_coord(person_id, 'right_eye', 1)

    def get_right_eye_score(self, person_id):
        return self._get_kp_coord(person_id, 'right_eye', 2)

    def get_left_ear_x(self, person_id):
        return self._get_kp_coord(person_id, 'left_ear', 0)

    def get_left_ear_y(self, person_id):
        return self._get_kp_coord(person_id, 'left_ear', 1)

    def get_left_ear_score(self, person_id):
        return self._get_kp_coord(person_id, 'left_ear', 2)

    def get_right_ear_x(self, person_id):
        return self._get_kp_coord(person_id, 'right_ear', 0)

    def get_right_ear_y(self, person_id):
        return self._get_kp_coord(person_id, 'right_ear', 1)

    def get_right_ear_score(self, person_id):
        return self._get_kp_coord(person_id, 'right_ear', 2)

    def get_left_shoulder_x(self, person_id):
        return self._get_kp_coord(person_id, 'left_shoulder', 0)

    def get_left_shoulder_y(self, person_id):
        return self._get_kp_coord(person_id, 'left_shoulder', 1)

    def get_left_shoulder_score(self, person_id):
        return self._get_kp_coord(person_id, 'left_shoulder', 2)

    def get_right_shoulder_x(self, person_id):
        return self._get_kp_coord(person_id, 'right_shoulder', 0)

    def get_right_shoulder_y(self, person_id):
        return self._get_kp_coord(person_id, 'right_shoulder', 1)

    def get_right_shoulder_score(self, person_id):
        return self._get_kp_coord(person_id, 'right_shoulder', 2)

    def get_left_elbow_x(self, person_id):
        return self._get_kp_coord(person_id, 'left_elbow', 0)

    def get_left_elbow_y(self, person_id):
        return self._get_kp_coord(person_id, 'left_elbow', 1)

    def get_left_elbow_score(self, person_id):
        return self._get_kp_coord(person_id, 'left_elbow', 2)

    def get_right_elbow_x(self, person_id):
        return self._get_kp_coord(person_id, 'right_elbow', 0)

    def get_right_elbow_y(self, person_id):
        return self._get_kp_coord(person_id, 'right_elbow', 1)

    def get_right_elbow_score(self, person_id):
        return self._get_kp_coord(person_id, 'right_elbow', 2)

    def get_left_wrist_x(self, person_id):
        return self._get_kp_coord(person_id, 'left_wrist', 0)

    def get_left_wrist_y(self, person_id):
        return self._get_kp_coord(person_id, 'left_wrist', 1)

    def get_left_wrist_score(self, person_id):
        return self._get_kp_coord(person_id, 'left_wrist', 2)

    def get_right_wrist_x(self, person_id):
        return self._get_kp_coord(person_id, 'right_wrist', 0)

    def get_right_wrist_y(self, person_id):
        return self._get_kp_coord(person_id, 'right_wrist', 1)

    def get_right_wrist_score(self, person_id):
        return self._get_kp_coord(person_id, 'right_wrist', 2)

    def get_left_hip_x(self, person_id):
        return self._get_kp_coord(person_id, 'left_hip', 0)

    def get_left_hip_y(self, person_id):
        return self._get_kp_coord(person_id, 'left_hip', 1)

    def get_left_hip_score(self, person_id):
        return self._get_kp_coord(person_id, 'left_hip', 2)

    def get_right_hip_x(self, person_id):
        return self._get_kp_coord(person_id, 'right_hip', 0)

    def get_right_hip_y(self, person_id):
        return self._get_kp_coord(person_id, 'right_hip', 1)

    def get_right_hip_score(self, person_id):
        return self._get_kp_coord(person_id, 'right_hip', 2)

    def get_left_knee_x(self, person_id):
        return self._get_kp_coord(person_id, 'left_knee', 0)

    def get_left_knee_y(self, person_id):
        return self._get_kp_coord(person_id, 'left_knee', 1)

    def get_left_knee_score(self, person_id):
        return self._get_kp_coord(person_id, 'left_knee', 2)

    def get_right_knee_x(self, person_id):
        return self._get_kp_coord(person_id, 'right_knee', 0)

    def get_right_knee_y(self, person_id):
        return self._get_kp_coord(person_id, 'right_knee', 1)

    def get_right_knee_score(self, person_id):
        return self._get_kp_coord(person_id, 'right_knee', 2)

    def get_left_ankle_x(self, person_id):
        return self._get_kp_coord(person_id, 'left_ankle', 0)

    def get_left_ankle_y(self, person_id):
        return self._get_kp_coord(person_id, 'left_ankle', 1)

    def get_left_ankle_score(self, person_id):
        return self._get_kp_coord(person_id, 'left_ankle', 2)

    def get_right_ankle_x(self, person_id):
        return self._get_kp_coord(person_id, 'right_ankle', 0)

    def get_right_ankle_y(self, person_id):
        return self._get_kp_coord(person_id, 'right_ankle', 1)

    def get_right_ankle_score(self, person_id):
        return self._get_kp_coord(person_id, 'right_ankle', 2)

    def get_total_person(self):
        return int(self.data["image_info"]["number_of_person"])

    def get_person_score(self, person_id):
        return float(self.data['person_list'][person_id]['person_score'])

    def get_face_keypoints(self, person_id):
        return [
            [self.get_left_eye_x(person_id), self.get_left_eye_y(person_id)],
            [self.get_right_eye_x(person_id), self.get_right_eye_y(person_id)],
            [self.get_nose_x(person_id), self.get_nose_y(person_id)]
        ]

    def get_cut_box_point(self, person_id):
        face_keypoints = self.get_face_keypoints(person_id)
        left_eye = np.array(face_keypoints[0])
        right_eye = np.array(face_keypoints[1])
        nose_x = np.array(face_keypoints[2][0])
        nose_y = np.array(face_keypoints[2][1])
        eye_dist = math.sqrt((right_eye[0] - left_eye[0]) ** 2 + (right_eye[1] - left_eye[1]) ** 2)
        x_left = nose_x - int(1.5 * eye_dist)
        x_right = nose_x + int(1.5 * eye_dist)
        y_top = nose_y - int(1 * eye_dist)
        y_bottom = nose_y + int(2 * eye_dist)
        return x_left, x_right, y_top, y_bottom

    def save_data(self, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)
        print(f"Successfully save to: {output_path}")


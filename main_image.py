from src.core.detector import HumanKeypointDetector
from src.core.face_processor import FaceProcessor
from src.core.classifier import MaskClassifier
from src.utils.drawer import KeypointsDrawer
from src.utils.parser import KeypointsParser
import cv2 as cv
import numpy as np


def run_mask_detection(img_path):
    detector = HumanKeypointDetector(device='auto')
    face_tool = FaceProcessor()
    mask_classifier = MaskClassifier('Model/facemask.pth')

    img = cv.imread(img_path)
    draw_img = img.copy()

    person_data = detector.detect(img_path)
    kp_parser = KeypointsParser(person_data)
    drawer = KeypointsDrawer(draw_img, person_data)

    for i in range(kp_parser.get_total_person()):
        if kp_parser.get_person_score(i) < 0.65:
            continue

        face_kp = kp_parser.get_face_keypoints(i)
        cut_box_point = kp_parser.get_cut_box_point(i)

        face_crop = face_tool.process(img, face_kp)
        cv.imwrite("cutted_img" + str(i) + ".jpg", face_crop)
        is_masked, score = mask_classifier.predict(face_crop)

        color = (0, 255, 0) if is_masked else (0, 0, 255)
        label = "Mask" if is_masked else "No Mask"

        x_left, x_right, y_top, y_bottom = cut_box_point
        drawer.draw_box(x_left, x_right, y_top, y_bottom, color, 2)

        label_text = f"{label}"
        cv.putText(draw_img, label_text, (int(x_left), int(y_top) - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    cv.imshow("Result", draw_img)
    cv.imwrite("assert/test_example_result07.jpg", draw_img)
    cv.waitKey(0)


if __name__ == "__main__":
    img_path01 = "data/data_samples_10/data/no_face_mask_0.jpg"
    img_path02 = "data/Data/data/partA115.jpg"

    img_pth='data/Data/data/partA123.jpg'

    # run_mask_detection(img_path00)
    run_mask_detection(img_pth)
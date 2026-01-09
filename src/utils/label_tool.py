import json
import cv2 as cv

from src.utils.img_tool import ImgProcessor

GREEN = (0, 0, 255)
RED = (0, 0, 225)
BLUE = (225, 0, 0)
PURPLE = (255, 0, 255)


class LabelParser:
    def __init__(self, label_path):
        self.label_path = label_path
        with open(label_path, 'r', encoding='utf-8') as f:
            self.label_data = json.load(f)

    def get_total_label(self):
        return self.label_data

    def get_num_boxes(self):
        return len(self.label_data["regions"])

    def get_tag(self, num):
        return self.label_data["regions"][num]["tags"]

    def get_box_points(self, num):
        x_min = self.label_data["regions"][num]["points"][0]["x"]
        y_min = self.label_data["regions"][num]["points"][0]["y"]
        x_max = self.label_data["regions"][num]["points"][2]["x"]
        y_max = self.label_data["regions"][num]["points"][2]["y"]
        return int(x_min), int(x_max), int(y_min), int(y_max)

    def get_keyinfor(self, num):
        box_points = self.get_box_points(num)
        if self.get_tag(num)[0] == "nomask":
            return [box_points[0], box_points[1], box_points[2], box_points[3], False]
        else:
            return [box_points[0], box_points[1], box_points[2], box_points[3], True]

    def get_img_label_keyinfor(self):
        label_keyinfor = []
        for n in range(self.get_num_boxes()):
            label_keyinfor.append(self.get_keyinfor(n))
        return label_keyinfor

    def get_tags_num(self):
        total_nomask = 0
        total_mask = 0
        for n in range(self.get_num_boxes()):
            print(self.get_tag(n)[0])
            if self.get_tag(n)[0] == "nomask":
                total_nomask = total_nomask + 1
            elif self.get_tag(n)[0] == "mask":
                total_mask = total_mask + 1
        return total_nomask, total_mask


class LabelVisualizer(ImgProcessor, LabelParser):
    def __init__(self, img, label_path):
        ImgProcessor.__init__(self, img)
        LabelParser.__init__(self, label_path)

    def one_box_visualizer(self, img, box_location):
        self.draw_box(img, box_location)

    def all_box_visualizer(self):
        for n in range(self.get_num_boxes()):
            box_points = self.get_box_points(n)
            x_min, x_max, y_min, y_max = box_points
            if self.get_tag(n)[0] == "nomask":
                self.draw_box(x_min, x_max, y_min, y_max, color=PURPLE, thickness=2)
            elif self.get_tag(n)[0] == "mask":
                self.draw_box(x_min, x_max, y_min, y_max, color=BLUE, thickness=2)


if __name__ == "__main__":
    label_path = '../../data/data_samples60/label/no_face_mask_0.json'
    img_pth = '../../data/data_samples60/data/no_face_mask_0.jpg'
    label_path00 = '../../data/data_samples60/label/partA12.json'
    img_pth00 = '../../data/data_samples60/data/partA12.jpg'

    lb_parser = LabelParser(label_path00)
    label_data = lb_parser.label_data
    print(lb_parser.label_data)
    print(lb_parser.get_num_boxes())
    # print(lb_parser.get_box_points(0))
    # for n in range(lb_parser.get_num_boxes()):
    # print(n)
    # print(lb_parser.get_box_points(n))
    # print(lb_parser.get_tag(n))
    img = cv.imread(img_pth00)
    draw_img = img.copy()
    lb_visualizer = LabelVisualizer(draw_img, label_path00)
    lb_visualizer.all_box_visualizer()
    cv.imwrite('../../assert/label_box01.jpg', draw_img)
    print(lb_parser.get_tags_num())

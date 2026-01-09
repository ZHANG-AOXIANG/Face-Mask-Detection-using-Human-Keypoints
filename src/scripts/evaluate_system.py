import os
import json

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.python.ops.gen_functional_ops import case

from src.utils.label_tool import LabelParser, LabelVisualizer
from src.utils.parser import KeypointsParser
from src.utils.drawer import KeypointsDrawer, DetectVisualizer
from src.core.classifier import MaskClassifier
from src.core.detector import HumanKeypointDetector
from src.core.face_processor import FaceProcessor


def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def plot_detection_metrics(metrics, save_dir):
    """
    Args:
        metrics: dict returned by compute_metrics_from_tp_fp_tn_fn
        save_dir: directory to save figures
    """
    ensure_dir(save_dir)

    iou = metrics["iou"]

    plt.figure(figsize=(10, 6))

    plt.plot(iou, metrics["precision"], marker='o', label='Precision')
    plt.plot(iou, metrics["recall"], marker='o', label='Recall')
    plt.plot(iou, metrics["f1"], marker='o', label='F1-score')
    plt.plot(iou, metrics["accuracy"], marker='o', linestyle='--', label='Accuracy')

    plt.xlabel("IoU Threshold")
    plt.ylabel("Score")
    plt.title("Detection Performance vs IoU Threshold")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(save_dir, "metrics_vs_iou.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[Saved] {save_path}")


def plot_fp_fn(test_data, save_dir):
    """
    Args:
        test_data: list of [iou, TP, FN, FP, TN]
    """
    ensure_dir(save_dir)

    iou = [x[0] for x in test_data]
    fp = [x[3] for x in test_data]
    fn = [x[2] for x in test_data]

    plt.figure(figsize=(8, 5))
    plt.plot(iou, fp, marker='o', label='False Positives (FP)')
    plt.plot(iou, fn, marker='o', label='False Negatives (FN)')

    plt.xlabel("IoU Threshold")
    plt.ylabel("Count")
    plt.title("FP / FN vs IoU Threshold")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(save_dir, "fp_fn_vs_iou.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[Saved] {save_path}")


'''
def plot_confusion_heatmap(eval_data, save_dir, iou_index=0):
    """
    Args:
        test_data: list of ([iou, TP, FN, FP, TN])
        save_dir: directory to save
        iou_index: which IoU result to visualize
    """
    ensure_dir(save_dir)

    iou, TP, FN, FP, TN = eval_data[iou_index]

    cm = np.array([
        [TP, FP],
        [FN, TN]
    ])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='YlGnBu',
        xticklabels=['Predicted Mask', 'Predicted NoMask'],
        yticklabels=['Actually Mask', 'Actually NoMask']
    )

    plt.title(f"Detection Confusion Matrix (IoU={iou})")
    plt.xlabel("Prediction")
    plt.ylabel("Actual")
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"confusion_matrix_iou_{iou}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[Saved] {save_path}")
'''


def plot_specific_confusion_matrix(eval_data, save_dir, iou_index=0):
    ensure_dir(save_dir)
    iou, tp, fn, fp, tn = eval_data[iou_index]
    cm = np.array([[tn, fp], [fn, tp]])  # 注意顺序：通常是 [[TN, FP], [FN, TP]]
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted NoMask', 'Predicted Mask'],
                yticklabels=['Actual NoMask', 'Actual Mask'])
    plt.title(f"Confusion Matrix at IoU={iou}")
    save_path = os.path.join(save_dir, f"confusion_matrix_iou_{iou}.png")
    # plt.show()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[Saved] {save_path}")


'''
def plot_roc_curve(metrics, save_dir, iou_index=0):
    """
    根据不同 IoU 下的 FPR 和 Recall 绘制 ROC 曲线
    注意：通常 ROC 是针对分类阈值的，但在目标检测评估中，
    展示不同 IoU 约束下的 ROC 趋势也非常有意义。
    """
    ensure_dir(save_dir)
    iou, tp, fn, fp, tn = eval_data[iou_index]
    fpr = fp / (fp + tn)
    recall = tp / (tp + fn)
    tpr=recall

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, marker='o', label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    ensure_dir(save_dir)
    save_path = os.path.join(save_dir, f"roc_curve.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[Saved] {save_path}")
'''


def compute_metrics_from_tp_fp_tn_fn(eval_data):
    """

    :param eval_data: ([iou, TP, FN, FP, TN])
    :return:
    """
    eval_metrics = [

    ]
    eval_metrics = {
        "iou": [],
        "accuracy": [],
        "recall": [],
        "fpr": [],
        "precision": [],
        "f1": [],
    }
    for iou, tp, fn, fp, tn in eval_data:
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        recall = tp / (tp + fn)
        fpr = fp / (fp + tn)
        precision = tp / (tp + fp)
        f1 = 2 * tp / (2 * tp + fp + fn)
        eval_metrics["iou"].append(iou)
        eval_metrics["accuracy"].append(accuracy)
        eval_metrics["recall"].append(recall)
        eval_metrics["fpr"].append(fpr)
        eval_metrics["precision"].append(precision)
        eval_metrics["f1"].append(f1)
    return eval_metrics


def compute_iou(test_data, label_data):
    iou = -1
    test_x_min = test_data[0]
    test_x_max = test_data[1]
    test_y_min = test_data[2]
    test_y_max = test_data[3]

    label_x_min = label_data[0]
    label_x_max = label_data[1]
    label_y_min = label_data[2]
    label_y_max = label_data[3]

    # there is no inter area
    if (test_y_max < label_y_min or
            test_y_min > label_y_max or
            test_x_max < label_x_min or
            test_x_min > label_x_max):
        return iou

    # print("test data for comupter iou:", test_data)
    # print("test label data for comupter iou:", label_data)
    inter_x_min = max(test_x_min, label_x_min)
    inter_x_max = min(test_x_max, label_x_max)
    inter_y_min = max(label_y_min, test_y_min)
    inter_y_max = min(label_y_max, test_y_max)
    # print("(inter_x_max - inter_x_min) = ",(inter_x_max - inter_x_min))
    # print("(inter_y_max - inter_y_min) = ",(inter_y_max - inter_y_min))
    # print("(test_x_max - test_x_min) = ",(test_x_max - test_x_min))
    # print("(test_y_max - test_y_min) = ",(test_y_max - test_y_min))
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    test_area = (test_x_max - test_x_min) * (test_y_max - test_y_min)
    label_area = (label_x_max - label_x_min) * (label_y_max - label_y_min)

    iou = inter_area / (test_area + label_area - inter_area)
    # print(f"[IoU] {iou}")

    return iou


def get_total_labeled_num(data_dir):
    label_dir = os.path.join(data_dir, "label")
    all_num_mask, all_num_nomask = 0, 0
    for fname in os.listdir(label_dir):
        label_path = os.path.join(label_dir, fname)
        lb_parser = LabelParser(label_path)
        each_nomask, each_mask = lb_parser.get_tags_num()
        all_num_nomask = all_num_nomask + each_nomask
        all_num_mask = all_num_mask + each_mask
    return all_num_nomask, all_num_mask


def test_one_img(img_path):
    img = cv.imread(img_path)
    person_data = detector.detect(img_path)
    kp_parser = KeypointsParser(person_data)
    test_keyinfor = []
    for i in range(kp_parser.get_total_person()):
        if kp_parser.get_person_score(i) < 0.65:
            continue
        cut_box_point = kp_parser.get_cut_box_point(i)
        x_left, x_right, y_top, y_bottom = cut_box_point
        face_crop = face_tool.process(img, kp_parser.get_face_keypoints(i))
        is_masked, score = mask_classifier.predict(face_crop)  # True/False
        label = "mask" if is_masked else "nomask"
        print(label, is_masked)
        test_keyinfor.append([int(x_left), int(x_right), int(y_top), int(y_bottom), is_masked])
    return test_keyinfor


def eval_one_img(test_keyinfor, label_keyinfor, iou_thresh):
    print("Start Evaluation One Image")
    print("Test keyinfor: ", test_keyinfor)
    print("Label keyinfor: ", label_keyinfor)
    tp, fn, fp, tn = 0, 0, 0, 0
    eval_infor = []
    for label_data in label_keyinfor:
        for test_data in test_keyinfor:
            iou = compute_iou(test_data, label_data)
            print("iou: ", iou)
            if iou > iou_thresh:
                match (label_data[-1], test_data[-1]):
                    case (True, True):
                        tp += 1
                    case (True, False):
                        fn += 1
                    case (False, True):
                        fp += 1
                    case (False, False):
                        tn += 1
                print("------" + str(iou) + "------")
                print(iou, label_data, test_data)
                print(tp, fn, fp, tn)
                print("------" + str(iou) + "------")
    print("Finish Evaluation One Image")
    print(tp, fn, fp, tn)
    print("```````````````````````````````````````````````````")
    return tp, fn, fp, tn


def eval_all_img(data_dir, iou_list):
    img_dir = os.path.join(data_dir, "data")
    label_dir = os.path.join(data_dir, "label")
    eval_data = []
    for iou in iou_list:
        TP, FP, TN, FN = 0, 0, 0, 0
        for img_name in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_name)
            label_path = os.path.join(label_dir, img_name[:-4] + ".json")
            print("img_path: ", img_path)
            # print("label_path: ", label_path)

            test_keyinfor = test_one_img(img_path)
            # print("test_keyinfor: ", test_keyinfor)
            lb_parser = LabelParser(label_path)
            label_keyinfor = lb_parser.get_img_label_keyinfor()
            # print("label_keyinfor: ", label_keyinfor)
            tp, fn, fp, tn = eval_one_img(test_keyinfor, label_keyinfor, iou)
            TP += tp
            FN += fn
            FP += fp
            TN += tn

        eval_data.append([iou, TP, FN, FP, TN])
    print(TP, FP, TN, FN)
    return eval_data


def test_box_visualize(img, test_keyinfor):
    draw_img = img.copy()
    drawer = DetectVisualizer(draw_img, test_keyinfor)
    drawer.detection_box_visualizer(test_keyinfor)
    return draw_img


def label_box_visualize(img, label_path):
    draw_img = img.copy()
    drawer = LabelVisualizer(draw_img, label_path)
    drawer.all_box_visualizer()
    return draw_img


def test_and_label_box_visualize(img, test_keyinfor, label_path):
    draw_img = img.copy()
    draw_img = test_box_visualize(draw_img, test_keyinfor)
    draw_img = label_box_visualize(draw_img, label_path)
    return draw_img


def visualize_dateset(data_dir, iou_list):
    img_dir = os.path.join(data_dir, "data")
    label_dir = os.path.join(data_dir, "label")
    done_dir = os.path.join(data_dir, "done_data")
    os.mkdir(done_dir)
    for iou_thresh in iou_list:
        # base_dir=os.path.dirname(data_dir)
        iou_dir = os.path.join(done_dir, "iou" + str(iou_thresh)[2:])
        os.mkdir(iou_dir)
        for img_name in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_name)
            label_path = os.path.join(label_dir, img_name[:-4] + ".json")
            draw_img_path = os.path.join(iou_dir, img_name[:-4] + "-iou" + str(iou_thresh)[2:] + ".jpg")
            # print(img_path)
            # print(label_path)
            # print(draw_img_path)
            img = cv.imread(img_path)
            test_keyinfor = test_one_img(img_path)
            draw_img = test_and_label_box_visualize(img, test_keyinfor, label_path)
            cv.imwrite(draw_img_path, draw_img)


if __name__ == "__main__":
    detector = HumanKeypointDetector(device='auto')
    face_tool = FaceProcessor()
    mask_classifier = MaskClassifier('../../Model/facemask.pth')
    data_dir = "../../data/Data"

    eval_history = {
        "Note": "eval_data: [iou, TP, FN, FP, TN]",
        "actual number of mask": [],
        "actual number of nomask": [],
        "eval_data": [],
        "eval_metrics": [],
    }
    label_num = get_total_labeled_num(data_dir)
    eval_history["actual number of mask"].append(label_num[1])
    eval_history["actual number of nomask"].append(label_num[0])
    # iou_list = [0.2,  0.3,  0.4, 0.5]
    iou_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    #iou_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    # iou_list = [0.1, 0.3, 0.5]
    # print(test_keyinfor)
    # print(label_keyinfor)
    # print(eval_one_img(test_keyinfor, label_keyinfor,iou))
    # visualize_dateset(data_dir, iou_list)
    # print(eval_all_img(data_dir))
    print("label summary: ", get_total_labeled_num(data_dir))
    eval_data = eval_all_img(data_dir, iou_list)
    eval_history["eval_data"].append(eval_data)
    print(eval_data)
    print(compute_metrics_from_tp_fp_tn_fn(eval_data))
    # visualize_dateset(data_dir, iou_list)

    save_dir = "../../assert/evaluation_results_figures"

    eval_metrics = compute_metrics_from_tp_fp_tn_fn(eval_data)
    eval_history["eval_metrics"].append(eval_metrics)
    # 保存操作
    eval_history_path = "../../assert"
    eval_history_path = os.path.join(eval_history_path, "eval_history.json")
    with open(eval_history_path, 'w', encoding='utf-8') as f:
        json.dump(eval_history, f, indent=4, ensure_ascii=False)
    print(f"数据已成功保存至 {eval_history_path}")

    plot_detection_metrics(eval_metrics, save_dir)
    plot_fp_fn(eval_data, save_dir)

    for i in range(len(eval_data)):
        plot_specific_confusion_matrix(eval_data, save_dir, iou_index=i)
        # plot_roc_curve(eval_data, save_dir,iou_index=i)
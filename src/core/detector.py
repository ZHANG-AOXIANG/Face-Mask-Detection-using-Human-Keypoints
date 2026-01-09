import torch
import torchvision
from PIL import Image
import numpy as np
import datetime
from typing import List, Dict, Union


class HumanKeypointDetector:
    def __init__(self, device: str = 'auto', model_weights: str = 'DEFAULT'):
        self.device = self.__auto_select_device(device)
        self.model = self.__load_model(model_weights).to(self.device)
        self.model.eval()
        # 关键点顺序必须与模型输出严格一致
        self.keypoint_labels = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]

    def __auto_select_device(self, device_flag: str) -> torch.device:
        if device_flag == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device_flag)

    def __load_model(self, weights_spec: Union[str, object]):
        if isinstance(weights_spec, str) and weights_spec.endswith('.pth'):
            model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=None, num_keypoints=17)
            model.load_state_dict(torch.load(weights_spec, map_location=self.device))
            return model
        return torchvision.models.detection.keypointrcnn_resnet50_fpn(
            weights=torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        )

    def preprocess(self, image: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        return transform(image).unsqueeze(0).to(self.device)

    def detect(self, image_input: Union[str, Image.Image, np.ndarray, torch.Tensor],
               score_threshold: float = 0.5) -> Dict:
        """
        优化后的检测方法：增加阈值过滤，动态构建关键点字典
        """
        if not isinstance(image_input, torch.Tensor):
            image_tensor = self.preprocess(image_input)
        else:
            image_tensor = image_input.to(self.device)

        with torch.no_grad():
            predictions = self.model(image_tensor)[0]

        # 转移至 CPU 并过滤低置信度结果
        scores = predictions['scores'].cpu().numpy()
        mask = scores > score_threshold

        filtered_boxes = predictions['boxes'].cpu().numpy()[mask]
        filtered_keypoints = predictions['keypoints'].cpu().numpy()[mask]
        filtered_kp_scores = predictions['keypoints_scores'].cpu().numpy()[mask]
        filtered_scores = scores[mask]

        person_detect_list = []
        for pid in range(len(filtered_scores)):
            x_min, y_min, x_max, y_max = filtered_boxes[pid]

            # 使用列表推导式动态生成关键点数据，消除冗余代码
            person_keypoints = {
                label: (float(filtered_keypoints[pid, i, 0]),
                        float(filtered_keypoints[pid, i, 1]),
                        float(filtered_kp_scores[pid, i]))
                for i, label in enumerate(self.keypoint_labels)
            }

            person_detect_list.append({
                "persons_id": pid,
                "height": float(y_max - y_min),  # 保持为 float 方便后期计算
                "width": float(x_max - x_min),
                "person_score": float(filtered_scores[pid]),
                "key_points": person_keypoints  # 键名必须与 Parser 对应
            })

        return {
            "image_info": {
                "detect_date": datetime.datetime.now().strftime('%Y-%m-%d'),
                "number_of_person": len(person_detect_list)
            },
            "person_list": person_detect_list
        }

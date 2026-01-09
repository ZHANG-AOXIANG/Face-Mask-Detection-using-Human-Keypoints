import torch
import torch.nn as nn
import torchvision
import cv2 as cv
import numpy as np


class MaskClassifier:
    def __init__(self, model_path, device='auto'):
        # 自动选择设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # 1. 构造 ResNet18 模型结构
        #self.model = torchvision.models.resnet18(pretrained=False)
        self.model = torchvision.models.resnet18(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)  # 二分类：戴口罩 / 未戴

        # 2. 加载权重 (只加载一次)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式

        # 3. 预处理参数 (保持与你训练时一致)
        self.mean = np.array([0.599, 0.482, 0.510])
        self.std = np.array([0.318, 0.301, 0.299])

    def predict(self, face_img):
        """
        输入：裁剪对齐后的面部 BGR 图像
        输出：bool (True: 戴口罩, False: 未戴口罩), score (置信度)
        """
        if face_img is None or face_img.size == 0:
            return False, 0.0

        # 图像预处理流水线
        img = cv.cvtColor(face_img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, (224, 224))
        img = np.float32(img) / 255.0
        img = (img - self.mean) / self.std
        img = np.clip(img, -1, 1)
        img = img.transpose((2, 0, 1))  # HWC -> CHW

        # 转换为 Tensor 并推理
        input_tensor = torch.from_numpy(img).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            # 你原始逻辑是判断第一个输出分量是否 > 0
            raw_score = output[0][0].item()
            is_masked = raw_score > 0

        return is_masked, raw_score
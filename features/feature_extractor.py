import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet18
import cv2

class FeatureExtractor:
    def __init__(self):
        model = resnet18(pretrained=False)
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.model.eval()
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor()
        ])

    def extract(self, crop):
        tensor = self.transform(crop).unsqueeze(0)
        with torch.no_grad():
            features = self.model(tensor).squeeze().numpy()
        return features

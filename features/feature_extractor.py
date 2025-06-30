# features/feature_extractor.py
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import ResNet18_Weights, resnet18


class FeatureExtractor:
    """
    Extracts a 512‑D feature vector from a person crop
    using the penultimate layer of ResNet‑18.
    """

    def __init__(self, use_cuda: bool = torch.cuda.is_available()):
        # Load ResNet‑18 with ImageNet weights
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Remove the final classifier layer → 512‑dim output
        self.model = nn.Sequential(*list(backbone.children())[:-1])
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model.to(self.device).eval()

        # Transformation pipeline: BGR → RGB, resize, tensor, normalise
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def extract(self, crop: np.ndarray) -> np.ndarray | None:
        """
        Parameters
        ----------
        crop : np.ndarray
            A BGR image patch (OpenCV format).

        Returns
        -------
        np.ndarray | None
            512‑D feature vector or None if the crop is empty.
        """
        if crop is None or crop.size == 0:
            return None

        # OpenCV → RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # Apply transforms and push to device
        tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.model(tensor).squeeze().cpu().numpy()  # (512,)

        return feat.astype(np.float32)

from __future__ import annotations

import torch.nn as nn
from torchvision import models


class ResNet18Transfer(nn.Module):
    """ResNet18 with pretrained ImageNet weights, fine-tuned for EuroSAT."""

    def __init__(self, num_classes: int = 10, freeze_backbone: bool = False) -> None:
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


class ResNet18FineTune(nn.Module):
    """ResNet18 with two-stage fine-tuning support.

    Stage 1: train only the classifier head (freeze_backbone=True).
    Stage 2: unfreeze and train all layers with a lower learning rate.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )

    def freeze_backbone(self) -> None:
        for name, param in self.backbone.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)

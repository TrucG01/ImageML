"""Model factory for DAVID segmentation."""

from __future__ import annotations

from torch import nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Construct a DeepLabV3+ model with a MobileNetV3 backbone."""
    model = deeplabv3_mobilenet_v3_large(weights="DEFAULT" if pretrained else None)
    model.classifier = DeepLabHead(1280, num_classes)
    if model.aux_classifier is not None:
        model.aux_classifier = FCNHead(256, num_classes)
    return model

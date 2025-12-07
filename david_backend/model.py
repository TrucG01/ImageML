"""Model factory for DAVID segmentation."""

from __future__ import annotations

from torch import nn
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)
from torchvision.ops.misc import FrozenBatchNorm2d


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Construct a DeepLabV3 model with a ResNet50 backbone.

    Uses torchvision factory to produce a classifier aligned with backbone features,
    avoiding channel mismatches seen with MobileNetV3 variants.
    """
    # When weights are provided, torchvision expects num_classes to match the
    # pretrained dataset (typically 21). To use a custom class count, initialize
    # without weights and optionally load backbone weights separately later.
    weights = None if num_classes != 21 else (DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None)
    # Use FrozenBatchNorm2d to avoid training-time BN issues with small batch sizes.
    model = deeplabv3_resnet50(weights=weights, num_classes=num_classes, norm_layer=FrozenBatchNorm2d)

    # Ensure all BatchNorm2d layers (including ASPP classifier) are frozen to avoid
    # "Expected more than 1 value per channel" errors with batch_size=1.
    def _replace_bn(module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, nn.BatchNorm2d):
                frozen = FrozenBatchNorm2d(child.num_features)
                setattr(module, name, frozen)
            else:
                _replace_bn(child)

    _replace_bn(model)
    return model

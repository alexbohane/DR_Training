import torch.nn as nn
from torchvision import models

EFF_VARIANTS = {
    "b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights),
    "b1": (models.efficientnet_b1, models.EfficientNet_B1_Weights),
    "b2": (models.efficientnet_b2, models.EfficientNet_B2_Weights),
    # add v2_s / v2_m / v2_l as needed
}

def get_efficientnet(variant="b2", num_classes=2, pretrained=True):
    builder, weight_enum = EFF_VARIANTS[variant]
    weights = weight_enum.DEFAULT  # if pretrained else None
    model = builder(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


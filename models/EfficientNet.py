from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
import torch.nn as nn

def get_efficientnet_b2_model(num_classes=2):
    weights = EfficientNet_B2_Weights.DEFAULT
    model = efficientnet_b2(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
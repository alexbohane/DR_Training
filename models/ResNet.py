from torchvision import models
import torch.nn as nn

def get_resnet18_model(num_classes=4):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


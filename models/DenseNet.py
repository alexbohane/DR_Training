from torchvision import models
import torch.nn as nn

def get_densenet121_model(num_classes=4):
    model = models.densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model

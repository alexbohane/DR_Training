import timm
import torch.nn as nn

def get_efficientnet_b2_model(num_classes=2):
    model = timm.create_model('efficientnet_b2', pretrained=True)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model


# from torchvision import models
# import torch.nn as nn

# # def get_resnet18_model(num_classes=2):
# #     model = models.resnet18(pretrained=True)
# #     model.fc = nn.Linear(model.fc.in_features, num_classes)
# #     return model


# def get_resnet50_model(num_classes=2, in_channels=3):
#     model = models.resnet18(pretrained=True)
#     if in_channels == 1:
#         # Change first conv layer to accept 1 channel
#         weight = model.conv1.weight.data
#         model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         # Optionally, copy weights from green channel or average
#         model.conv1.weight.data = weight[:,1:2,:,:]  # Use green channel weights
#     model.fc = nn.Linear(model.fc.in_features, num_classes)
#     return model

    # then call model = get_resnet18_model(num_classes=2, in_channels=1)

# from torchvision.models.quantization import resnet18
# from torchvision.models.quantization import ResNet18_QuantizedWeights
# import torch.nn as nn

# def get_quantized_resnet18_model(num_classes=2):
#     weights = ResNet18_QuantizedWeights.DEFAULT
#     model = resnet18(weights=weights, quantize=True)
#     model.fc = nn.Linear(model.fc.in_features, num_classes)
#     return model



from torchvision.models.quantization import resnet18, ResNet18_QuantizedWeights
import torch.nn as nn

def get_quantized_resnet18_model(num_classes=2):
    # Load pretrained quantized weights
    weights = ResNet18_QuantizedWeights.DEFAULT

    # Load the model with weights and quantization
    model = resnet18(weights=weights, quantize=True)

    # Update the classifier layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Get the official preprocessing transform for this model
    transform = weights.transforms()

    return model, transform

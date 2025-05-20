import torch.nn as nn
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np

def get_last_conv_layer(model):
    for layer in reversed(list(model.modules())):
        if isinstance(layer, nn.Conv2d):
            return layer
    raise ValueError("No Conv2d layer found in the model.")

def initialize_gradcam(model):
    target_layer = get_last_conv_layer(model)
    grad_cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])
    return grad_cam

def apply_gradcam(grad_cam, img_tensor, original_img, opacity=0.3):
    original_img_float = original_img / np.max(original_img)

    grayscale_cam = grad_cam(input_tensor=img_tensor, targets=None)[0]
    grad_cam._hooks_enabled = False

    overlay_rgb = show_cam_on_image(
        original_img_float,
        grayscale_cam,
        use_rgb=True,
        image_weight=(1 - opacity)
    )
    return overlay_rgb

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

class OpenCV_DR_Preprocessor:
    def __init__(self, apply_clahe=True):
        self.img_size = (600, 600)
        self.apply_clahe = apply_clahe

        # ImageNet normalization
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # SSL Normalization
        # self.mean = [0.5, 0.5, 0.5]
        # self.std = [0.5, 0.5, 0.5]

        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)


    def load_image(self, input):
        """Loads an image in RGB format from file path or PIL.Image"""
        if isinstance(input, str):  # file path
            img = cv2.imread(input)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(input, Image.Image):  # PIL Image
            img = np.array(input.convert("RGB"))  # ensure RGB and convert to np.array
        else:
            raise ValueError("Unsupported input type for image loading.")
        return img


    def apply_clahe_rgb(self, img):
        """Apply CLAHE to the L-channel in LAB color space."""
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)

        merged = cv2.merge((cl, a, b))
        img = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        return img

    def center_crop(self, img):
        h, w, _ = img.shape
        min_dim = min(h, w)
        start_x = (w - min_dim) // 2
        start_y = (h - min_dim) // 2
        return img[start_y:start_y+min_dim, start_x:start_x+min_dim]

    def preprocess(self, img_path):
        img = self.load_image(img_path)

        if self.apply_clahe:
            img = self.apply_clahe_rgb(img)

        img = self.center_crop(img)

        img = cv2.resize(img, self.img_size)  #
        img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]

        # FOR VISUALISATION
        image = img.copy()

        # Convert (H, W, C) to (C, H, W) for PyTorch
        img = np.transpose(img, (2, 0, 1))

        img = torch.from_numpy(img)  # Convert to tensor
        img = self.normalize(img)    # Normalize using ImageNet mean and std

        return img, image


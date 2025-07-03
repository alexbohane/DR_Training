import torch
import numpy as np
from utils.gradcam import apply_gradcam
from utils.pdf import create_report
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

class DRPredictor:
    def __init__(self, model, preprocessor, gradcam, class_labels, device):
        self.model = model
        self.preprocessor = preprocessor
        self.gradcam = gradcam
        self.class_labels = class_labels
        self.device = device

    def run_inference(self, img, opacity=0.3):
        img_tensor, original_img = self.preprocessor.preprocess(img)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()
            predicted_class = int(np.argmax(probs))
            confidence = float(probs[predicted_class])

        overlay = apply_gradcam(self.gradcam, img_tensor, original_img, opacity)
        label = self.class_labels[predicted_class]

        return overlay, probs, predicted_class, confidence, label, img  # raw PIL image

    def predict(self, img, opacity=0.3):
        overlay, probs, pred_class, conf, label, pil_img = self.run_inference(img, opacity)
        summary = f"Prediction: {label} (class {pred_class}, confidence: {conf:.2f})"
        pdf_path = create_report(pil_img, label, conf, probs, self.class_labels)

        # Generate matplotlib figure
        fig, ax = plt.subplots()
        ax.bar(['Healthy', 'Unhealthy'], probs)
        ax.set_ylim(0, 1)
        ax.set_title("Class Probabilities")
        ax.set_ylabel("Probability")

        return {
            "overlay": overlay,
            "plot": fig,
            "summary": summary,
            "pdf_path": pdf_path
        }

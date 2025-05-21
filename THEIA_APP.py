import gradio as gr
import torch
from models.ResNet import get_resnet18_model
from preprocess_class import OpenCV_DR_Preprocessor
from utils.gradcam import initialize_gradcam
from predictor import DRPredictor

# --- Load model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_resnet18_model(num_classes=4)
model.load_state_dict(torch.load("saved_models/resnet18_messidor.pth", map_location=device))
model.to(device)
model.eval()

class_labels = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe"
}

# --- Init pipeline ---
preprocessor = OpenCV_DR_Preprocessor(apply_clahe=True)
grad_cam = initialize_gradcam(model)
predictor = DRPredictor(model, preprocessor, grad_cam, class_labels, device)

# --- Gradio App ---
def run_app(img, opacity):
    result = predictor.predict(img, opacity)
    return result["overlay"], result["plot"], result["summary"], result["pdf_path"]

app = gr.Interface(
    fn=run_app,
    inputs=[
        gr.Image(type="pil"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.3, label="Overlay Opacity")
    ],
    outputs=[
        gr.Image(label="Grad-CAM Overlay", height=300, width=300),
        gr.Plot(label="Prediction Probabilities"),
        gr.Text(label="Final Prediction"),
        gr.File(label="Download Report")
    ],
    title="Diabetic Retinopathy Classifier",
    description="Upload a fundus image and adjust overlay opacity to visualize DR detection."
)

if __name__ == "__main__":
    app.launch(share=True)

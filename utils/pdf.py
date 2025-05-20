import matplotlib.pyplot as plt
from fpdf import FPDF
from PIL import Image
import tempfile
import os

def create_report(pil_img, label, confidence, probs, class_labels):
    # Create a temporary image file from the PIL image
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_img:
        pil_img.save(tmp_img.name)
        image_path = tmp_img.name

    # Create a temporary output path for the PDF
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
        output_path = tmp_pdf.name

    # --- Generate PDF ---
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # --- Title ---
    pdf.cell(200, 10, txt="Diabetic Retinopathy Screening Report", ln=True, align="C")
    pdf.ln(10)

    # --- Text Info ---
    pdf.cell(200, 10, txt=f"Prediction: {label}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence: {confidence:.2f}", ln=True)
    pdf.ln(10)

    # --- Bar chart as text (optional) ---
    pdf.set_font("Arial", size=10)
    for idx, prob in enumerate(probs):
        class_name = class_labels[idx]
        pdf.cell(200, 8, txt=f"{class_name}: {prob:.2f}", ln=True)

    # --- Insert the fundus image ---
    pdf.image(image_path, x=10, y=110, w=100)

    # --- Save the PDF ---
    pdf.output(output_path)

    # Optional cleanup (keep only the PDF)
    os.remove(image_path)

    return output_path
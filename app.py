import torch
import torch.nn as nn
from torchvision import models
from flask import Flask, render_template, request
from torchvision import transforms
from PIL import Image
import os

app = Flask(__name__)

#model architecture - matches training
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)  

#load trained weights
model.load_state_dict(torch.load('celebrity_classifier.pth', map_location=torch.device('cpu'), weights_only=True))
model.eval()

#image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = torch.sigmoid(model(image))
        return "AI-Generated" if output.item() > 0.5 else "Real"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No selected file")

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)
        prediction = predict_image(filepath)

        return render_template("index.html", filename=file.filename, prediction=prediction)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
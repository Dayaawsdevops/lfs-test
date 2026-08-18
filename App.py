"""
Image Classification API — FastAPI
Accepts image uploads, returns predicted class + confidence scores
"""

import os
import io
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Image Classification API",
    description="Upload an image, get back the predicted class and confidence score",
    version="1.0.0"
)

# ─────────────────────────────────────────────────────────────────
# Globals — loaded once at startup
# ─────────────────────────────────────────────────────────────────
MODEL_DIR   = Path(os.getenv("MODEL_DIR", "models"))
model       = None
class_names = []
img_size    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet normalisation (same as training)
inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


@app.on_event("startup")
def load_model():
    global model, class_names, img_size

    metadata_path = MODEL_DIR / "metadata.json"
    model_path    = MODEL_DIR / "best_model.pth"

    if not metadata_path.exists() or not model_path.exists():
        logger.warning("Model files not found. Run training first.")
        return

    with open(metadata_path) as f:
        meta = json.load(f)

    class_names = meta["class_names"]
    num_classes = meta["num_classes"]
    img_size    = meta.get("img_size", 224)

    # Rebuild model architecture and load weights
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    m.load_state_dict(torch.load(model_path, map_location=DEVICE))
    m.eval()

    model = m.to(DEVICE)
    logger.info(f"Model loaded — {num_classes} classes: {class_names}")


# ─────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence:      float
    all_scores:      dict[str, float]


# ─────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Image Classification API", "status": "running"}


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "classes": class_names}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Upload a JPG/PNG image and receive a classification result."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400,
                            detail=f"Expected an image file, got: {file.content_type}")

    try:
        contents = await file.read()
        image    = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor   = inference_transforms(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs    = model(tensor)
            probs      = torch.softmax(outputs, dim=1)[0]
            confidence, pred_idx = probs.max(0)

        predicted_class = class_names[pred_idx.item()]
        all_scores = {
            cls: round(probs[i].item(), 4)
            for i, cls in enumerate(class_names)
        }

        logger.info(f"Prediction: {predicted_class} ({confidence.item():.2%}) "
                    f"for file: {file.filename}")

        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=round(confidence.item(), 4),
            all_scores=all_scores
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "architecture": "ResNet18 (transfer learning)",
        "num_classes":  len(class_names),
        "class_names":  class_names,
        "device":       str(DEVICE)
    }


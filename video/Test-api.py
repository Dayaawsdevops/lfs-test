"""
Tests for Image Classification API
Covers: health check, prediction, validation, quality gate
"""
import io
import json
import os
import pytest
from PIL import Image
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from src.app import app
    return TestClient(app)


def make_dummy_image(width=224, height=224, color=(255, 0, 0)):
    """Create a simple red PNG image in memory."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


# ── API endpoint tests ────────────────────────────────────────────

def test_root_returns_200(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "status" in r.json()


def test_health_no_model_returns_503(client):
    import src.app as app_module
    original = app_module.model
    app_module.model = None
    r = client.get("/health")
    assert r.status_code == 503
    app_module.model = original


def test_predict_no_model_returns_503(client):
    import src.app as app_module
    original = app_module.model
    app_module.model = None
    img = make_dummy_image()
    r = client.post("/predict", files={"file": ("test.png", img, "image/png")})
    assert r.status_code == 503
    app_module.model = original


def test_predict_non_image_returns_400(client):
    r = client.post("/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")})
    assert r.status_code == 400


def test_predict_missing_file_returns_422(client):
    r = client.post("/predict")
    assert r.status_code == 422


# ── Quality gate tests ────────────────────────────────────────────

def test_metadata_file_exists():
    assert os.path.exists("models/metadata.json"), \
        "Run training first: python src/train.py"


def test_val_accuracy_above_threshold():
    if not os.path.exists("models/metadata.json"):
        pytest.skip("No metadata.json found — run training first")

    with open("models/metadata.json") as f:
        meta = json.load(f)

    val_acc   = meta["val_acc"]
    threshold = 0.75
    assert val_acc >= threshold, \
        f"Val accuracy {val_acc:.3f} is below threshold {threshold}"
    print(f"Val accuracy: {val_acc:.3f} (threshold: {threshold}) PASS")


def test_metadata_has_required_fields():
    if not os.path.exists("models/metadata.json"):
        pytest.skip("No metadata.json found")

    with open("models/metadata.json") as f:
        meta = json.load(f)

    for field in ["val_acc", "num_classes", "class_names", "model_arch"]:
        assert field in meta, f"Missing field: {field}"


# ── Unit tests ────────────────────────────────────────────────────

def test_pil_image_creation():
    img = make_dummy_image()
    pil = Image.open(img)
    assert pil.mode == "RGB"
    assert pil.size == (224, 224)


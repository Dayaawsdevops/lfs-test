#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# EC2 GPU Instance Bootstrap Script
# Run this ONCE on a fresh AWS Deep Learning AMI (Ubuntu)
# Recommended instance: g4dn.xlarge ($0.526/hr)
#
# Usage:
#   chmod +x scripts/setup_ec2.sh
#   ./scripts/setup_ec2.sh
# ─────────────────────────────────────────────────────────────────

set -e
echo "========================================"
echo "  EC2 GPU Setup for Image Classification"
echo "========================================"

# ── System packages ───────────────────────────────────────────────
echo "[1/7] Updating system packages..."
sudo apt-get update -y && sudo apt-get upgrade -y
sudo apt-get install -y git curl unzip python3-pip python3-venv

# ── Python virtual environment ────────────────────────────────────
echo "[2/7] Creating Python virtual environment..."
cd ~
python3 -m venv venv
source venv/bin/activate

# ── PyTorch with CUDA ─────────────────────────────────────────────
echo "[3/7] Installing PyTorch with CUDA 11.8 support..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify GPU is detected
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:             {torch.cuda.get_device_name(0)}')
    print(f'VRAM:            {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# ── ML dependencies ───────────────────────────────────────────────
echo "[4/7] Installing ML dependencies..."
pip install \
  mlflow==2.10.2 \
  fastapi==0.109.2 \
  uvicorn[standard]==0.27.1 \
  Pillow==10.2.0 \
  python-multipart==0.0.9 \
  boto3==1.34.0 \
  dvc[s3]==3.39.0 \
  pytest==8.0.0 \
  httpx==0.26.0

# ── AWS CLI ───────────────────────────────────────────────────────
echo "[5/7] Installing AWS CLI..."
curl -s "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip -q awscliv2.zip
sudo ./aws/install
rm -rf awscliv2.zip aws/
aws --version

# ── DVC + S3 remote ───────────────────────────────────────────────
echo "[6/7] Configuring DVC S3 remote..."
cd ~/image-mlops 2>/dev/null || mkdir -p ~/image-mlops && cd ~/image-mlops
dvc init 2>/dev/null || true
echo ""
echo "  >>> Set your S3 bucket:"
echo "      dvc remote add -d myremote s3://YOUR-BUCKET-NAME/dvc-store"
echo "      dvc remote modify myremote region YOUR-REGION"
echo ""

# ── Folder structure ──────────────────────────────────────────────
echo "[7/7] Creating project folders..."
mkdir -p ~/image-mlops/{data/images,models,src,tests}

# ── Auto-activate venv on login ───────────────────────────────────
echo 'source ~/venv/bin/activate' >> ~/.bashrc
echo 'cd ~/image-mlops'          >> ~/.bashrc

echo ""
echo "========================================"
echo "  Setup complete!"
echo "  Next steps:"
echo "  1. Upload your dataset:  aws s3 sync ./data s3://YOUR-BUCKET/data/"
echo "  2. Pull data with DVC:   dvc pull"
echo "  3. Run training:         python src/train.py --epochs 10"
echo "========================================"

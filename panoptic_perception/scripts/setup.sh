#!/bin/bash
set -e

WORKSPACE="/workspace" #change to your project root
DATA_DIR="${WORKSPACE}/data"

IMAGES_URL="http://128.32.162.150/bdd100k/bdd100k_images_100k.zip"
DET_LABELS_URL="http://128.32.162.150/bdd100k/bdd100k_labels.zip"
DRIVABLE_MAPS_URL="http://128.32.162.150/bdd100k/bdd100k_drivable_maps.zip"

echo "========================================"
echo "  Vast.ai Instance Setup"
echo "========================================"
echo ""

# ---- Step 1: System info ----
echo "[1/6] System Info"
echo "----------------------------------------"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "No GPU found"
python3 --version
echo ""

echo "[2/6] Creating data directory"
echo "----------------------------------------"
mkdir -p "${DATA_DIR}"
echo "Data directory created at ${DATA_DIR}"
echo ""

# ---- Step 3: Download BDD100K data ----
echo "[3/6] Downloading BDD100K data"
echo "----------------------------------------"
download_and_extract() {
    local url="$1"
    local dest="$2"
    local name="$3"

    if [[ "${url}" == *"REPLACE"* ]]; then
        echo "SKIPPED: ${name} - URL not set (replace placeholder in script)"
        return
    fi

    local filename=$(basename "${url}")

    mkdir -p "${dest}"
    echo "Downloading ${name}..."
    wget -q --show-progress -O "${dest}/${filename}" "${url}"

    if [[ "${filename}" == *.zip ]]; then
        echo "Extracting ${filename}..."
        unzip -q -o "${dest}/${filename}" -d "${dest}"
        rm "${dest}/${filename}"
    elif [[ "${filename}" == *.tar.gz || "${filename}" == *.tgz ]]; then
        echo "Extracting ${filename}..."
        tar -xzf "${dest}/${filename}" -C "${dest}"
        rm "${dest}/${filename}"
    elif [[ "${filename}" == *.tar ]]; then
        echo "Extracting ${filename}..."
        tar -xf "${dest}/${filename}" -C "${dest}"
        rm "${dest}/${filename}"
    fi

    echo "Done: ${name}"
    echo ""
}

download_and_extract "${IMAGES_URL}" "${DATA_DIR}/100k" "BDD100K Images"
download_and_extract "${DET_LABELS_URL}" "${DATA_DIR}/bdd100k_labels" "BDD100K Detection Labels"
download_and_extract "${DRIVABLE_MAPS_URL}" "${DATA_DIR}/drivable_maps" "BDD100K Drivable Maps"

# Final structure:
#   data/100k/100k/{train,val}/*.jpg                              ← images
#   data/bdd100k_labels/bdd100k_labels/{train,val}/*.json         ← detection labels
#   data/drivable_maps/drivable_maps/color_labels/{train,val}/    ← color drivable masks
#   data/drivable_maps/drivable_maps/labels/{train,val}/          ← drivable masks (grayscale)

# ---- Step 4: Verify data directories ----
echo "[4/6] Verifying extracted data"
echo "----------------------------------------"
for dir in "${DATA_DIR}/100k/100k/train" "${DATA_DIR}/100k/100k/val" "${DATA_DIR}/bdd100k_labels/bdd100k_labels/train" "${DATA_DIR}/bdd100k_labels/bdd100k_labels/val" "${DATA_DIR}/drivable_maps/drivable_maps/labels/train" "${DATA_DIR}/drivable_maps/drivable_maps/labels/val"; do
    if [ -d "${dir}" ]; then
        count=$(ls "${dir}" | wc -l)
        echo "OK: ${dir} (${count} files)"
    else
        echo "MISSING: ${dir}"
    fi
done
echo ""

# ---- Step 5: Check Python version and install system deps ----
echo "[5/6] Checking Python and installing system dependencies"
echo "----------------------------------------"
sudo apt update -qq

if ! command -v python3 &> /dev/null; then
    NEED_INSTALL=1
else
    NEED_INSTALL=$(python3 -c "import sys; print(int(sys.version_info < (3, 9)))")
fi

if [ "$NEED_INSTALL" -eq 1 ]; then
    echo "Installing Python 3.9+..."
    sudo apt install -y python3 python3-pip python3-venv
else
    echo "Python OK: $(python3 --version)"
fi
echo ""

# ---- Step 6: Install Python packages ----
echo "[6/6] Installing Python packages"
echo "----------------------------------------"
pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

pip install \
    albumentations==2.0.3 \
    opencv-python-headless==4.12.0.88 \
    numpy==2.1.2 scipy==1.15.3 pandas==2.3.3 \
    matplotlib seaborn \
    wandb==0.23.0 \
    tqdm==4.67.1 \
    terminaltables==3.1.10 \
    PyYAML==6.0.2 \
    pydantic==2.12.4 \
    Pillow==11.0.0 \
    transformers \
    onnxruntime-gpu \
    tensorrt \
    pycuda

echo ""
echo "Verifying installations..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, CUDA version: {torch.version.cuda}')"
python3 -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')"
python3 -c "import albumentations; print(f'Albumentations: {albumentations.__version__}')"
python3 -c "import wandb; print(f'WandB: {wandb.__version__}')"

# ---- Setup complete ----
echo ""
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. wandb login"
echo "  2. Start training:"
echo "     python3 -m panoptic_perception.scripts.train.train --config panoptic_perception/configs/trainer/train_kwargs.json"
echo ""

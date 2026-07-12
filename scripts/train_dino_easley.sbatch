#!/bin/bash
#SBATCH --job-name=ag_dino_v2             # Name of the job
#SBATCH --output=dino_%j.out               # Standard output (%j is the job ID)
#SBATCH --error=dino_%j.err                # Standard error
#SBATCH --partition=gpu4                   # Use a node in the gpu4 partition
#SBATCH --gres=gpu:tesla:1                 # Request one Tesla T4 GPU
#SBATCH --nodes=1                          # Use one compute node
#SBATCH --ntasks=1                         # Run one training process
#SBATCH --cpus-per-task=10                 # CPU cores for data loading
#SBATCH --mem=64G                          # System RAM
#SBATCH --time=2-00:00:00                  # Maximum runtime: two days
#SBATCH --mail-user=mza0288@auburn.edu     # Email notifications
#SBATCH --mail-type=ALL                    # Email on start, failure, and completion 

set -euo pipefail

# Project locations on Easley. Change these if you used different directories.
PROJECT_DIR="$HOME/AG_Dataset/AG-Foundational-Model"
CONDA_ENV_NAME="ag-foundation"
SHARDS="$HOME/AG_Dataset/shards/dataset-*.tar"
OUTPUT_DIR="$HOME/AG_Dataset/runs/dino-v2-production"
CONFIG="$PROJECT_DIR/configs/wds_dino_pretrain_v2.yaml"

echo "Starting agricultural DINOv2-style training."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Project: $PROJECT_DIR"
echo "Dataset: $SHARDS"
echo "Output: $OUTPUT_DIR"

# Initialize Conda in this non-interactive Slurm shell and activate the project.
if ! command -v conda >/dev/null 2>&1; then
    module load anaconda 2>/dev/null || module load miniconda 2>/dev/null || {
        echo "ERROR: Conda is unavailable. Load Easley's Anaconda/Miniconda module." >&2
        exit 2
    }
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"
export PYTHONNOUSERSITE=1
cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR"

# Make the source tree importable even if the editable install was not created yet.
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

# If the package is not available in this environment, install it in editable mode.
if ! python -c "import ag_foundation" >/dev/null 2>&1; then
    echo "ag_foundation is not installed in the active environment. Installing editable package..."
    python -m pip install -e "$PROJECT_DIR"
fi

# Prevent each DataLoader process from creating extra CPU thread pools.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TORCH_HOME="/scratch/$USER/ag-foundation/cache/torch"
export HF_HOME="/scratch/$USER/ag-foundation/cache/huggingface"
mkdir -p "$TORCH_HOME" "$HF_HOME"

# The PyTorch wheel includes its CUDA runtime; use the node's NVIDIA driver directly.
# Loading a CUDA toolkit module here can override libcuda and break initialization.

echo "GPU status:"
nvidia-smi

echo "Checking PyTorch GPU access:"
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); assert torch.cuda.is_available()"

echo "Beginning training."
python -u -m ag_foundation train-dino \
    --config "$CONFIG" \
    --data-root "$SHARDS" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda \
    --precision fp16 \
    --batch-size 2 \
    --gradient-accumulation-steps 128 \
    --epoch-batches 10000 \
    --val-epoch-batches 200 \
    --num-workers 8 \
    --pretrained-backbone \
    --pretrained-source dinov2

echo "Training job finished."
echo "Finish time: $(date)"
echo "Results: $OUTPUT_DIR"

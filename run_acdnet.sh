#!/bin/bash

# ACDNet Training Script
# Usage: ./run_acdnet.sh [model] [dataset] [data_root]

set -e

# Default values
MODEL=${1:-acdnet}
DATASET=${2:-esc50}
DATA_ROOT=${3:-/path/to/dataset}

# Training configuration
EPOCHS=2000
BATCH_SIZE=64
LR=0.1
WEIGHT_DECAY=5e-4
NUM_WORKERS=4

# Output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./checkpoints/${MODEL}_${DATASET}_${TIMESTAMP}"

echo "========================================"
echo "ACDNet Training"
echo "========================================"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Data root: $DATA_ROOT"
echo "Output: $OUTPUT_DIR"
echo "========================================"

# Check if data root exists
if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: Data root directory does not exist: $DATA_ROOT"
    echo "Please provide a valid path as the third argument"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
python train_acdnet.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --data-root "$DATA_ROOT" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --num-workers "$NUM_WORKERS" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "========================================"
echo "Training completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================"

# Show best accuracy
if [ -f "$OUTPUT_DIR/checkpoint_best.pt" ]; then
    echo ""
    echo "Best model checkpoint: $OUTPUT_DIR/checkpoint_best.pt"
    echo "To resume training or evaluate, use:"
    echo "  python train_acdnet.py --resume $OUTPUT_DIR/checkpoint_best.pt"
fi

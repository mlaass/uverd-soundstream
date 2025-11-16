#!/bin/bash

# ACDNet Training Script
#
# Defaults match ACDNet paper: SGD + MultiStepLR + warmup
#
# Basic Usage:
#   ./train_acdnet.sh                                    # Train with paper's setup
#   ./train_acdnet.sh --model micro_acdnet               # Train Micro-ACDNet
#   ./train_acdnet.sh --dataset esc10                    # Train on ESC-10 subset
#   ./train_acdnet.sh --data-root /path/to/dataset       # Custom dataset path
#   ./train_acdnet.sh --fold 2                           # Use different fold
#   ./train_acdnet.sh --resume checkpoint.pt             # Resume training
#   ./train_acdnet.sh --epochs 4000                      # Train for 4000 epochs
#
# All Python arguments are supported. Pass them after the shell script args:
#
# Examples:
#   # Use AdamW instead of SGD
#   ./train_acdnet.sh -- --optimizer adamw --lr 3e-4 --weight-decay 1e-4
#
#   # Use Cosine annealing
#   ./train_acdnet.sh -- --lr-scheduler cosine --lr-min-factor 0.001
#
#   # Disable warmup
#   ./train_acdnet.sh -- --warmup-epochs 0
#
#   # Custom MultiStepLR milestones
#   ./train_acdnet.sh -- --lr-milestones 800 1600 --lr-gamma 0.5
#
#   # Save checkpoints every 500 epochs
#   ./train_acdnet.sh -- --save-interval 500
#
#   # Augmentation control
#   ./train_acdnet.sh -- --augmentation-multiplier 1    # Disable data expansion (1x)
#   ./train_acdnet.sh -- --augmentation-multiplier 8    # 8x data expansion
#   ./train_acdnet.sh -- --no-augment --no-mixup        # Disable all augmentations
#   ./train_acdnet.sh -- --no-mixup                     # Keep augmentations, disable mixup

set -e

# Parse arguments
MODEL="acdnet"
DATASET="esc50"
DATA_ROOT="datasets/ESC-50-master"
FOLD=1
RESUME=""
LOG_DIR="./logs_acdnet"
EPOCHS=2000  # Default number of epochs

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --fold)
            FOLD="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./train_acdnet.sh [--model acdnet|micro_acdnet] [--dataset esc50|esc10|urbansound8k] [--data-root PATH] [--fold N] [--resume PATH] [--log-dir PATH] [--epochs N]"
            exit 1
            ;;
    esac
done

# Training configuration
# Note: LR defaults are now set in train_acdnet.py to match the paper:
#   - SGD: lr=0.1, weight_decay=5e-4, momentum=0.9
#   - MultiStepLR: milestones=[600, 1200, 1800], gamma=0.1
#   - Warmup: 10 epochs at 0.1x LR
BATCH_SIZE=64
NUM_WORKERS=16
ACCUMULATION_STEPS=1

# Output directory (Python script will add model/dataset/fold/timestamp subdirectory)
OUTPUT_DIR="./checkpoints_acdnet"

echo "========================================"
echo "ACDNet Training"
echo "========================================"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Fold: $FOLD"
echo "Epochs: $EPOCHS"
echo "Data root: $DATA_ROOT"
echo "Checkpoint base directory: $OUTPUT_DIR"
echo "Log base directory: $LOG_DIR"
if [ -n "$RESUME" ]; then
    echo "Resuming from: $RESUME"
fi
echo "========================================"
echo ""

# Check if data root exists
if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: Data root directory does not exist: $DATA_ROOT"
    echo "Please provide a valid path with --data-root"
    echo ""
    echo "Example: ./train_acdnet.sh --data-root datasets/ESC-50-master"
    exit 1
fi

# Build training arguments
# Note: LR, optimizer, scheduler, warmup defaults are now in train_acdnet.py
# Override them with additional flags if needed (see usage examples in header)
TRAIN_ARGS=(
    --model "$MODEL"
    --dataset "$DATASET"
    --data-root "$DATA_ROOT"
    --fold "$FOLD"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --accumulation-steps "$ACCUMULATION_STEPS"
    --output-dir "$OUTPUT_DIR"
    --log-dir "$LOG_DIR"
)

# Add --resume if specified
if [ -n "$RESUME" ]; then
    TRAIN_ARGS+=(--resume "$RESUME")
fi

# Run training
uv run python train_acdnet.py "${TRAIN_ARGS[@]}"

echo ""
echo "========================================"
echo "Training completed!"
echo "Results saved to: $OUTPUT_DIR/<model>_<dataset>_fold<N>_<timestamp>/"
echo "========================================"

#!/bin/bash

# ACDNet Training Script - AdamW Optimized Configuration
#
# "Ideal" training setup with:
#   - AdamW optimizer (lr=1e-3, weight_decay=1e-4)
#   - 4x data augmentation (default)
#   - MultiStepLR with steps every 500 epochs [500, 1000, 1500, 2000, 2500, 3000]
#   - Gamma=0.5 (gentler decay than SGD)
#   - 3500 epochs default (covers all milestones)
#   - Warmup: 10 epochs at 0.1x LR
#
# Basic Usage:
#   ./train_acdnet_adamw.sh                                    # Train with AdamW setup
#   ./train_acdnet_adamw.sh --model micro_acdnet               # Train Micro-ACDNet
#   ./train_acdnet_adamw.sh --dataset esc10                    # Train on ESC-10 subset
#   ./train_acdnet_adamw.sh --data-root /path/to/dataset       # Custom dataset path
#   ./train_acdnet_adamw.sh --fold 2                           # Use different fold
#   ./train_acdnet_adamw.sh --resume checkpoint.pt             # Resume training
#   ./train_acdnet_adamw.sh --epochs 5000                      # Train for 5000 epochs
#
# Override any parameter by passing additional args after --:
#
# Examples:
#   # Different learning rate
#   ./train_acdnet_adamw.sh -- --lr 5e-4
#
#   # Different milestones (every 1000 epochs)
#   ./train_acdnet_adamw.sh -- --lr-milestones 1000 2000 3000 4000
#
#   # Disable data expansion
#   ./train_acdnet_adamw.sh -- --augmentation-multiplier 1
#
#   # Use 8x data expansion
#   ./train_acdnet_adamw.sh -- --augmentation-multiplier 8

set -e

# Parse arguments
MODEL="acdnet"
DATASET="esc50"
DATA_ROOT="datasets/ESC-50-master"
FOLD=1
RESUME=""
LOG_DIR="./logs_acdnet"
EPOCHS=3500  # Default to cover all LR milestones

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
        --)
            # Pass remaining args to Python script
            shift
            break
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./train_acdnet_adamw.sh [--model acdnet|micro_acdnet] [--dataset esc50|esc10|urbansound8k] [--data-root PATH] [--fold N] [--resume PATH] [--log-dir PATH] [--epochs N] [-- EXTRA_PYTHON_ARGS...]"
            exit 1
            ;;
    esac
done

# Training configuration optimized for AdamW
BATCH_SIZE=64
NUM_WORKERS=16
ACCUMULATION_STEPS=1

# AdamW hyperparameters (can be overridden via -- args)
OPTIMIZER="adamw"
LR="1e-3"  # AdamW typically uses much lower LR than SGD
WEIGHT_DECAY="1e-4"  # Lower weight decay for AdamW

# LR Schedule: Steps every 500 epochs with gentler decay
LR_SCHEDULER="multistep"
LR_MILESTONES=(500 1000 1500 2000 2500 3000)
LR_GAMMA="0.5"  # Gentler decay (0.5 instead of 0.1)

# Warmup
WARMUP_EPOCHS=10
WARMUP_FACTOR="0.1"

# Augmentation (4x is default in train_acdnet.py)
AUGMENTATION_MULTIPLIER=4

# Checkpoints
SAVE_INTERVAL=250

# Output directory
OUTPUT_DIR="./checkpoints_acdnet"

echo "========================================"
echo "ACDNet Training - AdamW Optimized"
echo "========================================"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Fold: $FOLD"
echo "Epochs: $EPOCHS"
echo "Data root: $DATA_ROOT"
echo "Optimizer: $OPTIMIZER (LR=$LR, weight_decay=$WEIGHT_DECAY)"
echo "LR Schedule: ${LR_MILESTONES[@]} (gamma=$LR_GAMMA)"
echo "Augmentation multiplier: ${AUGMENTATION_MULTIPLIER}x"
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
    echo "Example: ./train_acdnet_adamw.sh --data-root datasets/ESC-50-master"
    exit 1
fi

# Build training arguments
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
    --optimizer "$OPTIMIZER"
    --lr "$LR"
    --weight-decay "$WEIGHT_DECAY"
    --lr-scheduler "$LR_SCHEDULER"
    --lr-milestones "${LR_MILESTONES[@]}"
    --lr-gamma "$LR_GAMMA"
    --warmup-epochs "$WARMUP_EPOCHS"
    --warmup-factor "$WARMUP_FACTOR"
    --augmentation-multiplier "$AUGMENTATION_MULTIPLIER"
    --save-interval "$SAVE_INTERVAL"
)

# Add --resume if specified
if [ -n "$RESUME" ]; then
    TRAIN_ARGS+=(--resume "$RESUME")
fi

# Add any extra arguments passed after --
if [ $# -gt 0 ]; then
    TRAIN_ARGS+=("$@")
fi

# Run training
uv run python train_acdnet.py "${TRAIN_ARGS[@]}"

echo ""
echo "========================================"
echo "Training completed!"
echo "Results saved to: $OUTPUT_DIR/<model>_<dataset>_fold<N>_<timestamp>/"
echo "========================================"

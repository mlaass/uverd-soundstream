#!/bin/bash

# Training script for SoundStream model
#
# Usage:
#   ./train.sh                              # Train with default config
#   ./train.sh --resume path.pt             # Resume training from checkpoint
#   ./train.sh --audio_dir path/to/audio    # Use custom dataset

# Parse arguments
AUDIO_DIR="datasets/ESC-50-master/audio"
RESUME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --audio_dir)
            AUDIO_DIR="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Training SOUNDSTREAM (full model)"
echo "Parameters: C=32, D=512, num_quantizers=8, codebook=1024, sample_rate=24000"
echo ""

# Build training arguments
TRAIN_ARGS=(
    --audio_dir "$AUDIO_DIR"
    --batch_size 8
    --audio_length 2.0
    --D 512
    --C 32
    --num_epochs 1000
    --num_quantizers 8
    --codebook_size 1024
    --g_lr 1e-4
    --d_lr 1e-4
    --disc_warmup_steps 5000
    --save_interval 5000
    --checkpoint_dir ./checkpoints
    --log_dir ./logs
)

# Add --resume if specified
if [ -n "$RESUME" ]; then
    TRAIN_ARGS+=(--resume "$RESUME")
    echo "Resuming from checkpoint: $RESUME"
    echo ""
fi

uv run python train.py "${TRAIN_ARGS[@]}"
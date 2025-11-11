#!/bin/bash

# Training script for TinyStream model (ESP32-S3 deployment)
#
# Usage:
#   ./train_tiny.sh                      # Train with 'tiny' config
#   ./train_tiny.sh --config ultra_tiny  # Train with 'ultra_tiny' config
#   ./train_tiny.sh --config small       # Train with 'small' config
#
# Note: Uses 2.0 second audio chunks for discriminator compatibility

# Parse config argument (default: tiny)
CONFIG="tiny"
AUDIO_DIR="datasets/ESC-50-master/audio"

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --audio_dir)
            AUDIO_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set parameters based on config
case $CONFIG in
    ultra_tiny)
        C=4
        D=64
        NUM_Q=2
        CODEBOOK=256
        SR=16000
        BATCH_SIZE=32
        echo "Training ULTRA_TINY config (ESP32-S3 with 512KB RAM)"
        echo "Expected model size: ~50KB (encoder only)"
        ;;
    small)
        C=12
        D=192
        NUM_Q=4
        CODEBOOK=1024
        SR=16000
        BATCH_SIZE=8
        echo "Training SMALL config (ESP32-S3 with PSRAM)"
        echo "Expected model size: ~800KB (encoder only)"
        ;;
    *)
        C=8
        D=128
        NUM_Q=4
        CODEBOOK=512
        SR=16000
        BATCH_SIZE=16
        echo "Training TINY config (ESP32-S3 with 512KB RAM)"
        echo "Expected model size: ~300KB (encoder only)"
        ;;
esac

echo "Parameters: C=$C, D=$D, num_quantizers=$NUM_Q, codebook=$CODEBOOK, sample_rate=$SR"
echo ""

# Run training
uv run python train.py \
    --model tinystream \
    --audio_dir "$AUDIO_DIR" \
    --batch_size $BATCH_SIZE \
    --audio_length 2.0 \
    --C $C \
    --D $D \
    --num_quantizers $NUM_Q \
    --codebook_size $CODEBOOK \
    --sample_rate $SR \
    --checkpoint_dir "./checkpoints_${CONFIG}" \
    --log_dir "./logs_${CONFIG}" \
    --g_lr 1e-4 \
    --d_lr 1e-4 \
    --disc_warmup_steps 5000 \
    --save_interval 5000 \
    --num_epochs 1000

#!/bin/bash

# Training script for TinyStream model (ESP32-S3 deployment)
#
# Usage:
#   ./train_tiny.sh                      # Train with 'tiny' config
#   ./train_tiny.sh --config ultra_tiny  # Train with 'ultra_tiny' config
#   ./train_tiny.sh --config small       # Train with 'small' config
#   ./train_tiny.sh --config medium      # Train with 'medium' config
#   ./train_tiny.sh --config full        # Train with 'full' config (1MB encoder)
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
        echo "Training ULTRA_TINY config (ESP32-S3 minimal, no PSRAM)"
        echo "Expected model size: ~0.02MB (encoder only)"
        ;;
    tiny)
        C=8
        D=128
        NUM_Q=4
        CODEBOOK=512
        SR=16000
        BATCH_SIZE=32
        echo "Training TINY config (ESP32-S3 minimal, no PSRAM)"
        echo "Expected model size: ~0.1MB (encoder only)"
        ;;
    small)
        C=12
        D=192
        NUM_Q=4
        CODEBOOK=1024
        SR=16000
        BATCH_SIZE=16
        echo "Training SMALL config (ESP32-S3 with limited PSRAM)"
        echo "Expected model size: ~0.15MB (encoder only)"
        ;;
    medium)
        C=16
        D=256
        NUM_Q=5
        CODEBOOK=1024
        SR=16000
        BATCH_SIZE=16
        echo "Training MEDIUM config (ESP32-S3 with PSRAM)"
        echo "Expected model size: ~0.3MB (encoder only)"
        ;;
    full)
        C=32
        D=256
        NUM_Q=6
        CODEBOOK=1024
        SR=16000
        BATCH_SIZE=16
        echo "Training FULL config (ESP32-S3 with PSRAM, max quality)"
        echo "Expected model size: ~1.0MB (encoder only)"
        ;;
    *)
        C=8
        D=128
        NUM_Q=4
        CODEBOOK=512
        SR=16000
        BATCH_SIZE=16
        echo "Training TINY config (default)"
        echo "Expected model size: ~0.1MB (encoder only)"
        ;;
esac

# Calculate bitrate
# Formula: bitrate (kbps) = num_quantizers * log2(codebook_size) * (sample_rate / downsampling_factor) / 1000
# TinyStream uses strides [4,4,4,4] = 256x downsampling
DOWNSAMPLING=256
EMBEDDING_RATE=$(echo "$SR / $DOWNSAMPLING" | bc -l)
LOG2_CODEBOOK=$(echo "l($CODEBOOK)/l(2)" | bc -l)
BITRATE=$(echo "$NUM_Q * $LOG2_CODEBOOK * $EMBEDDING_RATE / 1000" | bc -l)

echo "Parameters: C=$C, D=$D, num_quantizers=$NUM_Q, codebook=$CODEBOOK, sample_rate=$SR"
printf "Bitrate: %.2f kbps (%.2f bits/sample)\n" "$BITRATE" "$(echo "$BITRATE * 1000 / $SR" | bc -l)"
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
    --checkpoint_dir "./checkpoints_tiny" \
    --log_dir "./logs_tiny" \
    --g_lr 1e-4 \
    --d_lr 1e-4 \
    --disc_warmup_steps 5000 \
    --save_interval 5000 \
    --num_epochs 1000

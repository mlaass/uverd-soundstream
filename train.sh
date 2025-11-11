#!/bin/bash

uv run python train.py \
    --audio_dir datasets/ESC-50-master/audio \
    --batch_size 8 \
    --audio_length 2.0 \
    --D 512 \
    --C 32 \
    --num_epochs 1000 \
    --num_quantizers 8 \
    --codebook_size 1024 \
    --g_lr 1e-4 \
    --d_lr 1e-4 \
    --disc_warmup_steps 5000 \
    --save_interval 5000 \
    --checkpoint_dir ./checkpoints \
    --log_dir ./logs


# train.py \
#     --audio_dir datasets/ESC-50-master/audio \
#     --batch_size 6 \          # Adjust based on VRAM
#     --audio_length 2.0 \      # 2-second chunks
#     --C 32 \                  # Base channels
#     --D 512 \                 # Embedding dim
#     --num_quantizers 8 \      # For 3 kbps at 24kHz
#     --codebook_size 1024 \    # Codebook size per quantizer
#     --g_lr 1e-4 \
#     --d_lr 1e-4 \
#     --disc_warmup_steps 5000 \  # Train generator first
#     --save_interval 5000 \
#     --num_workers 4
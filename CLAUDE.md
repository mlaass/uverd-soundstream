# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains **PyTorch implementations of neural audio models optimized for constrained devices** like ESP32-S3, microcontrollers, and edge hardware. The focus is on efficient models that balance quality with deployment constraints (memory, compute, power).

### Available Models

1. **SoundStream** - Neural audio codec for high-quality compression (3-6 kbps)
2. **TinyStream** - Lightweight SoundStream variant for ESP32-S3 deployment (~0.3-1 MB)
3. **ACDNet** - Acoustic Classification Deep Network for environmental sound classification
4. **Micro-ACDNet** - Compressed ACDNet for MCU deployment (97% size reduction, 0.5 MB)

---

## 1. SoundStream (Neural Audio Codec)

### Overview
End-to-end neural audio codec optimized for natural sounds (birds, forest ambience). Achieves 3-6 kbps bitrate compression at 24kHz sample rate.

**Paper**: https://arxiv.org/abs/2107.03312

### Architecture (model.py)

- **Encoder**: Causal convolution blocks with residual units, 320x downsampling (24kHz → 75Hz)
  - Base channels `C` (default: 32), embedding dimension `D` (default: 512)
  - Strides: [2, 4, 5, 8]
  - Each block: 3 residual units with dilations [1, 3, 9]

- **Residual Vector Quantizer (RVQ)**: 8 layers, 1024 codebook size per layer
  - EMA updates for codebook learning
  - Quantizer dropout for bitrate scalability
  - Commitment loss weight: 0.25

- **Decoder**: Transposed convolutions, mirror architecture of encoder

- **Discriminators** (discriminator.py):
  - Multi-scale Wave Discriminator (3 scales)
  - STFT Discriminator (1024 window, 256 hop)

### Training SoundStream

**Quick start**:
```bash
./train.sh                                    # Train with defaults
./train.sh --resume checkpoint.pt             # Resume training
./train.sh --audio_dir path/to/dataset        # Custom dataset
```

**Full command**:
```bash
uv run python train.py \
    --audio_dir datasets/ESC-50-master/audio \
    --batch_size 8 \
    --audio_length 2.0 \
    --C 32 \
    --D 512 \
    --num_quantizers 8 \
    --codebook_size 1024 \
    --commitment_weight 0.25 \
    --g_lr 1e-4 \
    --d_lr 1e-4 \
    --disc_warmup_steps 5000 \
    --save_interval 5000 \
    --checkpoint_dir ./checkpoints \
    --log_dir ./logs
```

### Model Sizes

| Config | C | D | Params | VRAM | Quality |
|--------|---|---|--------|------|---------|
| Tiny   | 16 | 256 | 2.4M | 4GB | Good |
| Small  | 16 | 512 | 8.4M | 8GB | Better |
| Base   | 32 | 512 | 33M | 12GB | Best |

### Bitrate

Bitrate (kbps) = `(num_quantizers × log2(codebook_size) × embedding_rate) / 1000`

For 24kHz, 320x downsampling (75Hz):
- 8 quantizers: 6 kbps
- 4 quantizers: 3 kbps

### Training Timeline

- Steps 0-10k: Basic structure
- Steps 10k-50k: Quality improves
- Steps 50k-150k: Fine details
- Steps 150k+: Diminishing returns

Expect 3-5 days for 100k-150k steps on 12GB GPU (batch_size=8).

### Inference

```bash
# Reconstruct audio
uv run python infer.py \
    --checkpoint ./checkpoints/soundstream_step_100000.pt \
    --input test_audio.wav \
    --output reconstructed.wav

# Lower bitrate (use fewer quantizers)
uv run python infer.py \
    --checkpoint ./checkpoints/soundstream_step_100000.pt \
    --input test_audio.wav \
    --output reconstructed_3kbps.wav \
    --num_quantizers 4
```

---

## 2. TinyStream (ESP32-S3 Audio Codec)

### Overview
Ultra-lightweight SoundStream variant designed for ESP32-S3 deployment. Uses depthwise separable convolutions for efficiency.

### Architecture (model_tiny.py)

- **Encoder**: Depthwise separable convolutions, 256x downsampling (16kHz → 62.5Hz)
  - Strides: [4, 4, 4, 4]
  - 3 residual units per block with dilations [1, 3, 9]
  - Configurable base channels `C` and embedding dimension `D`

- **Residual VQ**: Simplified vector quantization
  - Fewer quantizers (2-8 depending on config)
  - Smaller codebooks (256-1024)

- **Decoder**: Only for training (not deployed to ESP32)

### Training TinyStream

**Quick start**:
```bash
./train_tiny.sh --config medium                    # Train medium config
./train_tiny.sh --config full                      # Train full (1MB encoder)
./train_tiny.sh --config medium --resume path.pt   # Resume training
```

**Available configs**:

| Config | C | D | Quantizers | Codebook | Encoder Size | Bitrate |
|--------|---|---|------------|----------|--------------|---------|
| ultra_tiny | 4 | 64 | 2 | 256 | ~0.02 MB | 1.00 kbps |
| tiny | 8 | 128 | 4 | 512 | ~0.10 MB | 2.25 kbps |
| small | 12 | 128 | 4 | 1024 | ~0.15 MB | 2.50 kbps |
| medium | 16 | 128 | 5 | 1024 | ~0.30 MB | 3.12 kbps |
| full | 16 | 128 | 6 | 1024 | ~1.00 MB | 3.75 kbps |

### ESP32 Deployment

Only encoder + quantizer are deployed to ESP32-S3:
- Encoder runs on-device to compress audio to codes
- Codes transmitted to server
- Server runs decoder to reconstruct audio

This minimizes memory/compute on the edge device.

---

## 3. ACDNet (Acoustic Classification)

### Overview
Acoustic Classification Deep Network for environmental sound classification using raw audio waveforms.

**Paper**: https://doi.org/10.1016/j.patcog.2022.109025

### Architecture (acdnet_model.py)

- **SFEB (Spectral Feature Extraction Block)**:
  - 1D convolutions at 10ms frame rate
  - Extracts low-level spectral features from raw audio
  - Input: raw waveform (20kHz, ~1.51s)

- **TFEB (Temporal Feature Extraction Block)**:
  - VGG-13-style 2D convolutions
  - Extracts high-level temporal features
  - Adaptive pooling for variable-length inputs

- **Specifications**:
  - Input: 30225 samples (~1.51s @ 20kHz)
  - Parameters: ~4.7M
  - Size: ~18 MB (FP32)
  - Designed for ESC-50/ESC-10 datasets

### Training ACDNet

```bash
# Quick start (use script)
./run_acdnet.sh acdnet esc50 /path/to/ESC-50-master

# Manual training
python train_acdnet.py \
    --model acdnet \
    --dataset esc50 \
    --data-root /path/to/ESC-50-master \
    --epochs 2000 \
    --batch-size 64 \
    --lr 0.1 \
    --weight-decay 5e-4
```

### Training Details

- 2000 epochs
- SGD with Nesterov momentum (0.9)
- LR: 0.1, decay at epochs [600, 1200, 1800] by 10x
- Warm-up: first 10 epochs with 0.1x LR
- Mixup augmentation (optional)
- Weight decay: 5e-4

---

## 4. Micro-ACDNet (MCU Classification)

### Overview
Compressed ACDNet for microcontroller deployment with minimal accuracy loss.

### Architecture (acdnet_micro.py)

**Compression results**:
- 97.22% size reduction (18 MB → 0.5 MB)
- 97.28% FLOP reduction (544M → 14.82M)
- Parameters: 4.74M → 0.131M

**Filter configuration**:
- SFEB: [7, 20] (vs [8, 64] in ACDNet)
- TFEB: [10, 14, 22, 31, 35, 41, 51, 67, 69] (vs larger filters in ACDNet)

### Training Micro-ACDNet

```bash
# Using run script
./run_acdnet.sh micro_acdnet esc50 /path/to/ESC-50-master

# Manual training
python train_acdnet.py \
    --model micro_acdnet \
    --dataset esc50 \
    --data-root /path/to/ESC-50-master \
    --epochs 2000 \
    --batch-size 64
```

### Model Sizes

| Model | Parameters | Size (FP32) | Size (INT8) | FLOPs |
|-------|------------|-------------|-------------|-------|
| ACDNet | 4.74M | 18.06 MB | 4.5 MB | 544M |
| Micro-ACDNet | 0.131M | 0.50 MB | 0.13 MB | 14.82M |

---

## Common Commands

### Dataset Management

```bash
# Download ESC-50
./download_datasets.sh --esc50

# Download LibriSpeech
./download_datasets.sh --librispeech

# Download all free datasets
./download_datasets.sh --all

# Download bird sounds (Xeno-canto)
python download_xenocanto.py --query "forest birds" --quality A --max 100
```

### Monitoring

```bash
# TensorBoard (SoundStream/TinyStream)
tensorboard --logdir ./logs

# Check training progress
cat checkpoints/*/config.json
```

### Evaluation

```bash
# Evaluate SoundStream/TinyStream
python evaluation/evaluate.py \
    --checkpoint checkpoints/tinystream_step_50000.pt \
    --audio_dir datasets/ESC-50-master/audio \
    --num_samples 10

# Test ACDNet
python test_acdnet.py \
    --checkpoint checkpoints/acdnet_best.pt \
    --data-root /path/to/ESC-50-master
```

---

## Key Hyperparameters

### SoundStream/TinyStream

- **commitment_weight** (default: 0.25): VQ commitment loss weight. Controls how strongly encoder matches quantized codes.
- **disc_warmup_steps** (default: 5000): Discriminator warmup. Generator trains alone for N steps before discriminator starts.
- **lambda_adv** (default: 1.0): Adversarial loss weight
- **lambda_feat** (default: 100.0): Feature matching loss weight
- **lambda_rec** (default: 1.0): Reconstruction loss weight
- **lr_decay** (default: 0.999996): Learning rate decay gamma
- **max_checkpoints** (default: 5): Keep only N most recent checkpoints

### ACDNet/Micro-ACDNet

- **epochs** (default: 2000): Total training epochs
- **lr** (default: 0.1): Initial learning rate (decays at [600, 1200, 1800])
- **weight_decay** (default: 5e-4): L2 regularization
- **momentum** (default: 0.9): Nesterov momentum
- **mixup** (default: True): Use mixup augmentation
- **dropout_rate** (default: 0.2): Dropout in TFEB

---

## Code Organization

### SoundStream/TinyStream
- **model.py**: SoundStream encoder/decoder/RVQ
- **model_tiny.py**: TinyStream with depthwise separable convs
- **discriminator.py**: Multi-scale wave + STFT discriminators
- **losses.py**: All loss functions
- **dataset.py**: Audio loading, preprocessing
- **train.py**: Main training loop
- **train.sh / train_tiny.sh**: Training scripts with presets
- **infer.py**: Inference script

### ACDNet
- **acdnet_model.py**: Full ACDNet implementation
- **acdnet_micro.py**: Compressed Micro-ACDNet
- **acdnet_dataset.py**: Dataset loading for classification
- **train_acdnet.py**: Training loop for classification
- **test_acdnet.py**: Testing/evaluation
- **run_acdnet.sh**: Training script with defaults

### Evaluation
- **evaluation/**: HTML report generation with metrics
  - metrics.py, visualizations.py, evaluate.py

---

## Important Implementation Details

### SoundStream/TinyStream

**Quantizer Dropout**: During training, 50% of batches randomly use fewer quantizers (1 to num_quantizers) for bitrate scalability.

**Causal Convolutions**: All encoder convolutions are causal (pad left only) for streaming inference.

**Audio Preprocessing**:
- Resamples to target sample rate (24kHz for SoundStream, 16kHz for TinyStream)
- Converts to mono
- Extracts random chunks
- Peak normalizes to [-1, 1]

**Checkpoint Structure**:
- model_state_dict, discriminator_state_dict
- Optimizer and scheduler states
- epoch, global_step, run_name
- Full config dict with metadata

### ACDNet/Micro-ACDNet

**Dynamic Pooling**: Pooling kernel sizes calculated dynamically based on input dimensions to ensure consistent output shapes.

**Mixup Augmentation**: Linear combinations of training examples with mixed labels for better generalization.

**Warm-up Schedule**: First 10 epochs use 0.1x learning rate to stabilize training.

---

## Troubleshooting

### SoundStream/TinyStream

**OOM (Out of Memory)**:
- Reduce `--batch_size` (try 6, 4, or 2)
- Reduce `--audio_length` (try 1.5 or 1.0 seconds)
- Use smaller model: `--C 16 --D 256`

**Training Instability**:
- Increase `--disc_warmup_steps` (try 10000)
- Lower learning rates: `--g_lr 5e-5 --d_lr 5e-5`
- Check for corrupted audio files

**Poor Quality**:
- Train longer (aim for 150k+ steps)
- Check discriminator is learning (d_loss > 0 in TensorBoard)
- Verify reconstruction loss decreases steadily

### ACDNet/Micro-ACDNet

**Low Accuracy**:
- Ensure correct dataset format (ESC-50 structure)
- Verify mixup augmentation is working
- Check learning rate schedule is correct
- Train for full 2000 epochs

**OOM**:
- Reduce batch_size (try 32 or 16)
- Use Micro-ACDNet instead of full ACDNet

---

## Development Notes

- Uses `uv` for dependency management (pyproject.toml)
- Python 3.12+ required
- CUDA recommended (CPU training is extremely slow)
- Don't modify pyproject.toml directly - use `uv add` for dependencies
- All models support resuming from checkpoints
- TensorBoard logging for SoundStream/TinyStream
- JSON config files saved with each training run

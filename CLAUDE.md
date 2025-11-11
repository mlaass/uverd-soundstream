# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch implementation of **SoundStream**, a neural audio codec optimized for natural sounds (birds, forest ambience). It's designed to train efficiently on 12GB VRAM and achieves 3-6 kbps bitrate compression at 24kHz sample rate.

## Core Architecture

### Model Components (model.py)

- **Encoder**: Causal convolution blocks with residual units that downsample audio 320x (24kHz → 75Hz embeddings)
  - Uses configurable base channels `C` (default 32) and embedding dimension `D` (default 512)
  - Strides: [2, 4, 5, 8] for 320x total downsampling
  - Each block has 3 residual units with dilations [1, 3, 9]

- **Residual Vector Quantizer (RVQ)**: 8 layers with 1024 codebook size per layer
  - Implements EMA updates for codebook learning
  - Supports quantizer dropout for bitrate scalability (train with 8, use 4 for lower bitrate)
  - Commitment loss weight: 0.25

- **Decoder**: Transposed convolution blocks that upsample back to original sample rate
  - Mirror architecture of encoder with same residual unit structure

### Discriminators (discriminator.py)

- **Multi-scale Wave Discriminator**: 3 scales (1x, 2x, 4x downsampling)
- **STFT Discriminator**: 1024 window, 256 hop length
- Both output intermediate features for feature matching loss

### Loss Functions (losses.py)

- **Adversarial Loss**: Hinge loss for GAN training
- **Feature Matching Loss**: L1 loss on discriminator intermediate features (weight: 100.0)
- **Multi-scale Spectral Loss**: Mel-spectrogram L1 + log L2 at scales [2048, 1024, 512, 256, 128]
- **Commitment Loss**: From vector quantization

## Commands

### Training

**Standard training** (12GB VRAM):
```bash
uv run python train.py \
    --audio_dir datasets/ESC-50-master/audio \
    --batch_size 8 \
    --audio_length 2.0 \
    --C 32 \
    --D 512 \
    --num_quantizers 8 \
    --codebook_size 1024 \
    --g_lr 1e-4 \
    --d_lr 1e-4 \
    --disc_warmup_steps 5000 \
    --save_interval 5000 \
    --checkpoint_dir ./checkpoints \
    --log_dir ./logs
```

**Quick training** (use provided script):
```bash
./train.sh
```

**Resume from checkpoint**:
```bash
uv run python train.py \
    --audio_dir datasets/ESC-50-master/audio \
    --resume ./checkpoints/soundstream_step_50000.pt \
    --batch_size 8
```

### Inference

**Reconstruct audio**:
```bash
uv run python infer.py \
    --checkpoint ./checkpoints/soundstream_step_100000.pt \
    --input test_audio.wav \
    --output reconstructed.wav
```

**Lower bitrate inference** (use fewer quantizers):
```bash
uv run python infer.py \
    --checkpoint ./checkpoints/soundstream_step_100000.pt \
    --input test_audio.wav \
    --output reconstructed_3kbps.wav \
    --num_quantizers 4  # Use 4 instead of 8 for 3 kbps
```

### Monitoring

**View training progress**:
```bash
tensorboard --logdir ./logs
```

### Dataset Management

**Download datasets**:
```bash
./download_datasets.sh --all          # All free datasets
./download_datasets.sh --esc50        # ESC-50 only
./download_datasets.sh --librispeech  # LibriSpeech only
./download_datasets.sh --list         # Show available datasets
```

**Download bird sounds** (Xeno-canto):
```bash
python download_xenocanto.py --query "forest birds" --quality A --max 100
```

## Training Configuration

### Model Sizes

| Config | C | D | Params | VRAM | Quality |
|--------|---|---|--------|------|---------|
| Tiny   | 16 | 256 | 2.4M | 4GB | Good |
| Small  | 16 | 512 | 8.4M | 8GB | Better |
| Base   | 32 | 512 | 33M | 12GB | Best |

### Bitrate Calculation

Bitrate (kbps) = `(num_quantizers × log2(codebook_size) × embedding_rate) / 1000`

For 24kHz audio with 320x downsampling (75Hz):
- 8 quantizers: 6 kbps
- 4 quantizers: 3 kbps

### Training Timeline

- Steps 0-10k: Basic structure learned
- Steps 10k-50k: Quality improves significantly
- Steps 50k-150k: Fine details emerge
- Steps 150k+: Diminishing returns

Expect ~3-5 days for 100k-150k steps on 12GB GPU with batch_size=8.

### Key Hyperparameters

- **disc_warmup_steps**: Start training discriminator after N steps (default: 5000) to stabilize early training
- **lambda_adv**: Adversarial loss weight (default: 1.0)
- **lambda_feat**: Feature matching loss weight (default: 100.0)
- **lambda_rec**: Reconstruction loss weight (default: 1.0)
- **lr_decay**: Learning rate decay gamma (default: 0.999996)
- **max_checkpoints**: Keep only N most recent checkpoints (default: 5)

## Important Implementation Details

### Quantizer Dropout

During training, 50% of batches randomly use fewer quantizers (1 to num_quantizers) for bitrate scalability. This allows the model to work at multiple bitrates without retraining.

### Causal Convolutions

All encoder convolutions are causal (pad left only) to enable streaming inference, even though current implementation processes full files.

### Audio Preprocessing

The dataset.py module automatically:
- Resamples to 24kHz
- Converts to mono
- Extracts random 2-second chunks
- Peak normalizes to [-1, 1]
- Pads/truncates to exact length

### Checkpoint Structure

Checkpoints contain:
- model_state_dict: Generator weights
- discriminator_state_dict: Discriminator weights
- g_optimizer_state_dict & d_optimizer_state_dict: Optimizer states
- g_scheduler_state_dict & d_scheduler_state_dict: LR scheduler states
- epoch & global_step: Training progress
- config: All hyperparameters

## Code Organization

- **model.py**: SoundStream encoder/decoder/RVQ implementation
- **discriminator.py**: Multi-scale wave + STFT discriminators
- **losses.py**: All loss functions (adversarial, feature matching, spectral reconstruction)
- **dataset.py**: Audio loading, preprocessing, augmentation utilities
- **train.py**: Training loop with SoundStreamTrainer class
- **infer.py**: Inference script for encode-decode reconstruction
- **train.sh**: Convenience script with default training config
- **download_datasets.sh**: Automated dataset download/extraction
- **download_xenocanto.py**: Python API for downloading bird sounds

## Troubleshooting

### OOM (Out of Memory)
- Reduce `--batch_size` (try 6, 4, or 2)
- Reduce `--audio_length` (try 1.5 or 1.0 seconds)
- Use smaller model: `--C 16 --D 256`

### Training Instability
- Increase `--disc_warmup_steps` (try 10000)
- Lower learning rates: `--g_lr 5e-5 --d_lr 5e-5`
- Check for corrupted audio files in dataset

### Poor Quality After Training
- Train longer (aim for 150k+ steps)
- Check discriminator is learning (d_loss should be > 0 in tensorboard)
- Ensure dataset diversity (mix different sound types)
- Verify reconstruction loss decreases steadily

### Audio Loading Issues
- If you see torchcodec errors, the dataset.py automatically falls back to soundfile backend
- Supported formats: .wav, .mp3, .flac, .ogg
- Dataset returns random noise if a file fails to load (check console for errors)

## Development Notes

- The project uses `uv` for dependency management (pyproject.toml)
- Python 3.12+ required
- CUDA recommended (CPU training is extremely slow)
- Gradient clipping at max_norm=1.0 for both generator and discriminator
- Exponential learning rate decay throughout training
- dont modify pyproject toml directly, just use uv add so we get the latest releases
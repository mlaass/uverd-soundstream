# SoundStream: Neural Audio Codec for Natural Sounds

Clean PyTorch implementation of [SoundStream](https://arxiv.org/abs/2107.03312) optimized for natural sounds (birds, forest ambience, etc.).

## Features

- ✅ Full encoder-decoder architecture with causal convolutions
- ✅ Residual Vector Quantization (RVQ) with EMA updates
- ✅ Multi-scale wave discriminator + STFT discriminator
- ✅ Quantizer dropout for bitrate scalability
- ✅ Multi-scale spectral reconstruction loss
- ✅ Optimized for 12GB VRAM training
- ✅ TensorBoard logging
- ✅ Checkpoint management

## Architecture

```
Input Audio (24kHz) 
    ↓
Encoder (Causal Conv Blocks)
    ↓
Embeddings @ 75Hz (320x downsampling)
    ↓
Residual Vector Quantizer (8 layers, 1024 codebook size)
    ↓
Quantized Embeddings
    ↓
Decoder (Transposed Conv Blocks)
    ↓
Reconstructed Audio
```

**Discriminators:**
- Multi-scale wave discriminator (3 scales)
- STFT discriminator (1024 window, 256 hop)

**Losses:**
- Adversarial loss (hinge)
- Feature matching loss (L1 on discriminator features)
- Multi-scale spectral loss (mel-spectrogram L1 + log L2)
- Commitment loss (from VQ)

## Installation

```bash
# Clone repository
git clone <repo-url>
cd soundstream

# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation

### Quick Start: Automated Download

Use the provided script to download datasets automatically:

```bash
# Download all free datasets (ESC-50, LibriSpeech)
./download_datasets.sh --all

# Download specific datasets
./download_datasets.sh --esc50
./download_datasets.sh --librispeech

# List available datasets
./download_datasets.sh --list
```

See [datasets/README.md](datasets/README.md) for detailed information about each dataset.

### Available Datasets

**General Audio:**
- **ESC-50**: Environmental sounds (600MB, 2000 files) - Auto download
- **FSD50K**: Freesound dataset (80GB, 51K files) - Manual download
- **UrbanSound8K**: Urban sounds (6GB, 8732 files) - Manual download  
- **LibriSpeech**: Speech corpus (350MB, 2600 files) - Auto download

**Birds & Forest (Natural Sounds):**
- **Xeno-canto**: 500k+ bird recordings (customizable) - Python API script
- **FSC22**: Forest sounds (5GB, 7K files) - Manual download
- **Bird Audio Detection**: Forest monitoring (20GB, 15K files) - Manual download

### Option: Your Own Audio
Place any `.wav`, `.mp3`, `.flac`, or `.ogg` files in a directory:
```
/path/to/audio/
├── bird1.wav
├── bird2.mp3
├── forest_ambience.flac
└── ...
```

## Training

### Quick Start (12GB VRAM)
```bash
# First, download a dataset
./download_datasets.sh --esc50

# Then train
python train.py \
    --audio_dir datasets/ESC-50-master/audio \
    --batch_size 8 \
    --audio_length 2.0 \
    --num_epochs 1000 \
    --checkpoint_dir ./checkpoints \
    --log_dir ./logs
```

### Recommended Settings for 12GB VRAM

```bash
python train.py \
    --audio_dir datasets/ESC-50-master/audio \
    --batch_size 6 \          # Adjust based on VRAM
    --audio_length 2.0 \      # 2-second chunks
    --C 32 \                  # Base channels
    --D 512 \                 # Embedding dim
    --num_quantizers 8 \      # For 3 kbps at 24kHz
    --codebook_size 1024 \    # Codebook size per quantizer
    --g_lr 1e-4 \
    --d_lr 1e-4 \
    --disc_warmup_steps 5000 \  # Train generator first
    --save_interval 5000 \
    --num_workers 4
```

### Training from Checkpoint
```bash
python train.py \
    --audio_dir datasets/ESC-50-master/audio \
    --resume ./checkpoints/soundstream_step_50000.pt \
    --batch_size 8
```

### Monitor Training
```bash
tensorboard --logdir ./logs
```

## Inference

Test reconstruction on a single audio file:

```python
import torch
from model import SoundStream
import torchaudio

# Load model
model = SoundStream(
    C=32,
    D=512,
    strides=[2, 4, 5, 8],
    num_quantizers=8,
    codebook_size=1024,
    sample_rate=24000
)

# Load checkpoint
checkpoint = torch.load('checkpoints/soundstream_step_100000.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load audio
audio, sr = torchaudio.load('test_audio.wav')
audio = torchaudio.transforms.Resample(sr, 24000)(audio)
audio = audio.mean(0, keepdim=True).unsqueeze(0)  # Convert to mono

# Encode and decode
with torch.no_grad():
    indices = model.encode(audio)
    reconstructed = model.decode(indices)

# Save
torchaudio.save('reconstructed.wav', reconstructed[0], 24000)
```

## Configuration

### Model Size vs Quality Tradeoff

| Config | Params | VRAM | Bitrate | Quality |
|--------|--------|------|---------|---------|
| Tiny   | 2.4M   | 4GB  | 3 kbps  | Good    |
| Small  | 8.4M   | 8GB  | 3 kbps  | Better  |
| Base   | 33M    | 12GB | 3 kbps  | Best    |

**Tiny (C=16, D=256):**
```bash
python train.py --C 16 --D 256 --batch_size 16
```

**Small (C=16, D=512):**
```bash
python train.py --C 16 --D 512 --batch_size 12
```

**Base (C=32, D=512):** - Recommended for 12GB
```bash
python train.py --C 32 --D 512 --batch_size 8
```

### Bitrate Control

Bitrate = `(num_quantizers * log2(codebook_size) * embedding_rate) / 1000`

For 24kHz audio with 320x downsampling (75Hz embeddings):
- 8 quantizers × 10 bits × 75 Hz = **6 kbps**
- 4 quantizers × 10 bits × 75 Hz = **3 kbps**

During inference, use fewer quantizers for lower bitrate:
```python
# 6 kbps (all 8 quantizers)
indices = model.encode(audio)  # Uses all quantizers

# 3 kbps (4 quantizers)
indices = indices[:, :4, :]  # Use first 4 quantizers only
reconstructed = model.decode(indices)
```

## Training Timeline

Based on experiments (approximate):
- **Steps 0-10k**: Model learns basic structure
- **Steps 10k-50k**: Quality improves significantly  
- **Steps 50k-150k**: Fine details emerge
- **Steps 150k+**: Diminishing returns

Expect ~100k-150k steps for good quality on 12GB GPU (~3-5 days with batch_size=8).

## Project Structure

```
soundstream/
├── model.py           # SoundStream encoder/decoder/RVQ
├── discriminator.py   # Multi-scale + STFT discriminators
├── losses.py          # Loss functions
├── dataset.py         # Audio data loading
├── train.py          # Training script
├── infer.py          # Inference script
├── requirements.txt   # Dependencies
└── README.md         # This file
```

## Tips for Training on Natural Sounds

1. **Data Diversity**: Mix different bird species and forest sounds
2. **Audio Length**: 2-3 second chunks work well
3. **Normalization**: Peak normalize to avoid clipping
4. **Discriminator Warmup**: Use `--disc_warmup_steps 5000` to stabilize early training
5. **Monitoring**: Watch reconstruction loss - should decrease steadily

## Troubleshooting

### OOM (Out of Memory)
- Reduce `--batch_size`
- Reduce `--audio_length`
- Use smaller model: `--C 16 --D 256`

### Training Instability
- Increase `--disc_warmup_steps`
- Check data quality (remove corrupted files)
- Reduce learning rates: `--g_lr 5e-5 --d_lr 5e-5`

### Poor Quality After Training
- Train longer (150k+ steps)
- Check discriminator is working (d_loss should be > 0)
- Ensure diverse training data

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{zeghidour2021soundstream,
  title={SoundStream: An End-to-End Neural Audio Codec},
  author={Zeghidour, Neil and Luebs, Alejandro and Omran, Ahmed and Skoglund, Jan and Tagliasacchi, Marco},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year={2021}
}
```

## License

MIT License - see LICENSE file

## Acknowledgments

- Original SoundStream paper by Google Research
- lucidrains/audiolm-pytorch for reference implementation
- Xeno-canto for bird sound data

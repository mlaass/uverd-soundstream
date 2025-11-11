# SoundStream Command Cheat Sheet

## Setup
```bash
# Install
pip install torch torchaudio tensorboard numpy tqdm

# Test installation
python test.py
```

## Training Commands

### Basic Training (12GB VRAM)
```bash
python train.py \
    --audio_dir /path/to/audio \
    --batch_size 8
```

### With All Options
```bash
python train.py \
    --audio_dir /path/to/bird_sounds \
    --batch_size 8 \
    --audio_length 2.0 \
    --num_epochs 1000 \
    --C 32 \
    --D 512 \
    --num_quantizers 8 \
    --codebook_size 1024 \
    --sample_rate 24000 \
    --g_lr 1e-4 \
    --d_lr 1e-4 \
    --disc_warmup_steps 5000 \
    --save_interval 5000 \
    --checkpoint_dir ./checkpoints \
    --log_dir ./logs \
    --num_workers 4
```

### Resume Training
```bash
python train.py \
    --audio_dir /path/to/audio \
    --resume ./checkpoints/soundstream_step_50000.pt
```

### Low VRAM (4-8GB)
```bash
python train.py \
    --audio_dir /path/to/audio \
    --C 16 \
    --D 256 \
    --batch_size 4 \
    --audio_length 1.5
```

## Monitoring

```bash
# Start TensorBoard
tensorboard --logdir ./logs

# Open browser to
http://localhost:6006
```

## Inference Commands

### Basic Reconstruction
```bash
python infer.py \
    --checkpoint ./checkpoints/soundstream_step_100000.pt \
    --input test_audio.wav \
    --output reconstructed.wav
```

### With Lower Bitrate (Use Fewer Quantizers)
```bash
# 3 kbps (4 quantizers instead of 8)
python infer.py \
    --checkpoint ./checkpoints/soundstream_step_100000.pt \
    --input test_audio.wav \
    --output reconstructed_3kbps.wav \
    --num_quantizers 4
```

### CPU Inference
```bash
python infer.py \
    --checkpoint ./checkpoints/soundstream_step_100000.pt \
    --input test_audio.wav \
    --output reconstructed.wav \
    --device cpu
```

## Testing Commands

### Run All Tests
```bash
python test.py
```

### Test Dataset Only
```bash
python dataset.py /path/to/audio
```

### Check GPU
```bash
nvidia-smi

# Watch GPU usage in real-time
watch -n 1 nvidia-smi
```

## Useful Utilities

### Count Audio Files
```bash
find /path/to/audio -type f \( -name "*.wav" -o -name "*.mp3" \) | wc -l
```

### Check Audio Properties
```bash
ffprobe -hide_banner test_audio.wav
```

### Convert Audio to 24kHz WAV
```bash
ffmpeg -i input.mp3 -ar 24000 -ac 1 output.wav
```

### Batch Convert Directory
```bash
for file in *.mp3; do
    ffmpeg -i "$file" -ar 24000 -ac 1 "${file%.mp3}.wav"
done
```

## Model Configurations

### Tiny (4GB VRAM)
```bash
--C 16 --D 256 --batch_size 4
# Params: ~2.4M
```

### Small (8GB VRAM)
```bash
--C 16 --D 512 --batch_size 6
# Params: ~8.4M
```

### Base (12GB VRAM) - Recommended
```bash
--C 32 --D 512 --batch_size 8
# Params: ~33M
```

### Large (16GB+ VRAM)
```bash
--C 32 --D 512 --batch_size 12
# Params: ~33M
```

## Checkpoint Management

### List Checkpoints
```bash
ls -lh checkpoints/
```

### Delete Old Checkpoints (Keep Last 5)
```bash
cd checkpoints
ls -t soundstream_step_*.pt | tail -n +6 | xargs rm
```

### Copy Best Checkpoint
```bash
cp checkpoints/soundstream_step_150000.pt soundstream_best.pt
```

## TensorBoard Tips

### View Specific Runs
```bash
tensorboard --logdir ./logs --reload_interval 30
```

### Export Scalars
```bash
tensorboard --logdir ./logs --export_scalars scalars.json
```

## Quick Debugging

### Check Model Size
```python
from model import SoundStream
model = SoundStream(C=32, D=512)
print(f"Params: {model.get_num_params():,}")
```

### Verify Data Loading
```python
from dataset import create_dataloader
loader = create_dataloader("/path/to/audio", batch_size=4)
batch = next(iter(loader))
print(f"Batch shape: {batch.shape}")
```

### Test Single Forward Pass
```python
import torch
from model import SoundStream

model = SoundStream()
x = torch.randn(1, 1, 24000)
recon, indices, loss = model(x)
print(f"Output shape: {recon.shape}")
```

## Common Issues & Fixes

### "No audio files found"
```bash
# Check directory structure
ls -R /path/to/audio | grep -E '\.(wav|mp3|flac|ogg)$'
```

### "CUDA out of memory"
```bash
# Reduce batch size
python train.py --audio_dir /path/to/audio --batch_size 4
```

### "Discriminator loss collapsed to 0"
```bash
# Increase warmup steps
python train.py --audio_dir /path/to/audio --disc_warmup_steps 10000
```

### "Training too slow"
```bash
# Check GPU utilization
nvidia-smi

# Reduce workers if CPU bottlenecked
python train.py --audio_dir /path/to/audio --num_workers 2
```

## Bitrate Reference

| Quantizers | Bits/frame | Bitrate @24kHz | Quality |
|------------|------------|----------------|---------|
| 2          | 20         | 1.5 kbps       | Low     |
| 4          | 40         | 3 kbps         | Good    |
| 6          | 60         | 4.5 kbps       | Better  |
| 8          | 80         | 6 kbps         | Best    |

## Training Timeline

| Steps | Time (12GB) | Quality      | Status |
|-------|-------------|--------------|---------|
| 10k   | ~3 hours    | Recognizable | Early   |
| 50k   | ~15 hours   | Good         | Usable  |
| 100k  | ~30 hours   | Very Good    | Production |
| 150k+ | ~5 days     | Excellent    | Best    |

## Directory Structure
```
soundstream/
├── model.py              # ← Core model
├── train.py             # ← Start here
├── infer.py             # ← Test here
├── checkpoints/         # ← Saves here
└── logs/                # ← Monitor here
```

## Quick Links

- Full docs: `README.md`
- Quick start: `QUICKSTART.md`
- Details: `IMPLEMENTATION.md`
- This file: `CHEATSHEET.md`

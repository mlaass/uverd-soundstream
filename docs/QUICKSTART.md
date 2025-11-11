# SoundStream Quick Start Guide

## Setup (5 minutes)

### 1. Install Dependencies
```bash
cd /home/claude/soundstream
pip install torch torchaudio tensorboard numpy tqdm
```

Or if using conda:
```bash
conda install pytorch torchaudio -c pytorch
pip install tensorboard tqdm
```

### 2. Verify Installation
```bash
python test.py
```

Expected output:
```
==================================================
SoundStream Implementation Test
==================================================
Testing imports...
✓ All dependencies installed
...
✓ All tests passed!
==================================================
```

## Dataset Setup

### Option 1: Use Xeno-canto Bird Sounds

1. Go to https://xeno-canto.org
2. Search for forest birds (e.g., "forest bird germany")
3. Download recordings (look for Creative Commons licensed ones)
4. Organize in a directory:
```
/path/to/bird_sounds/
├── Common_Blackbird_001.mp3
├── European_Robin_002.mp3
├── Great_Tit_003.wav
└── ...
```

### Option 2: Download ESC-50 Dataset
```bash
wget https://github.com/karolpiczak/ESC-50/archive/master.zip
unzip master.zip
mv ESC-50-master/audio ./esc50_audio
```

### Option 3: Record Your Own
Use any audio recorder to capture:
- Bird songs
- Forest ambience
- Natural water sounds
- Wind through trees

Tips:
- 24kHz or 48kHz sample rate recommended
- Mono or stereo (will be converted to mono)
- Any common format (.wav, .mp3, .flac, .ogg)

## Training

### Start Training (Recommended for 12GB VRAM)
```bash
python train.py \
    --audio_dir /path/to/your/audio \
    --batch_size 8 \
    --audio_length 2.0 \
    --num_epochs 1000 \
    --g_lr 1e-4 \
    --d_lr 1e-4 \
    --disc_warmup_steps 5000 \
    --save_interval 5000 \
    --checkpoint_dir ./checkpoints \
    --log_dir ./logs
```

### Monitor Training
```bash
# In another terminal
tensorboard --logdir ./logs
# Then open http://localhost:6006
```

Watch these metrics:
- `train/g_rec`: Should decrease (reconstruction loss)
- `train/d_loss`: Should be > 0 and relatively stable
- `train/g_total`: Should decrease overall

### Expected Training Time
- **10k steps**: ~2-3 hours, basic structure learned
- **50k steps**: ~12-15 hours, quality improves
- **100k steps**: ~24-30 hours, good quality
- **150k+ steps**: ~3-5 days, best quality

## Testing

After training, test reconstruction:

```bash
python infer.py \
    --checkpoint ./checkpoints/soundstream_step_100000.pt \
    --input /path/to/test_audio.wav \
    --output reconstructed.wav
```

Compare original vs reconstructed:
```bash
# Play both files
# Original
ffplay -autoexit /path/to/test_audio.wav

# Reconstructed  
ffplay -autoexit reconstructed.wav
```

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python train.py --audio_dir /path/to/audio --batch_size 4

# Or use smaller model
python train.py --audio_dir /path/to/audio --C 16 --D 256 --batch_size 12
```

### Training is Slow
- Check GPU utilization: `nvidia-smi`
- Reduce `num_workers` if CPU bottlenecked
- Ensure data is on fast storage (SSD)

### No Audio Files Found
```bash
# Check directory
ls /path/to/audio

# Check file extensions
python -c "from pathlib import Path; print(list(Path('/path/to/audio').rglob('*.wav'))[:5])"
```

### Poor Quality After Training
- Train longer (150k+ steps recommended)
- Check discriminator loss isn't collapsed (should be > 0)
- Try different learning rates
- Ensure diverse training data

## Model Configurations

### For Different VRAM:

**4GB VRAM:**
```bash
python train.py --C 16 --D 256 --batch_size 4 --audio_length 1.5
```

**8GB VRAM:**
```bash
python train.py --C 16 --D 512 --batch_size 6 --audio_length 2.0
```

**12GB VRAM (Recommended):**
```bash
python train.py --C 32 --D 512 --batch_size 8 --audio_length 2.0
```

**16GB+ VRAM:**
```bash
python train.py --C 32 --D 512 --batch_size 12 --audio_length 2.5
```

## Next Steps

1. **Experiment with bitrates**: Use fewer quantizers during inference for lower bitrate
2. **Fine-tune on specific sounds**: Continue training on specialized dataset
3. **Compare with baselines**: Test against MP3/Opus at similar bitrates
4. **Analyze codebook usage**: Check if all codebook entries are being used

## File Structure

```
soundstream/
├── model.py              # Core model (Encoder/Decoder/RVQ)
├── discriminator.py      # Multi-scale + STFT discriminators  
├── losses.py            # Loss functions
├── dataset.py           # Audio loading
├── train.py             # Training script
├── infer.py             # Inference script
├── test.py              # Verification script
├── requirements.txt     # Dependencies
├── README.md           # Full documentation
├── QUICKSTART.md       # This file
├── checkpoints/        # Model checkpoints (created during training)
└── logs/               # TensorBoard logs (created during training)
```

## Questions?

Check the full README.md for:
- Detailed architecture explanation
- Loss function details
- Advanced configuration options
- Dataset recommendations
- Training tips for natural sounds

Good luck with your training! 🎵🦜🌲

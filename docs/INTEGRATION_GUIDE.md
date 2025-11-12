# Integrating ACDNet into Your SoundStream Project

## Overview

This guide shows how to integrate the ACDNet implementation into your existing `uverd-soundstream` project structure.

## Project Structure

Your current structure:
```
uverd-soundstream/
├── model.py              # SoundStream encoder/decoder
├── model_tiny.py         # TinyStream variant
├── train.py              # Training script
├── dataset.py            # Dataset loader
├── discriminator.py      # Discriminator for adversarial training
└── ...
```

After integration:
```
uverd-soundstream/
├── models/
│   ├── __init__.py
│   ├── soundstream.py       # Your SoundStream model (renamed from model.py)
│   ├── tinystream.py        # Your TinyStream model (renamed from model_tiny.py)
│   ├── acdnet.py            # ACDNet model
│   └── acdnet_micro.py      # Micro-ACDNet model
├── datasets/
│   ├── __init__.py
│   ├── audio_dataset.py     # Your existing dataset
│   └── esc_dataset.py       # Environmental sound datasets
├── train_soundstream.py     # Your existing training
├── train_acdnet.py          # ACDNet training
└── ...
```

## Step-by-Step Integration

### Step 1: Organize Model Files

```bash
# Create models directory
mkdir -p models
touch models/__init__.py

# Move existing models
cp model.py models/soundstream.py
cp model_tiny.py models/tinystream.py

# Add ACDNet models
cp acdnet_model.py models/acdnet.py
cp acdnet_micro.py models/acdnet_micro.py
```

Update `models/__init__.py`:
```python
from .soundstream import SoundStream
from .tinystream import TinyStream
from .acdnet import ACDNet, create_acdnet
from .acdnet_micro import MicroACDNet, create_micro_acdnet

__all__ = [
    'SoundStream', 'TinyStream',
    'ACDNet', 'MicroACDNet',
    'create_acdnet', 'create_micro_acdnet'
]
```

### Step 2: Organize Dataset Files

```bash
# Create datasets directory
mkdir -p datasets
touch datasets/__init__.py

# Move/copy existing dataset
cp dataset.py datasets/audio_dataset.py

# Add environmental sound datasets
cp acdnet_dataset.py datasets/esc_dataset.py
```

Update `datasets/__init__.py`:
```python
from .audio_dataset import AudioDataset  # Your existing dataset
from .esc_dataset import (
    ESC50Dataset,
    UrbanSound8KDataset,
    create_dataloaders
)

__all__ = [
    'AudioDataset',
    'ESC50Dataset',
    'UrbanSound8KDataset',
    'create_dataloaders'
]
```

### Step 3: Update Training Scripts

Keep both training scripts separate:
- `train_soundstream.py` - For codec/reconstruction tasks
- `train_acdnet.py` - For classification tasks

You could also create a unified training interface:

```python
# train.py
import argparse
from train_soundstream import train_soundstream
from train_acdnet import train_acdnet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['soundstream', 'acdnet'])
    parser.add_argument('--config', type=str, required=True)
    args, unknown = parser.parse_known_args()
    
    if args.task == 'soundstream':
        train_soundstream()
    elif args.task == 'acdnet':
        train_acdnet()

if __name__ == '__main__':
    main()
```

### Step 4: Shared Components

Create a `utils/` directory for shared utilities:

```bash
mkdir -p utils
touch utils/__init__.py
```

Shared components could include:
- Audio preprocessing utilities
- Training utilities (checkpointing, logging)
- Evaluation metrics
- Visualization tools

Example `utils/audio.py`:
```python
import torch
import torchaudio

def load_audio(path, target_sr=24000, target_length=None):
    """Load and preprocess audio file"""
    waveform, sr = torchaudio.load(path)
    
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    
    # Pad or crop
    if target_length is not None:
        current_length = waveform.shape[1]
        if current_length < target_length:
            waveform = torch.nn.functional.pad(
                waveform, (0, target_length - current_length)
            )
        elif current_length > target_length:
            waveform = waveform[:, :target_length]
    
    return waveform

def normalize_audio(waveform, method='peak'):
    """Normalize audio waveform"""
    if method == 'peak':
        peak = torch.max(torch.abs(waveform))
        if peak > 0:
            waveform = waveform / peak
    elif method == 'rms':
        rms = torch.sqrt(torch.mean(waveform ** 2))
        if rms > 0:
            waveform = waveform / rms
    return waveform
```

### Step 5: Configuration Management

Create a `configs/` directory for experiment configurations:

```bash
mkdir -p configs
```

Example configurations:

`configs/acdnet_esc50.yaml`:
```yaml
model:
  name: acdnet
  num_classes: 50
  input_length: 30225
  sample_rate: 20000
  base_filters: 8

dataset:
  name: esc50
  root: /path/to/ESC-50
  target_sr: 20000
  target_length: 30225
  augment: true
  mixup: true

training:
  epochs: 2000
  batch_size: 64
  lr: 0.1
  weight_decay: 5e-4
  momentum: 0.9
  num_workers: 4

output:
  dir: ./checkpoints
  save_freq: 100
  tensorboard: true
```

`configs/micro_acdnet_esc50.yaml`:
```yaml
model:
  name: micro_acdnet
  num_classes: 50
  input_length: 30225
  sample_rate: 20000

dataset:
  name: esc50
  root: /path/to/ESC-50
  target_sr: 20000
  target_length: 30225
  augment: true
  mixup: true

training:
  epochs: 2000
  batch_size: 64
  lr: 0.1
  weight_decay: 5e-4
  momentum: 0.9
  num_workers: 4

output:
  dir: ./checkpoints
  save_freq: 100
  tensorboard: true
```

### Step 6: Update Requirements

Update your `pyproject.toml` to include ACDNet dependencies:

```toml
[project]
name = "uverd-soundstream"
version = "0.1.0"
dependencies = [
    "torch>=2.0.0",
    "torchaudio>=2.0.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",  # For ESC dataset metadata
    "tqdm>=4.65.0",
    "tensorboard>=2.13.0",
    "einops>=0.6.1",
    "PyYAML>=6.0",  # For config files
]

[project.optional-dependencies]
dev = [
    "pytest>=7.3.1",
    "black>=23.3.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
]
```

### Step 7: Create Evaluation Scripts

Create `evaluate_acdnet.py`:
```python
"""
Evaluate ACDNet models on test sets
"""
import torch
from pathlib import Path
from models import create_acdnet, create_micro_acdnet
from datasets import create_dataloaders

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for audio, labels in test_loader:
            audio, labels = audio.to(device), labels.to(device)
            outputs = model(audio)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def main():
    # Load checkpoint
    checkpoint_path = Path("checkpoints/checkpoint_best.pt")
    checkpoint = torch.load(checkpoint_path)
    
    # Create model
    model = create_acdnet(num_classes=50)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create test loader
    _, test_loader = create_dataloaders(
        dataset_name='esc50',
        root='/path/to/ESC-50',
        fold=1,
        batch_size=32
    )
    
    # Evaluate
    accuracy = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    main()
```

### Step 8: Testing Integration

Create tests to ensure integration works:

```bash
mkdir -p tests
```

`tests/test_models.py`:
```python
import torch
import pytest
from models import (
    create_acdnet,
    create_micro_acdnet,
    SoundStream,
    TinyStream
)

def test_acdnet_forward():
    model = create_acdnet(num_classes=50)
    x = torch.randn(2, 1, 30225)
    y = model(x)
    assert y.shape == (2, 50)

def test_micro_acdnet_forward():
    model = create_micro_acdnet(num_classes=50)
    x = torch.randn(2, 1, 30225)
    y = model(x)
    assert y.shape == (2, 50)

def test_soundstream_forward():
    # Your existing SoundStream test
    pass

def test_tinystream_forward():
    # Your existing TinyStream test
    pass
```

## Final Project Structure

```
uverd-soundstream/
├── models/
│   ├── __init__.py
│   ├── soundstream.py
│   ├── tinystream.py
│   ├── acdnet.py
│   └── acdnet_micro.py
├── datasets/
│   ├── __init__.py
│   ├── audio_dataset.py
│   └── esc_dataset.py
├── utils/
│   ├── __init__.py
│   ├── audio.py
│   ├── training.py
│   └── evaluation.py
├── configs/
│   ├── acdnet_esc50.yaml
│   └── micro_acdnet_esc50.yaml
├── tests/
│   ├── test_models.py
│   └── test_datasets.py
├── evaluation/
│   ├── evaluate.py
│   ├── metrics.py
│   └── visualizations.py
├── train_soundstream.py
├── train_acdnet.py
├── evaluate_acdnet.py
├── discriminator.py
├── losses.py
├── pyproject.toml
├── README.md
└── ACDNET_README.md
```

## Usage Examples

### Train ACDNet:
```bash
python train_acdnet.py \
    --model acdnet \
    --dataset esc50 \
    --data-root /path/to/ESC-50 \
    --fold 1
```

### Train Micro-ACDNet:
```bash
python train_acdnet.py \
    --model micro \
    --dataset esc50 \
    --data-root /path/to/ESC-50 \
    --fold 1
```

### Train SoundStream (your existing):
```bash
python train_soundstream.py
```

### Evaluate:
```bash
python evaluate_acdnet.py \
    --checkpoint checkpoints/checkpoint_best.pt \
    --dataset esc50 \
    --data-root /path/to/ESC-50
```

## Benefits of This Structure

1. **Separation of Concerns**: Different models for different tasks
2. **Reusable Components**: Shared utilities for audio processing
3. **Easy Experimentation**: Clear config files
4. **Maintainable**: Organized code structure
5. **Extensible**: Easy to add new models or datasets

## Next Steps

1. Move files into the new structure
2. Update import statements
3. Test that everything works
4. Add documentation for each component
5. Create example notebooks for different use cases

# ACDNet PyTorch Implementation Summary

## Overview

I've created a complete PyTorch implementation of ACDNet and Micro-ACDNet from the paper "Environmental Sound Classification on the Edge" (Pattern Recognition, 2023).

## Files Created

### Core Models

1. **acdnet_model.py** - Full ACDNet implementation
   - `SpectralFeatureExtractionBlock (SFEB)`: Extracts spectral features from raw audio
   - `TemporalFeatureExtractionBlock (TFEB)`: Extracts temporal features with VGG-style architecture
   - `ACDNet`: Main model combining SFEB and TFEB
   - Dynamic pooling calculations following paper's formulas
   - Flexible architecture adapting to different audio lengths

2. **acdnet_micro.py** - Compressed Micro-ACDNet implementation
   - Follows exact filter configuration from Table 11 in the paper
   - SFEB: [7, 20] filters (vs [8, 64] in ACDNet)
   - TFEB: [10, 14, 22, 31, 35, 41, 51, 67, 69, 48] filters
   - Total: 415 filters (vs 2074 in ACDNet)
   - Target: 0.131M parameters, 0.50MB model size

### Data & Training

3. **acdnet_dataset.py** - Dataset loaders
   - `EnvironmentalSoundDataset`: Base class for audio datasets
   - `ESC50Dataset`: ESC-50 and ESC-10 dataset support
   - `UrbanSound8KDataset`: UrbanSound8K dataset support
   - Audio preprocessing (resampling, padding/cropping, normalization)
   - Data augmentation (time stretching, pitch shifting, noise)
   - Mixup augmentation following EnvNet-v2 approach

4. **train_acdnet.py** - Training script
   - Implements exact training procedure from paper:
     * 2000 epochs
     * Learning rate: 0.1 with warm-up and decay
     * SGD with Nesterov momentum (0.9)
     * Weight decay: 5e-4
     * Batch size: 64
   - KL Divergence Loss for mixup training
   - TensorBoard logging
   - Checkpoint saving
   - Cross-validation support

### Documentation & Testing

5. **ACDNET_README.md** - Comprehensive documentation
   - Architecture explanation
   - Installation instructions
   - Dataset preparation guides
   - Training examples
   - Performance benchmarks
   - Edge deployment information

6. **test_acdnet.py** - Verification tests
   - Forward pass tests
   - Parameter counting
   - FLOP estimation
   - Filter configuration validation
   - Compression ratio verification

7. **requirements_acdnet.txt** - Dependencies

## Key Implementation Details

### Architecture Faithfulness

The implementation closely follows the paper:

**SFEB (Spectral Feature Extraction Block):**
- Conv1: kernel=(1,9), stride=(1,2), filters=x
- Conv2: kernel=(1,5), stride=(1,2), filters=x*2^3
- MaxPool: dynamic size for ~10ms frame rate
- Axis swap to prepare for 2D convolutions

**TFEB (Temporal Feature Extraction Block):**
- 12 convolution layers in VGG-13 style
- Dynamic pooling based on spatial dimensions
- Dropout (0.2)
- 1x1 convolution to num_classes
- Adaptive dense layer for compression-friendly design

**Micro-ACDNet:**
- Exact filter counts from Table 11
- Fixed pooling sizes matching paper's architecture
- Maintains the same overall structure as ACDNet

### Training Procedure

Matches paper exactly:
- Warm-up phase: First 10 epochs at 0.01 learning rate
- Learning rate schedule: Decay by 10x at epochs [600, 1200, 1800]
- Mixup data augmentation with gain-based ratio calculation
- KL Divergence Loss for soft labels

### Data Augmentation

Following EnvNet-v2 approach:
```python
# Mixup formula from paper
p = 1 / (1 + 10^((g1-g2)/20 * (1-r)/r))
mixed_audio = (p*s1 + (1-p)*s2) / sqrt(p^2 + (1-p)^2)
```

## Expected Performance

### ACDNet (Full Model)
- ESC-10: 96.65% (SOTA for raw audio)
- ESC-50: 87.10% (SOTA for raw audio)
- Parameters: 4.74M
- Size: 18.06 MB
- FLOPs: 544M

### Micro-ACDNet (Compressed)
- ESC-10: 96.25%
- ESC-50: 83.65% (above human: 81.30%)
- Parameters: 0.131M
- Size: 0.50 MB (97.22% reduction)
- FLOPs: 14.82M (97.28% reduction)

## Usage Examples

### Train ACDNet on ESC-50
```bash
python train_acdnet.py \
    --model acdnet \
    --dataset esc50 \
    --data-root /path/to/ESC-50 \
    --fold 1 \
    --epochs 2000 \
    --batch-size 64
```

### Train Micro-ACDNet
```bash
python train_acdnet.py \
    --model micro \
    --dataset esc50 \
    --data-root /path/to/ESC-50 \
    --fold 1 \
    --epochs 2000
```

### 5-Fold Cross-Validation
```bash
for fold in {1..5}; do
    python train_acdnet.py \
        --dataset esc50 \
        --data-root /path/to/ESC-50 \
        --fold $fold \
        --output-dir ./cv_fold_$fold
done
```

## Model Architecture Diagrams

### ACDNet Flow
```
Raw Audio (1, 30225) @ 20kHz
    ↓
SFEB (Spectral Features)
    Conv1 (1,9) stride=2 → (x, 15109)
    Conv2 (1,5) stride=2 → (x*8, 7553)
    MaxPool (50) → (x*8, 151)
    ↓
Axis Swap: (x*8, 151) → (1, x*8, 151)
    ↓
TFEB (Temporal Features)
    Conv3 (3,3) + Pool → (x*4, 16, 75)
    [Conv4-5] + Pool → (x*8, 8, 37)
    [Conv6-7] + Pool → (x*16, 4, 18)
    [Conv8-9] + Pool → (x*32, 2, 9)
    [Conv10-11] + Pool → (x*64, 1, 4)
    ↓
Conv12 (1,1) → (num_classes, 1, 4)
AvgPool → (num_classes, 1, 1)
Dense → (num_classes)
```

### Micro-ACDNet Specifics
```
SFEB: [7, 20] filters
TFEB: [10, 14, 22, 31, 35, 41, 51, 67, 69, 48] filters
Total: 415 filters (only 20% of ACDNet's 2074)
```

## Implementation Highlights

1. **Flexible Architecture**: Automatically adapts pooling sizes to different audio lengths and sample rates

2. **Compression-Friendly Design**: Dynamic dense layer ensures the model can be compressed without breaking

3. **Paper-Accurate Training**: Exact reproduction of training procedure including warm-up, LR schedule, and mixup

4. **Comprehensive Dataset Support**: Handles ESC-50, ESC-10, and UrbanSound8K with proper cross-validation splits

5. **Production-Ready**: Includes checkpointing, TensorBoard logging, and evaluation metrics

## Next Steps

To use this implementation:

1. Install dependencies: `pip install -r requirements_acdnet.txt`
2. Download datasets (ESC-50 or UrbanSound8K)
3. Run training: `python train_acdnet.py --dataset esc50 --data-root /path/to/data`
4. Monitor progress: `tensorboard --logdir checkpoints/*/logs`
5. Evaluate best model on test set

For edge deployment:
1. Train Micro-ACDNet
2. Apply 8-bit quantization
3. Export to TensorFlow Lite or ONNX
4. Deploy on MCU (e.g., Sony Spresense, Nordic nRF52840)

## References

Paper: Mohaimenuzzaman et al., "Environmental Sound Classification on the Edge: A Pipeline for Deep Acoustic Networks on Extremely Resource-Constrained Devices", Pattern Recognition, 2023.
DOI: 10.1016/j.patcog.2022.109025

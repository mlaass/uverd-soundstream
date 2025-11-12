# ACDNet: Environmental Sound Classification on the Edge

PyTorch implementation of **ACDNet** (Acoustic Classification Deep Network) and **Micro-ACDNet** from the paper:

> **"Environmental Sound Classification on the Edge: A Pipeline for Deep Acoustic Networks on Extremely Resource-Constrained Devices"**  
> Md Mohaimenuzzaman, Christoph Bergmeir, Ian West, Bernd Meyer  
> Pattern Recognition, Volume 133, 2023  
> https://doi.org/10.1016/j.patcog.2022.109025

## Overview

This implementation provides:

- **ACDNet**: State-of-the-art model for raw audio classification
  - **ESC-10**: 96.65% accuracy (SOTA for raw audio)
  - **ESC-50**: 87.10% accuracy (SOTA for raw audio)
  - **UrbanSound8K**: 84.45% accuracy
  - **AudioEvent**: 92.57% accuracy

- **Micro-ACDNet**: Compressed model for edge deployment
  - **97.22% size reduction** (18.06MB → 0.50MB)
  - **97.28% FLOP reduction** (544M → 14.82M FLOPs)
  - **ESC-50**: 83.65% accuracy (above human performance of 81.30%)

## Architecture

### ACDNet

ACDNet consists of two main blocks:

1. **SFEB** (Spectral Feature Extraction Block)
   - Extracts low-level spectral features from raw audio
   - 1D convolutions with stride 2
   - Frame rate of ~10ms

2. **TFEB** (Temporal Feature Extraction Block)
   - Extracts high-level temporal features
   - VGG-13 style architecture with 2D convolutions
   - Adaptive pooling based on input dimensions

Key features:
- Direct raw audio input (no hand-crafted features)
- Single-channel input (no multi-stream)
- Flexible architecture that adapts to different audio lengths
- Compression-friendly design with dynamic dense layer

### Micro-ACDNet

Compressed version following Table 11 from the paper:

```
SFEB filters: [7, 20]
TFEB filters: [10, 14, 22, 31, 35, 41, 51, 67, 69, 48]
Total filters: 415 (vs 2074 in ACDNet)
```

## Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/acdnet-pytorch
cd acdnet-pytorch

# Install dependencies
pip install torch torchaudio pandas numpy tqdm tensorboard
```

## Dataset Preparation

### ESC-50 / ESC-10

Download from: https://github.com/karolpiczak/ESC-50

```bash
wget https://github.com/karoldvl/ESC-50/archive/master.zip
unzip master.zip
```

Expected structure:
```
ESC-50/
├── audio/
│   ├── 1-100032-A-0.wav
│   ├── ...
└── meta/
    └── esc50.csv
```

### UrbanSound8K

Download from: https://urbansounddataset.weebly.com/urbansound8k.html

Expected structure:
```
UrbanSound8K/
├── audio/
│   ├── fold1/
│   ├── fold2/
│   ├── ...
│   └── fold10/
└── metadata/
    └── UrbanSound8K.csv
```

## Usage

### Training ACDNet

Train ACDNet on ESC-50 dataset:

```bash
python train_acdnet.py \
    --model acdnet \
    --dataset esc50 \
    --data-root /path/to/ESC-50 \
    --fold 1 \
    --epochs 2000 \
    --batch-size 64 \
    --output-dir ./checkpoints
```

### Training Micro-ACDNet

Train Micro-ACDNet (compressed model):

```bash
python train_acdnet.py \
    --model micro \
    --dataset esc50 \
    --data-root /path/to/ESC-50 \
    --fold 1 \
    --epochs 2000 \
    --batch-size 64 \
    --output-dir ./checkpoints
```

### 5-Fold Cross-Validation

Run cross-validation for ESC-50:

```bash
for fold in {1..5}; do
    python train_acdnet.py \
        --model acdnet \
        --dataset esc50 \
        --data-root /path/to/ESC-50 \
        --fold $fold \
        --output-dir ./checkpoints/fold_$fold
done
```

### Training Options

```
Model:
  --model {acdnet,micro}     Model architecture (default: acdnet)

Dataset:
  --dataset {esc50,esc10,urbansound8k}
  --data-root PATH           Path to dataset root directory
  --fold INT                 Fold number for cross-validation (default: 1)

Training:
  --epochs INT               Number of epochs (default: 2000)
  --batch-size INT           Batch size (default: 64)
  --lr FLOAT                 Initial learning rate (default: 0.1)
  --weight-decay FLOAT       Weight decay (default: 5e-4)
  --momentum FLOAT           SGD momentum (default: 0.9)

Augmentation:
  --no-augment               Disable data augmentation
  --no-mixup                 Disable mixup augmentation

Audio:
  --target-length INT        Target audio length in samples (default: 30225)
  --sample-rate INT          Target sample rate (default: 20000)

Output:
  --output-dir PATH          Output directory (default: ./checkpoints)
  --device {cuda,cpu}        Device to use (default: cuda)
  --num-workers INT          Number of dataloader workers (default: 4)
```

## Training Details

Following the paper's training procedure:

- **Optimizer**: SGD with Nesterov momentum (0.9)
- **Learning rate**: 0.1 (initial)
  - Warm-up: First 10 epochs with 0.01
  - Decay: By factor of 10 at epochs [600, 1200, 1800]
- **Weight decay**: 5e-4
- **Batch size**: 64
- **Epochs**: 2000
- **Loss function**: 
  - KL Divergence Loss (with mixup)
  - Cross Entropy (without mixup)

### Data Augmentation

Following EnvNet-v2 approach:

1. **Training Samples**: 
   - Mix two random samples from different classes
   - Pad with T/2 zeros on each side
   - Randomly crop T-length sections
   - Calculate mixing ratio based on signal gain

2. **Testing**: 
   - Pad with T/2 zeros
   - Extract 10 windows at regular intervals
   - Average predictions

## Model Performance

### ACDNet (Full Model)

| Dataset | Accuracy | Parameters | Size (FP32) | FLOPs |
|---------|----------|------------|-------------|-------|
| ESC-10  | 96.65%   | 4.74M      | 18.06 MB    | 544M  |
| ESC-50  | 87.10%   | 4.74M      | 18.06 MB    | 544M  |
| US8K    | 84.45%   | 4.74M      | 18.06 MB    | 544M  |
| AE      | 92.57%   | 4.74M      | 18.06 MB    | 544M  |

### Micro-ACDNet (Compressed)

| Dataset | Accuracy | Parameters | Size (FP32) | FLOPs | Reduction |
|---------|----------|------------|-------------|-------|-----------|
| ESC-10  | 96.25%   | 0.131M     | 0.50 MB     | 14.82M| 97.22%    |
| ESC-50  | 83.65%   | 0.131M     | 0.50 MB     | 14.82M| 97.22%    |
| US8K    | 78.28%   | 0.131M     | 0.50 MB     | 14.82M| 97.22%    |
| AE      | 89.69%   | 0.131M     | 0.50 MB     | 14.82M| 97.22%    |

Note: Micro-ACDNet achieves **above human performance** (81.30%) on ESC-50 despite 97% compression!

## Edge Deployment

Micro-ACDNet is designed for deployment on MCUs with:
- **< 512KB SRAM** for inference
- **< 1MB Flash** for model storage
- **No GPU required**

Target devices:
- Sony Spresense
- Nordic nRF52840
- STM32F4
- Other ARM Cortex-M4F based MCUs

### Quantization

8-bit quantization reduces model size to **~157KB**:

```python
# Export model for quantization
import torch

model = create_micro_acdnet(num_classes=50)
model.load_state_dict(torch.load('checkpoint_best.pt')['model_state_dict'])
model.eval()

# Quantize to INT8
model_int8 = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save quantized model
torch.save(model_int8.state_dict(), 'model_int8.pt')
```

## Code Structure

```
.
├── acdnet_model.py          # ACDNet full model implementation
├── acdnet_micro.py          # Micro-ACDNet compressed model
├── acdnet_dataset.py        # Dataset loaders (ESC-50, US8K)
├── train_acdnet.py          # Training script
├── README.md                # This file
└── requirements.txt         # Python dependencies
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{mohaimenuzzaman2023environmental,
  title={Environmental sound classification on the edge: A pipeline for deep acoustic networks on extremely resource-constrained devices},
  author={Mohaimenuzzaman, Md and Bergmeir, Christoph and West, Ian and Meyer, Bernd},
  journal={Pattern Recognition},
  volume={133},
  pages={109025},
  year={2023},
  publisher={Elsevier}
}
```

## License

This implementation is provided for research purposes. Please refer to the original paper for details.

## Acknowledgments

- Original paper authors: Md Mohaimenuzzaman, Christoph Bergmeir, Ian West, Bernd Meyer
- Monash University, Department of Data Science and AI
- Datasets: ESC-50, UrbanSound8K, AudioEvent

## TODO

- [ ] Implement compression pipeline (pruning + quantization)
- [ ] Add evaluation script with TTA (Test-Time Augmentation)
- [ ] Add AudioEvent dataset support
- [ ] Implement knowledge distillation training
- [ ] Add TensorFlow Lite export for edge deployment
- [ ] Add ONNX export support
- [ ] Add pre-trained model weights
- [ ] Implement Bootstrap CI calculation
- [ ] Add real-world deployment examples

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub.

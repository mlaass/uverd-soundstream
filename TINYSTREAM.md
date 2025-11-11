# TinyStream: Lightweight SoundStream for ESP32-S3

TinyStream is an ultra-lightweight variant of SoundStream optimized for ESP32-S3 deployment. It uses depthwise separable convolutions and a simplified architecture to achieve sub-MB model sizes while maintaining reasonable audio quality.

## Key Features

- **Encoder-only deployment**: Only encoder + quantizer run on ESP32, decoder stays on server
- **Lightweight architecture**: Depthwise separable convolutions, single residual unit per block
- **Fixed quantizers**: No quantizer dropout during training (simpler, more stable)
- **Multiple configs**: Ultra-tiny (~50KB), Tiny (~300KB), Small (~800KB)
- **Same training pipeline**: Fully compatible with train.py

## Model Specifications

| Config | C | D | Quantizers | Codebook | Parameters | Size (FP32) | Size (INT8) | Target Device |
|--------|---|---|------------|----------|------------|-------------|-------------|---------------|
| ultra_tiny | 4 | 64 | 2 | 256 | ~15K | ~60 KB | ~15 KB | ESP32-S3 512KB RAM |
| tiny | 8 | 128 | 4 | 512 | ~290K | ~1.1 MB | ~290 KB | ESP32-S3 512KB RAM |
| small | 12 | 192 | 4 | 1024 | ~1.1M | ~4.2 MB | ~1.1 MB | ESP32-S3 with PSRAM |

*Note: Size shown is full model with decoder. For ESP32 deployment (encoder only), divide by ~2.*

## Architecture Differences from SoundStream

| Feature | SoundStream | TinyStream |
|---------|-------------|------------|
| Base channels (C) | 32 | 8 |
| Embedding dim (D) | 512 | 128 |
| Strides | [2, 4, 5, 8] (320x) | [4, 4, 4, 4] (256x) |
| Residual units per block | 3 | 1 |
| Convolution type | Standard | Depthwise separable |
| Activation | ELU | ReLU (faster on embedded) |
| Quantizer dropout | Yes | No (fixed quantizers) |
| Sample rate | 24 kHz | 16 kHz |

## Training

### Quick Start

```bash
# Train with default 'tiny' config
./train_tiny.sh

# Train with ultra-tiny config (smallest model)
./train_tiny.sh --config ultra_tiny

# Train with small config (best quality)
./train_tiny.sh --config small

# Custom audio directory
./train_tiny.sh --config tiny --audio_dir /path/to/audio
```

### Manual Training

```bash
uv run python train.py \
    --model tinystream \
    --audio_dir datasets/ESC-50-master/audio \
    --batch_size 16 \
    --audio_length 1.0 \
    --C 8 \
    --D 128 \
    --num_quantizers 4 \
    --codebook_size 512 \
    --sample_rate 16000 \
    --checkpoint_dir ./checkpoints_tiny \
    --log_dir ./logs_tiny
```

### Training Notes

1. **No quantizer dropout**: TinyStream trains with a fixed number of quantizers for stability
2. **Audio chunk length**: Use 2.0 seconds (same as SoundStream) for discriminator compatibility
3. **Larger batch sizes**: Smaller model allows bigger batches (16-32 vs 8)
4. **Lower sample rate**: 16kHz vs 24kHz reduces computation and memory

## Inference

### Python API

```python
import torch
from model_tiny import TinyStream

# Load model
model = TinyStream(
    C=8,
    D=128,
    strides=[4, 4, 4, 4],
    num_quantizers=4,
    codebook_size=512,
    sample_rate=16000
)

# Load checkpoint
checkpoint = torch.load('checkpoints_tiny/soundstream_20250111_123456_step_50000.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Encode audio (for ESP32 deployment)
import torchaudio
audio, sr = torchaudio.load('test.wav')
audio = torchaudio.transforms.Resample(sr, 16000)(audio)
audio = audio.mean(0, keepdim=True).unsqueeze(0)  # Convert to mono

with torch.no_grad():
    indices = model.encode(audio)  # Returns (batch, num_quantizers, time)

print(f"Encoded to {indices.shape} discrete codes")
print(f"Compression: {audio.shape[-1]} samples -> {indices.shape[-1]} frames")

# Decode (server-side only, not used on ESP32)
with torch.no_grad():
    reconstructed = model.decode(indices)

torchaudio.save('reconstructed.wav', reconstructed[0], 16000)
```

## ESP32-S3 Deployment

### 1. Export Encoder Only

```python
import torch
from model_tiny import TinyStream

# Load trained model
model = TinyStream(C=8, D=128, num_quantizers=4, codebook_size=512, sample_rate=16000)
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Extract encoder + quantizer only (no decoder)
encoder_only = torch.nn.Sequential(
    model.encoder,
    model.quantizer
)

# Test
x = torch.randn(1, 1, 16000)
with torch.no_grad():
    z = model.encoder(x)
    quantized, indices, _ = model.quantizer(z)

print(f"Encoder output: {z.shape}")
print(f"Quantized indices: {indices.shape}")
```

### 2. Convert to ONNX

```python
import torch.onnx

# Export to ONNX
dummy_input = torch.randn(1, 1, 16000)
torch.onnx.export(
    encoder_only,
    dummy_input,
    "tinystream_encoder.onnx",
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=['audio'],
    output_names=['quantized', 'indices', 'commitment_loss'],
    dynamic_axes={
        'audio': {2: 'time'},
        'indices': {2: 'frames'}
    }
)
```

### 3. Quantize to INT8

```python
# Use PyTorch quantization or TensorFlow Lite converter
import torch.quantization as quantization

# Post-training static quantization
model_fp32 = encoder_only
model_fp32.eval()

# Fuse modules
model_fp32_fused = torch.quantization.fuse_modules(model_fp32, [['conv', 'relu']])

# Prepare for quantization
model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)

# Calibrate with representative data
for batch in calibration_data:
    model_fp32_prepared(batch)

# Convert to INT8
model_int8 = torch.quantization.convert(model_fp32_prepared)

# Model size reduced by ~4x
```

### 4. Deploy to ESP32-S3

- Convert to TensorFlow Lite or ESP-NN format
- Load model into ESP32-S3 SRAM/PSRAM
- Run inference on 16kHz audio chunks
- Send indices to server for decoding

## Bitrate and Quality

### Bitrate Calculation

```
Bitrate = (num_quantizers × log2(codebook_size) × embedding_rate) / 1000

For tiny config (4 quantizers, 512 codebook, 16kHz, 256x downsampling):
Embedding rate = 16000 / 256 = 62.5 Hz
Bitrate = 4 × log2(512) × 62.5 / 1000 = 4 × 9 × 62.5 / 1000 = 2.25 kbps
```

### Quality vs Size Tradeoff

| Config | Bitrate | Model Size | Quality | Use Case |
|--------|---------|------------|---------|----------|
| ultra_tiny (C=4, D=64, 2q) | 1.1 kbps | ~15 KB | ?? | ?? |
| tiny (C=8, D=128, 4q) | 2.25 kbps | ~290 KB | ?? | ?? |
| small (C=12, D=192, 4q) | 2.25 kbps | ~1.1 MB | ?? | ?? |

## Performance

### ESP32-S3 Specifications

- **CPU**: Dual-core Xtensa LX7 @ 240 MHz
- **RAM**: 512 KB SRAM (+ optional 2-8 MB PSRAM)
- **Inference speed**: ~10-50ms per chunk (depends on config)
- **Power**: ~100-200 mW during inference

### Recommended Settings

```python
# For real-time streaming on ESP32-S3
chunk_size = 16000  # 1 second chunks
hop_size = 8000     # 50% overlap
buffer_size = 2048  # Small buffer for low latency

# Model config
config = {
    'C': 8,
    'D': 128,
    'num_quantizers': 4,
    'codebook_size': 512,
    'sample_rate': 16000
}
```

## Comparison with SoundStream

| Metric | SoundStream (Base) | TinyStream (Tiny) | Reduction |
|--------|-------------------|-------------------|-----------|
| Parameters | 33M | 290K | 113x smaller |
| Model size (FP32) | 132 MB | 1.1 MB | 120x smaller |
| Model size (INT8) | 33 MB | 290 KB | 113x smaller |
| Sample rate | 24 kHz | 16 kHz | 0.67x |
| Bitrate | 6 kbps | 2.25 kbps | 0.38x |
| VRAM (training) | 12 GB | 4 GB | 3x less |

## Limitations

1. **Lower quality**: Simpler architecture means lower reconstruction quality
2. **Lower sample rate**: 16kHz vs 24kHz (bandwidth limited to 8kHz)
3. **Fixed quantizers**: Can't vary bitrate at inference time
4. **No streaming**: Current implementation processes full chunks

## Future Improvements

- [ ] Streaming inference support
- [ ] WebAssembly deployment
- [ ] ARM Cortex-M optimization
- [ ] Causal inference mode
- [ ] Online quantization (for continuous learning)

## Citation

Based on SoundStream by Zeghidour et al. (2021):

```bibtex
@article{zeghidour2021soundstream,
  title={SoundStream: An End-to-End Neural Audio Codec},
  author={Zeghidour, Neil and Luebs, Alejandro and Omran, Ahmed and Skoglund, Jan and Tagliasacchi, Marco},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year={2021}
}
```

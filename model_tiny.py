"""
TinyStream: Ultra-lightweight SoundStream for ESP32-S3 deployment
Encoder-only architecture optimized for embedded systems

Target specs:
- Model size: < 2 MB
- RAM usage: < 200 KB
- Real-time at 16 kHz
- Encoder + RVQ only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable convolution - much more efficient"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        # Depthwise
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=False
        )
        # Pointwise
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class TinyResidualUnit(nn.Module):
    """Lightweight residual unit with dilation support"""
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv1d(
            channels, channels, kernel_size=3,
            padding=dilation, dilation=dilation
        )
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.activation = nn.ReLU(inplace=True)  # ReLU is faster than ELU on embedded

    def forward(self, x):
        residual = x
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        return x + residual


class TinyEncoderBlock(nn.Module):
    """Tiny encoder block with multiple residual units (like SoundStream)"""
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        # Three residual units with different dilations (like SoundStream)
        self.res_units = nn.ModuleList([
            TinyResidualUnit(in_channels, dilation=1),
            TinyResidualUnit(in_channels, dilation=3),
            TinyResidualUnit(in_channels, dilation=9)
        ])
        # Downsampling with depthwise separable
        self.downsample = DepthwiseSeparableConv1d(
            in_channels, out_channels,
            kernel_size=2*stride, stride=stride, padding=stride//2
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        for res_unit in self.res_units:
            x = res_unit(x)
        x = self.activation(self.downsample(x))
        return x


class TinyEncoder(nn.Module):
    """Ultra-lightweight encoder for ESP32-S3"""
    def __init__(
        self,
        C: int = 8,           # Much smaller base channels
        D: int = 128,         # Much smaller embedding dimension
        strides: List[int] = [4, 4, 4, 4]  # 256x downsampling, 62.5 Hz @ 16kHz
    ):
        super().__init__()
        self.C = C
        self.D = D
        self.strides = strides
        
        # Initial convolution
        self.initial_conv = nn.Conv1d(1, C, kernel_size=3, padding=1, bias=False)
        
        # Encoder blocks with channel progression
        channels = [C, C*2, C*3, C*4]  # [8, 16, 24, 32]
        self.blocks = nn.ModuleList([
            TinyEncoderBlock(channels[i], channels[i+1] if i < 3 else channels[-1], stride=strides[i])
            for i in range(len(strides))
        ])
        
        # Final projection to embedding dimension
        self.final_conv = nn.Conv1d(channels[-1], D, kernel_size=1, bias=False)
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        """
        Args:
            x: (batch, 1, time) - input audio waveform
        Returns:
            (batch, D, time//M) - embeddings at reduced rate
        """
        x = self.activation(self.initial_conv(x))
        
        for block in self.blocks:
            x = block(x)
            
        x = self.final_conv(x)
        return x


class TinyDecoderBlock(nn.Module):
    """Tiny decoder block with multiple residual units (like SoundStream)"""
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        # Upsampling
        self.upsample = nn.ConvTranspose1d(
            in_channels, out_channels,
            kernel_size=2*stride, stride=stride, padding=stride//2,
            output_padding=stride-1, bias=False
        )
        # Three residual units with different dilations (like SoundStream)
        self.res_units = nn.ModuleList([
            TinyResidualUnit(out_channels, dilation=1),
            TinyResidualUnit(out_channels, dilation=3),
            TinyResidualUnit(out_channels, dilation=9)
        ])
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.activation(self.upsample(x))
        for res_unit in self.res_units:
            x = res_unit(x)
        return x


class TinyDecoder(nn.Module):
    """Ultra-lightweight decoder (training only, not for ESP32 deployment)"""
    def __init__(
        self,
        C: int = 8,
        D: int = 128,
        strides: List[int] = [4, 4, 4, 4]
    ):
        super().__init__()
        self.C = C
        self.D = D
        self.strides = strides

        # Channel progression (reverse of encoder)
        channels = [C*4, C*3, C*2, C]  # [32, 24, 16, 8]

        # Initial projection from embedding dimension
        self.initial_conv = nn.Conv1d(D, channels[0], kernel_size=1, bias=False)

        # Decoder blocks (reverse order)
        self.blocks = nn.ModuleList([
            TinyDecoderBlock(
                channels[i],
                channels[i+1] if i < len(channels)-1 else channels[-1],
                stride=strides[-(i+1)]
            )
            for i in range(len(strides))
        ])

        # Final convolution to audio
        self.final_conv = nn.Conv1d(channels[-1], 1, kernel_size=3, padding=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Args:
            x: (batch, D, time) - quantized embeddings
        Returns:
            (batch, 1, time*M) - reconstructed audio waveform
        """
        x = self.activation(self.initial_conv(x))

        for block in self.blocks:
            x = block(x)

        x = self.final_conv(x)
        return x


class TinyVectorQuantizer(nn.Module):
    """Simplified VQ for embedded deployment"""
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_weight: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_weight = commitment_weight

        # Codebook
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, x):
        """
        Args:
            x: (batch, dim, time)
        Returns:
            quantized: (batch, dim, time) - quantized embeddings
            indices: (batch, time) - codebook indices
            commitment_loss: scalar - commitment loss
        """
        B, D, T = x.shape
        x_flat = x.permute(0, 2, 1).reshape(-1, D)  # (B*T, D)

        # Compute distances (this is the heavy computation)
        distances = torch.sum(x_flat**2, dim=1, keepdim=True) + \
                   torch.sum(self.embeddings.weight**2, dim=1) - \
                   2 * torch.matmul(x_flat, self.embeddings.weight.t())

        # Get nearest codebook entry
        indices = torch.argmin(distances, dim=1)
        indices = indices.view(B, T)

        # Get quantized embeddings
        quantized_flat = self.embeddings(indices.view(-1))  # (B*T, D)
        quantized = quantized_flat.view(B, T, D).permute(0, 2, 1)  # (B, D, T)

        # Compute commitment loss (encourages encoder output to stay close to codebook)
        commitment_loss = self.commitment_weight * torch.mean((x.detach() - quantized) ** 2)

        # Straight-through estimator: copy gradients from quantized to x
        quantized = x + (quantized - x).detach()

        return quantized, indices, commitment_loss


class TinyResidualVQ(nn.Module):
    """Lightweight residual VQ"""
    def __init__(
        self,
        num_quantizers: int = 4,      # Fewer quantizers
        codebook_size: int = 512,     # Smaller codebooks
        embedding_dim: int = 128,
        commitment_weight: float = 0.25
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size

        self.quantizers = nn.ModuleList([
            TinyVectorQuantizer(codebook_size, embedding_dim, commitment_weight)
            for _ in range(num_quantizers)
        ])

    def forward(self, x):
        """
        Args:
            x: (batch, dim, time)
        Returns:
            quantized: (batch, dim, time) - sum of all quantized layers
            indices: (batch, num_quantizers, time) - codebook indices for all layers
            commitment_loss: scalar - sum of commitment losses
        """
        residual = x
        all_indices = []
        all_quantized = []
        total_commitment_loss = 0.0

        for quantizer in self.quantizers:
            quantized, indices, commitment_loss = quantizer(residual)
            all_indices.append(indices)
            all_quantized.append(quantized)
            total_commitment_loss = total_commitment_loss + commitment_loss

            # Subtract quantized version from residual for next layer
            residual = residual - quantized

        # Stack indices
        all_indices = torch.stack(all_indices, dim=1)  # (batch, num_q, time)

        # Sum all quantized layers
        final_quantized = torch.stack(all_quantized, dim=0).sum(dim=0)  # (batch, dim, time)

        return final_quantized, all_indices, total_commitment_loss


class TinyStream(nn.Module):
    """
    Complete tiny model for ESP32-S3 deployment

    Training: Uses encoder + quantizer + decoder (full model)
    Deployment (ESP32): Uses encoder + quantizer only (decoder stays on server)

    Note: Fixed number of quantizers (no dropout during training)
    """
    def __init__(
        self,
        C: int = 8,
        D: int = 128,
        strides: List[int] = [4, 4, 4, 4],
        num_quantizers: int = 4,
        codebook_size: int = 512,
        sample_rate: int = 16000,
        commitment_weight: float = 0.25
    ):
        super().__init__()
        self.C = C
        self.D = D
        self.strides = strides
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.sample_rate = sample_rate
        self.commitment_weight = commitment_weight

        # Calculate downsampling factor
        self.downsample_factor = 1
        for s in strides:
            self.downsample_factor *= s

        # Encoder + quantizer (for both training and ESP32 deployment)
        self.encoder = TinyEncoder(C, D, strides)
        self.quantizer = TinyResidualVQ(num_quantizers, codebook_size, D, commitment_weight)

        # Decoder (for training only, not deployed to ESP32)
        self.decoder = TinyDecoder(C, D, strides)

    def forward(self, x, num_quantizers=None):
        """
        Forward pass for training (matches SoundStream interface)

        Args:
            x: (batch, 1, time) - input waveform
            num_quantizers: ignored (fixed number of quantizers, no dropout)
        Returns:
            reconstructed: (batch, 1, time) - reconstructed audio
            indices: (batch, num_quantizers, time//M) - discrete codes
            commitment_loss: scalar - commitment loss
        """
        input_length = x.shape[-1]

        # Encode
        z = self.encoder(x)

        # Quantize (returns quantized embeddings, indices, and commitment loss)
        quantized, indices, commitment_loss = self.quantizer(z)

        # Decode
        reconstructed = self.decoder(quantized)

        # Ensure output matches input length (crop or pad as needed)
        if reconstructed.shape[-1] != input_length:
            if reconstructed.shape[-1] > input_length:
                # Crop from center
                start = (reconstructed.shape[-1] - input_length) // 2
                reconstructed = reconstructed[..., start:start + input_length]
            else:
                # Pad with zeros
                padding = input_length - reconstructed.shape[-1]
                reconstructed = torch.nn.functional.pad(reconstructed, (0, padding))

        return reconstructed, indices, commitment_loss

    def encode(self, x):
        """
        Encode audio to discrete codes (for ESP32 deployment)
        Returns only indices (no decoder needed on ESP32)
        """
        z = self.encoder(x)
        _, indices, _ = self.quantizer(z)
        return indices

    def decode(self, indices):
        """
        Decode indices back to audio (for testing, not used on ESP32)

        Args:
            indices: (batch, num_quantizers, time) - discrete codes
        Returns:
            audio: (batch, 1, time*M) - reconstructed audio
        """
        # Reconstruct quantized embeddings from indices
        B, num_q, T = indices.shape
        quantized = torch.zeros(B, self.D, T, device=indices.device)

        for i, quantizer in enumerate(self.quantizer.quantizers):
            layer_indices = indices[:, i, :]  # (B, T)
            layer_quantized = quantizer.embeddings(layer_indices)  # (B, T, D)
            layer_quantized = layer_quantized.permute(0, 2, 1)  # (B, D, T)
            quantized = quantized + layer_quantized

        # Decode
        audio = self.decoder(quantized)
        return audio
    
    def get_num_params(self):
        """Count parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_encoder_num_params(self):
        """Count encoder-only parameters (for ESP32 deployment)"""
        return sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)

    def get_model_size_mb(self):
        """Get model size in MB (fp32)"""
        return self.get_num_params() * 4 / (1024 * 1024)

    def get_encoder_size_mb(self):
        """Get encoder-only size in MB (fp32) - for ESP32 deployment"""
        return self.get_encoder_num_params() * 4 / (1024 * 1024)


def create_esp32_configs():
    """Predefined configurations for different ESP32 capabilities"""
    
    configs = {
        'ultra_tiny': {
            'C': 4,
            'D': 64,
            'strides': [4, 4, 4, 4],
            'num_quantizers': 2,
            'codebook_size': 256,
            'sample_rate': 16000,
            'target': 'ESP32-S3 with 512KB RAM',
            'model_size': '~50KB'
        },
        'tiny': {
            'C': 8,
            'D': 128,
            'strides': [4, 4, 4, 4],
            'num_quantizers': 4,
            'codebook_size': 512,
            'sample_rate': 16000,
            'target': 'ESP32-S3 with 512KB RAM',
            'model_size': '~300KB'
        },
        'small': {
            'C': 12,
            'D': 192,
            'strides': [4, 4, 4, 4],
            'num_quantizers': 4,
            'codebook_size': 1024,
            'sample_rate': 16000,
            'target': 'ESP32-S3 with PSRAM',
            'model_size': '~800KB'
        }
    }
    
    return configs


if __name__ == "__main__":
    # Test all configurations
    configs = create_esp32_configs()
    
    for name, config in configs.items():
        print(f"\n{'='*50}")
        print(f"Configuration: {name.upper()}")
        print(f"{'='*50}")
        
        # Create model
        model = TinyStream(
            C=config['C'],
            D=config['D'],
            strides=config['strides'],
            num_quantizers=config['num_quantizers'],
            codebook_size=config['codebook_size'],
            sample_rate=config['sample_rate']
        )
        
        # Model stats
        num_params = model.get_num_params()
        model_size_mb = model.get_model_size_mb()
        model_size_kb = model_size_mb * 1024
        
        print(f"Parameters: {num_params:,}")
        print(f"Model size (FP32): {model_size_kb:.1f} KB ({model_size_mb:.3f} MB)")
        print(f"Model size (INT8): {model_size_kb/4:.1f} KB")
        print(f"Downsampling: {model.downsample_factor}x")
        print(f"Embedding rate: {config['sample_rate'] / model.downsample_factor:.1f} Hz")
        
        # Calculate bitrate
        bits_per_frame = config['num_quantizers'] * torch.log2(torch.tensor(config['codebook_size'])).item()
        embedding_rate = config['sample_rate'] / model.downsample_factor
        bitrate_kbps = (bits_per_frame * embedding_rate) / 1000
        print(f"Bitrate: {bitrate_kbps:.1f} kbps")
        
        # Test forward pass
        batch_size = 1
        audio_length = config['sample_rate']  # 1 second
        x = torch.randn(batch_size, 1, audio_length)
        
        with torch.no_grad():
            indices = model(x)
        
        print(f"\nTest inference:")
        print(f"  Input: {x.shape}")
        print(f"  Output codes: {indices.shape}")
        print(f"  Codes per second: {indices.shape[-1]} frames")
        print(f"  Bytes per second: {indices.numel()} bytes")
        
        # Estimate RAM usage
        activation_size = x.numel() * 4  # Input
        for module in model.modules():
            if isinstance(module, nn.Conv1d):
                # Rough estimate of activation memory
                activation_size += 10000  # Conservative estimate
        
        print(f"\nEstimated RAM usage: {activation_size / 1024:.1f} KB")
        print(f"Target device: {config['target']}")

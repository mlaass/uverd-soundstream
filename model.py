"""
SoundStream: End-to-End Neural Audio Codec
Based on: https://arxiv.org/abs/2107.03312

Clean implementation for training on natural sounds (birds, forest ambience).
Optimized for 12GB VRAM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


class CausalConv1d(nn.Module):
    """Causal 1D convolution for streaming inference"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, dilation=dilation, padding=0
        )
        
    def forward(self, x):
        # Pad on the left (past) only for causality
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class ResidualUnit(nn.Module):
    """Residual unit with dilated causal convolution"""
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size=7, dilation=dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size=1)
        self.activation = nn.ELU()
        
    def forward(self, x):
        residual = x
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        return x + residual


class EncoderBlock(nn.Module):
    """Encoder block: 3 residual units + downsampling"""
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        # Three residual units with different dilations
        self.res_units = nn.ModuleList([
            ResidualUnit(in_channels, dilation=1),
            ResidualUnit(in_channels, dilation=3),
            ResidualUnit(in_channels, dilation=9)
        ])
        # Downsampling
        self.downsample = CausalConv1d(in_channels, out_channels, kernel_size=2*stride, stride=stride)
        self.activation = nn.ELU()
        
    def forward(self, x):
        for res_unit in self.res_units:
            x = res_unit(x)
        x = self.activation(self.downsample(x))
        return x


class DecoderBlock(nn.Module):
    """Decoder block: upsampling + 3 residual units"""
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        # Upsampling
        self.upsample = nn.ConvTranspose1d(
            in_channels, out_channels, 
            kernel_size=2*stride, stride=stride, padding=stride//2
        )
        # Three residual units with different dilations
        self.res_units = nn.ModuleList([
            ResidualUnit(out_channels, dilation=1),
            ResidualUnit(out_channels, dilation=3),
            ResidualUnit(out_channels, dilation=9)
        ])
        self.activation = nn.ELU()
        
    def forward(self, x):
        x = self.activation(self.upsample(x))
        for res_unit in self.res_units:
            x = res_unit(x)
        return x


class Encoder(nn.Module):
    """SoundStream Encoder"""
    def __init__(
        self,
        C: int = 32,          # Base number of channels
        D: int = 512,         # Embedding dimension
        strides: List[int] = [2, 4, 5, 8]
    ):
        super().__init__()
        self.C = C
        self.D = D
        self.strides = strides
        
        # Initial convolution
        self.initial_conv = CausalConv1d(1, C, kernel_size=7)
        
        # Encoder blocks
        channels = [C, C*2, C*4, C*8, C*16]
        self.blocks = nn.ModuleList([
            EncoderBlock(channels[i], channels[i+1], stride=strides[i])
            for i in range(len(strides))
        ])
        
        # Final convolution to embedding dimension
        self.final_conv = CausalConv1d(channels[-1], D, kernel_size=3)
        
    def forward(self, x):
        """
        Args:
            x: (batch, 1, time) - input audio waveform
        Returns:
            (batch, D, time//M) - embeddings at reduced rate
        """
        x = self.initial_conv(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.final_conv(x)
        return x


class Decoder(nn.Module):
    """SoundStream Decoder"""
    def __init__(
        self,
        C: int = 32,
        D: int = 512,
        strides: List[int] = [2, 4, 5, 8]
    ):
        super().__init__()
        self.C = C
        self.D = D
        self.strides = list(reversed(strides))
        
        # Initial convolution from embedding dimension
        channels = [C*16, C*8, C*4, C*2, C]
        self.initial_conv = CausalConv1d(D, channels[0], kernel_size=7)
        
        # Decoder blocks (reverse order of encoder)
        self.blocks = nn.ModuleList([
            DecoderBlock(channels[i], channels[i+1], stride=self.strides[i])
            for i in range(len(self.strides))
        ])
        
        # Final convolution to waveform
        self.final_conv = CausalConv1d(channels[-1], 1, kernel_size=7)
        
    def forward(self, x):
        """
        Args:
            x: (batch, D, time//M) - quantized embeddings
        Returns:
            (batch, 1, time) - reconstructed waveform
        """
        x = self.initial_conv(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.final_conv(x)
        return x


class VectorQuantizer(nn.Module):
    """Single layer of Vector Quantization with EMA updates"""
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Codebook
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
        # EMA parameters
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', torch.zeros(num_embeddings, embedding_dim))
        self.ema_decay = 0.99
        self.epsilon = 1e-5
        
    def forward(self, x):
        """
        Args:
            x: (batch, dim, time) - input embeddings
        Returns:
            quantized: (batch, dim, time) - quantized embeddings
            indices: (batch, time) - codebook indices
            commitment_loss: scalar - commitment loss
        """
        # Reshape: (batch, dim, time) -> (batch*time, dim)
        B, D, T = x.shape
        x_flat = x.permute(0, 2, 1).reshape(-1, D)
        
        # Compute distances to codebook entries
        distances = torch.sum(x_flat**2, dim=1, keepdim=True) + \
                   torch.sum(self.embeddings.weight**2, dim=1) - \
                   2 * torch.matmul(x_flat, self.embeddings.weight.t())
        
        # Get nearest codebook entry
        indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(indices, self.num_embeddings).float()
        
        # Quantize
        quantized_flat = torch.matmul(encodings, self.embeddings.weight)
        quantized = quantized_flat.view(B, T, D).permute(0, 2, 1)
        
        # Commitment loss
        commitment_loss = F.mse_loss(quantized.detach(), x) * self.commitment_cost
        
        # Straight-through estimator
        quantized = x + (quantized - x).detach()
        
        # EMA update (only during training)
        if self.training:
            with torch.no_grad():
                # Update cluster sizes
                self.ema_cluster_size = self.ema_cluster_size * self.ema_decay + \
                                       (1 - self.ema_decay) * torch.sum(encodings, dim=0)
                
                # Laplace smoothing
                n = torch.sum(self.ema_cluster_size)
                self.ema_cluster_size = (
                    (self.ema_cluster_size + self.epsilon) /
                    (n + self.num_embeddings * self.epsilon) * n
                )
                
                # Update embeddings
                dw = torch.matmul(encodings.t(), x_flat)
                self.ema_w = self.ema_w * self.ema_decay + (1 - self.ema_decay) * dw
                self.embeddings.weight.data = self.ema_w / self.ema_cluster_size.unsqueeze(1)
        
        indices = indices.view(B, T)
        return quantized, indices, commitment_loss


class ResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer with quantizer dropout"""
    def __init__(
        self,
        num_quantizers: int = 8,
        codebook_size: int = 1024,
        embedding_dim: int = 512,
        commitment_cost: float = 0.25
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        
        # Create quantizer layers
        self.quantizers = nn.ModuleList([
            VectorQuantizer(codebook_size, embedding_dim, commitment_cost)
            for _ in range(num_quantizers)
        ])
        
    def forward(self, x, num_quantizers: Optional[int] = None):
        """
        Args:
            x: (batch, dim, time) - input embeddings
            num_quantizers: number of quantizers to use (for dropout during training)
        Returns:
            quantized: (batch, dim, time) - quantized embeddings
            indices: (batch, num_quantizers, time) - all codebook indices
            commitment_loss: scalar
        """
        if num_quantizers is None:
            num_quantizers = self.num_quantizers
        
        quantized = torch.zeros_like(x)
        residual = x
        all_indices = []
        total_commitment_loss = 0.0
        
        # Apply quantizers sequentially to residuals
        for i in range(num_quantizers):
            q, indices, commit_loss = self.quantizers[i](residual)
            quantized = quantized + q
            residual = residual - q
            all_indices.append(indices)
            total_commitment_loss = total_commitment_loss + commit_loss
        
        # Stack indices: (batch, num_quantizers, time)
        all_indices = torch.stack(all_indices, dim=1)
        
        return quantized, all_indices, total_commitment_loss
    
    def decode_indices(self, indices):
        """Decode from codebook indices back to embeddings"""
        quantized = 0
        for i, quantizer in enumerate(self.quantizers[:indices.shape[1]]):
            codes = quantizer.embeddings(indices[:, i, :])  # (batch, time, dim)
            quantized = quantized + codes.permute(0, 2, 1)  # (batch, dim, time)
        return quantized


class SoundStream(nn.Module):
    """Complete SoundStream model"""
    def __init__(
        self,
        C: int = 32,                    # Base channels
        D: int = 512,                   # Embedding dimension
        strides: List[int] = [2, 4, 5, 8],
        num_quantizers: int = 8,
        codebook_size: int = 1024,
        sample_rate: int = 24000
    ):
        super().__init__()
        self.C = C
        self.D = D
        self.strides = strides
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.sample_rate = sample_rate
        
        # Calculate downsampling factor
        self.downsample_factor = 1
        for s in strides:
            self.downsample_factor *= s
        
        # Components
        self.encoder = Encoder(C, D, strides)
        self.quantizer = ResidualVectorQuantizer(num_quantizers, codebook_size, D)
        self.decoder = Decoder(C, D, strides)
        
    def forward(self, x, num_quantizers: Optional[int] = None):
        """
        Args:
            x: (batch, 1, time) - input waveform
            num_quantizers: number of quantizers to use (for quantizer dropout)
        Returns:
            recon: (batch, 1, time) - reconstructed waveform
            indices: (batch, num_quantizers, time//M) - codebook indices
            commitment_loss: scalar
        """
        # Encode
        z = self.encoder(x)
        
        # Quantize
        z_q, indices, commitment_loss = self.quantizer(z, num_quantizers)
        
        # Decode
        recon = self.decoder(z_q)
        
        return recon, indices, commitment_loss
    
    @torch.no_grad()
    def encode(self, x):
        """Encode audio to codebook indices"""
        z = self.encoder(x)
        _, indices, _ = self.quantizer(z)
        return indices
    
    @torch.no_grad()
    def decode(self, indices):
        """Decode codebook indices to audio"""
        z_q = self.quantizer.decode_indices(indices)
        return self.decoder(z_q)
    
    def get_num_params(self):
        """Count parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = SoundStream(
        C=32,
        D=512,
        strides=[2, 4, 5, 8],
        num_quantizers=8,
        codebook_size=1024,
        sample_rate=24000
    )
    
    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Downsampling factor: {model.downsample_factor}x")
    
    # Test forward pass
    batch_size = 2
    audio_length = 24000  # 1 second at 24kHz
    x = torch.randn(batch_size, 1, audio_length)
    
    recon, indices, commit_loss = model(x, num_quantizers=8)
    
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Indices shape: {indices.shape}")
    print(f"Commitment loss: {commit_loss.item():.4f}")
    
    # Test encoding/decoding
    indices_test = model.encode(x)
    recon_test = model.decode(indices_test)
    print(f"Encode-decode test passed: {recon_test.shape}")

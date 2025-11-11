"""
Discriminators for SoundStream
- Multi-scale wave discriminator (MelGAN-style)
- STFT discriminator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class DiscriminatorBlock(nn.Module):
    """Single discriminator with grouped convolutions"""
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(1, 16, kernel_size=15, stride=1, padding=7),
            nn.Conv1d(16, 64, kernel_size=41, stride=4, padding=20, groups=4),
            nn.Conv1d(64, 256, kernel_size=41, stride=4, padding=20, groups=16),
            nn.Conv1d(256, 1024, kernel_size=41, stride=4, padding=20, groups=64),
            nn.Conv1d(1024, 1024, kernel_size=41, stride=4, padding=20, groups=256),
            nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2),
        ])
        self.final_conv = nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1)
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        """
        Args:
            x: (batch, 1, time)
        Returns:
            logits: (batch, 1, time')
            feature_maps: list of intermediate activations
        """
        feature_maps = []
        
        for conv in self.convs:
            x = conv(x)
            x = self.activation(x)
            feature_maps.append(x)
        
        logits = self.final_conv(x)
        
        return logits, feature_maps


class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator operating on different resolutions"""
    def __init__(self, num_scales: int = 3):
        super().__init__()
        self.num_scales = num_scales
        
        # Create discriminators for each scale
        self.discriminators = nn.ModuleList([
            DiscriminatorBlock() for _ in range(num_scales)
        ])
        
        # Pooling for downsampling (except for first scale)
        self.pooling = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)
        
    def forward(self, x):
        """
        Args:
            x: (batch, 1, time) - input waveform
        Returns:
            logits_list: list of (batch, 1, time') tensors
            feature_maps_list: list of lists of feature maps
        """
        logits_list = []
        feature_maps_list = []
        
        for i, disc in enumerate(self.discriminators):
            if i > 0:
                # Downsample for subsequent scales
                x = self.pooling(x)
            
            logits, feature_maps = disc(x)
            logits_list.append(logits)
            feature_maps_list.append(feature_maps)
        
        return logits_list, feature_maps_list


class STFTDiscriminator(nn.Module):
    """STFT-based discriminator operating in time-frequency domain"""
    def __init__(
        self,
        C: int = 32,          # Base number of channels
        window_length: int = 1024,
        hop_length: int = 256
    ):
        super().__init__()
        self.window_length = window_length
        self.hop_length = hop_length
        self.C = C
        
        # Number of frequency bins (keeping DC, omitting Nyquist)
        self.F = window_length // 2
        
        # Initial 2D convolution (operates on real and imaginary parts)
        self.initial_conv = nn.Conv2d(2, C, kernel_size=7, stride=1, padding=3)
        
        # Residual blocks with alternating strides
        # Strides: (time, freq)
        strides = [(1, 2), (2, 2), (1, 2), (2, 2), (1, 2), (2, 2)]
        channels = [C, C*2, C*4, C*4, C*8, C*8]
        
        self.blocks = nn.ModuleList()
        in_channels = C
        for out_channels, (stride_t, stride_f) in zip(channels, strides):
            self.blocks.append(
                self._make_residual_block(in_channels, out_channels, stride_t, stride_f)
            )
            in_channels = out_channels
        
        # Final projection to aggregate across frequency
        # After 6 blocks with stride (*, 2), freq is reduced by 2^6 = 64
        final_freq = self.F // 64
        self.final_conv = nn.Conv2d(channels[-1], 1, kernel_size=(1, final_freq))
        
        self.activation = nn.LeakyReLU(0.2)
        
    def _make_residual_block(self, in_channels, out_channels, stride_t, stride_f):
        """Create a residual block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=(stride_t+2, stride_f+2),
                stride=(stride_t, stride_f),
                padding=1
            )
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, 1, time) - input waveform
        Returns:
            logits: (batch, 1, time')
            feature_maps: list of intermediate activations
        """
        # Compute STFT
        spec = torch.stft(
            x.squeeze(1),
            n_fft=self.window_length,
            hop_length=self.hop_length,
            window=torch.hann_window(self.window_length).to(x.device),
            return_complex=True
        )
        
        # Split into real and imaginary parts: (batch, freq, time, 2) -> (batch, 2, freq, time)
        real = spec.real
        imag = spec.imag
        x = torch.stack([real, imag], dim=1)  # (batch, 2, freq, time)
        
        # Initial conv
        x = self.activation(self.initial_conv(x))
        
        # Residual blocks
        feature_maps = []
        for block in self.blocks:
            x = self.activation(block(x))
            feature_maps.append(x)
        
        # Final projection: aggregate across frequency dimension
        logits = self.final_conv(x)  # (batch, 1, time', 1)
        logits = logits.squeeze(-1)  # (batch, 1, time')
        
        return logits, feature_maps


class CombinedDiscriminator(nn.Module):
    """Combined multi-scale wave + STFT discriminators"""
    def __init__(
        self,
        num_wave_scales: int = 3,
        stft_window_length: int = 1024,
        stft_hop_length: int = 256
    ):
        super().__init__()
        self.wave_disc = MultiScaleDiscriminator(num_wave_scales)
        self.stft_disc = STFTDiscriminator(
            window_length=stft_window_length,
            hop_length=stft_hop_length
        )
        
    def forward(self, real_audio, fake_audio):
        """
        Args:
            real_audio: (batch, 1, time) - real waveform
            fake_audio: (batch, 1, time) - generated waveform
        Returns:
            Dictionary with discriminator outputs for real and fake audio
        """
        # Multi-scale wave discriminator
        real_wave_logits, real_wave_features = self.wave_disc(real_audio)
        fake_wave_logits, fake_wave_features = self.wave_disc(fake_audio)
        
        # STFT discriminator
        real_stft_logits, real_stft_features = self.stft_disc(real_audio)
        fake_stft_logits, fake_stft_features = self.stft_disc(fake_audio)
        
        return {
            'real_wave_logits': real_wave_logits,
            'fake_wave_logits': fake_wave_logits,
            'real_wave_features': real_wave_features,
            'fake_wave_features': fake_wave_features,
            'real_stft_logits': real_stft_logits,
            'fake_stft_logits': fake_stft_logits,
            'real_stft_features': real_stft_features,
            'fake_stft_features': fake_stft_features
        }


def hinge_discriminator_loss(real_logits_list, fake_logits_list):
    """Hinge loss for discriminator"""
    loss = 0
    for real_logits, fake_logits in zip(real_logits_list, fake_logits_list):
        # Discriminator wants: real > 1, fake < -1
        loss += torch.mean(F.relu(1 - real_logits)) + torch.mean(F.relu(1 + fake_logits))
    return loss / len(real_logits_list)


def hinge_generator_loss(fake_logits_list):
    """Hinge loss for generator"""
    loss = 0
    for fake_logits in fake_logits_list:
        # Generator wants: fake > 1
        loss += torch.mean(F.relu(1 - fake_logits))
    return loss / len(fake_logits_list)


def feature_matching_loss(real_features_list, fake_features_list):
    """Feature matching loss (L1 distance between discriminator features)"""
    loss = 0
    count = 0
    
    for real_features, fake_features in zip(real_features_list, fake_features_list):
        for real_feat, fake_feat in zip(real_features, fake_features):
            loss += F.l1_loss(real_feat, fake_feat)
            count += 1
    
    return loss / count if count > 0 else loss


if __name__ == "__main__":
    # Test discriminators
    batch_size = 2
    audio_length = 24000
    
    real_audio = torch.randn(batch_size, 1, audio_length)
    fake_audio = torch.randn(batch_size, 1, audio_length)
    
    # Test combined discriminator
    discriminator = CombinedDiscriminator()
    outputs = discriminator(real_audio, fake_audio)
    
    print("Wave discriminator outputs:")
    print(f"  Real logits scales: {[x.shape for x in outputs['real_wave_logits']]}")
    print(f"  Fake logits scales: {[x.shape for x in outputs['fake_wave_logits']]}")
    
    print("\nSTFT discriminator outputs:")
    print(f"  Real logits: {outputs['real_stft_logits'].shape}")
    print(f"  Fake logits: {outputs['fake_stft_logits'].shape}")
    
    # Test losses
    disc_loss = hinge_discriminator_loss(
        outputs['real_wave_logits'] + [outputs['real_stft_logits']],
        outputs['fake_wave_logits'] + [outputs['fake_stft_logits']]
    )
    
    gen_loss = hinge_generator_loss(
        outputs['fake_wave_logits'] + [outputs['fake_stft_logits']]
    )
    
    feat_loss = feature_matching_loss(
        outputs['real_wave_features'] + [outputs['real_stft_features']],
        outputs['fake_wave_features'] + [outputs['fake_stft_features']]
    )
    
    print(f"\nDiscriminator loss: {disc_loss.item():.4f}")
    print(f"Generator adversarial loss: {gen_loss.item():.4f}")
    print(f"Feature matching loss: {feat_loss.item():.4f}")

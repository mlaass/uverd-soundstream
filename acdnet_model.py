"""
ACDNet: Acoustic Classification Deep Network
Implementation based on "Environmental Sound Classification on the Edge"
Paper: https://doi.org/10.1016/j.patcog.2022.109025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class SpectralFeatureExtractionBlock(nn.Module):
    """
    SFEB: Spectral Feature Extraction Block
    Extracts low-level audio features (spectral features) from raw audio
    through 1D convolutions at a frame rate of 10ms.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_filters: int = 8,
        input_length: int = 30225,
        sample_rate: int = 20000
    ):
        super().__init__()
        
        self.input_length = input_length
        self.sample_rate = sample_rate
        
        # First convolution: (1, 9) kernel, stride (1, 2)
        filters_1 = base_filters
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=filters_1,
            kernel_size=9,
            stride=2,
            padding=0
        )
        self.bn1 = nn.BatchNorm1d(filters_1)
        
        # Second convolution: (1, 5) kernel, stride (1, 2)
        filters_2 = int(filters_1 * (2**3))  # x * 2^3
        self.conv2 = nn.Conv1d(
            in_channels=filters_1,
            out_channels=filters_2,
            kernel_size=5,
            stride=2,
            padding=0
        )
        self.bn2 = nn.BatchNorm1d(filters_2)
        
        # Calculate pooling size dynamically
        # SFEB_PS = w / ((i_len/sr) * 1000) / 10)
        # This ensures ~10ms frame rate
        self.pool_size = None  # Will be calculated in forward pass
        
    def _calculate_pool_size(self, w: int) -> int:
        """Calculate the pooling kernel size based on output width"""
        target_frame_rate_ms = 10
        audio_duration_ms = (self.input_length / self.sample_rate) * 1000
        pool_size = int(w / (audio_duration_ms / target_frame_rate_ms))
        return max(1, pool_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, 1, time)
        Returns:
            Output tensor of shape (batch, channels, frequency, time)
        """
        # Conv1: (B, 1, T) -> (B, C1, T1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Conv2: (B, C1, T1) -> (B, C2, T2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # MaxPool with dynamic kernel size
        w = x.size(-1)
        if self.pool_size is None:
            self.pool_size = self._calculate_pool_size(w)
        
        x = F.max_pool1d(x, kernel_size=self.pool_size, stride=self.pool_size)
        
        # Swap axes from (B, C, T) to (B, 1, C, T) then reshape to (B, C, 1, T)
        # This prepares for 2D convolutions in TFEB
        x = x.unsqueeze(2)  # (B, C, 1, T)
        x = x.transpose(1, 2)  # (B, 1, C, T)
        
        return x


class TemporalFeatureExtractionBlock(nn.Module):
    """
    TFEB: Temporal Feature Extraction Block
    Extracts high-level hierarchical temporal features using 2D convolutions.
    Similar to VGG-13 architecture.
    """
    
    def __init__(
        self,
        in_channels: int,
        base_filters: int,
        num_classes: int,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        
        self.num_classes = num_classes

        # First conv block
        # TFEB starts at base_filters (32), not base_filters * 4!
        filters_3 = base_filters  # 32
        self.conv3 = nn.Conv2d(in_channels, filters_3, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(filters_3)

        # VGG-style blocks: 2 convs + pool
        # Block 1
        filters_4 = int(base_filters * 2)  # 64
        self.conv4 = nn.Conv2d(filters_3, filters_4, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(filters_4)
        self.conv5 = nn.Conv2d(filters_4, filters_4, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(filters_4)

        # Block 2
        filters_6 = int(base_filters * 4)  # 128
        self.conv6 = nn.Conv2d(filters_4, filters_6, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(filters_6)
        self.conv7 = nn.Conv2d(filters_6, filters_6, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(filters_6)

        # Block 3
        filters_8 = int(base_filters * 8)  # 256
        self.conv8 = nn.Conv2d(filters_6, filters_8, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(filters_8)
        self.conv9 = nn.Conv2d(filters_8, filters_8, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(filters_8)

        # Block 4
        filters_10 = int(base_filters * 16)  # 512
        self.conv10 = nn.Conv2d(filters_8, filters_10, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(filters_10)
        self.conv11 = nn.Conv2d(filters_10, filters_10, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(filters_10)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Conv12: 1x1 convolution to reduce channels (512 → num_classes)
        # This is specified in Table 2 of the ACDNet paper
        self.conv12 = nn.Conv2d(filters_10, num_classes, kernel_size=1)

        # Dense layer: num_classes → num_classes (after global pooling)
        # This is the final classification layer from the paper
        self.dense = nn.Linear(num_classes, num_classes)

        # Initialize dense layer with proper initialization
        nn.init.xavier_uniform_(self.dense.weight)
        nn.init.constant_(self.dense.bias, 0)
        
    def _calculate_pool_size(self, h: int, w: int, pool_idx: int, num_pools: int) -> Tuple[int, int]:
        """
        Calculate pooling kernel size according to the paper's formula.
        f(x, i) = 2 if x > 2 and i < N
                  1 if x = 1
                  x / 2^(N-1) if i = N
        """
        def f(x: int, i: int) -> int:
            if x > 2 and i < num_pools - 1:
                return 2
            elif x == 1:
                return 1
            elif i == num_pools - 1:
                return max(1, x // (2 ** (num_pools - 1)))
            else:
                return 1
        
        kh = f(h, pool_idx)
        kw = f(w, pool_idx)
        return kh, kw
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, 1, freq, time)
        Returns:
            Output tensor of shape (batch, num_classes)
        """
        batch_size = x.size(0)
        num_pools = 6  # Total number of pooling layers
        pool_idx = 0
        
        # First conv + pool
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        h, w = x.size(2), x.size(3)
        kh, kw = self._calculate_pool_size(h, w, pool_idx, num_pools)
        x = F.max_pool2d(x, kernel_size=(kh, kw), stride=(kh, kw))
        pool_idx += 1
        
        # Block 1: conv4, conv5, pool
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        
        h, w = x.size(2), x.size(3)
        kh, kw = self._calculate_pool_size(h, w, pool_idx, num_pools)
        x = F.max_pool2d(x, kernel_size=(kh, kw), stride=(kh, kw))
        pool_idx += 1
        
        # Block 2: conv6, conv7, pool
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        
        x = self.conv7(x)
        x = self.bn7(x)
        x = F.relu(x)
        
        h, w = x.size(2), x.size(3)
        kh, kw = self._calculate_pool_size(h, w, pool_idx, num_pools)
        x = F.max_pool2d(x, kernel_size=(kh, kw), stride=(kh, kw))
        pool_idx += 1
        
        # Block 3: conv8, conv9, pool
        x = self.conv8(x)
        x = self.bn8(x)
        x = F.relu(x)
        
        x = self.conv9(x)
        x = self.bn9(x)
        x = F.relu(x)
        
        h, w = x.size(2), x.size(3)
        kh, kw = self._calculate_pool_size(h, w, pool_idx, num_pools)
        x = F.max_pool2d(x, kernel_size=(kh, kw), stride=(kh, kw))
        pool_idx += 1
        
        # Block 4: conv10, conv11, pool
        x = self.conv10(x)
        x = self.bn10(x)
        x = F.relu(x)
        
        x = self.conv11(x)
        x = self.bn11(x)
        x = F.relu(x)
        
        h, w = x.size(2), x.size(3)
        kh, kw = self._calculate_pool_size(h, w, pool_idx, num_pools)
        x = F.max_pool2d(x, kernel_size=(kh, kw), stride=(kh, kw))
        pool_idx += 1
        
        # Dropout
        x = self.dropout(x)

        # Conv12: 1x1 convolution to reduce channels
        # (B, 512, h, w) -> (B, num_classes, h, w)
        x = self.conv12(x)
        x = F.relu(x)

        # Global average pooling: (B, num_classes, h, w) -> (B, num_classes, 1, 1)
        x = F.adaptive_avg_pool2d(x, (1, 1))

        # Flatten: (B, num_classes, 1, 1) -> (B, num_classes)
        x = x.view(batch_size, -1)

        # Dense layer: (B, num_classes) -> (B, num_classes)
        x = self.dense(x)

        return x


class ACDNet(nn.Module):
    """
    ACDNet: Acoustic Classification Deep Network
    
    A flexible and compression-friendly architecture for environmental sound
    classification using raw audio waveforms as input.
    
    Args:
        num_classes: Number of output classes
        input_length: Length of input audio in samples (default: 30225 ~1.51s @ 20kHz)
        sample_rate: Sample rate of input audio (default: 20000)
        base_filters: Base number of filters for SFEB (default: 8)
        dropout_rate: Dropout rate in TFEB (default: 0.2)
    """
    
    def __init__(
        self,
        num_classes: int,
        input_length: int = 30225,
        sample_rate: int = 20000,
        base_filters: int = 8,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_length = input_length
        self.sample_rate = sample_rate
        
        # Spectral Feature Extraction Block
        self.sfeb = SpectralFeatureExtractionBlock(
            in_channels=1,
            base_filters=base_filters,
            input_length=input_length,
            sample_rate=sample_rate
        )
        
        # Calculate SFEB output channels (not used by TFEB)
        sfeb_out_channels = int(base_filters * (2**3))

        # Temporal Feature Extraction Block
        # TFEB has its own independent base_filters=32 (matches reference implementation)
        # This is NOT derived from SFEB - it's a fixed constant per the paper
        self.tfeb = TemporalFeatureExtractionBlock(
            in_channels=1,  # After axis swap
            base_filters=32,  # Fixed at 32 per ACDNet paper/reference implementation
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, 1, time) or (batch, time)
        Returns:
            Output logits of shape (batch, num_classes)
        """
        # Ensure input is (B, 1, T)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # SFEB: Extract spectral features
        x = self.sfeb(x)  # (B, 1, C, T)
        
        # TFEB: Extract temporal features and classify
        x = self.tfeb(x)  # (B, num_classes)
        
        return x
    
    def get_num_parameters(self) -> int:
        """Return the total number of parameters in the model"""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_parameters(self) -> int:
        """Return the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_acdnet(
    num_classes: int,
    input_length: int = 30225,
    sample_rate: int = 20000,
    **kwargs
) -> ACDNet:
    """
    Factory function to create ACDNet model.
    
    Example:
        # For ESC-50 dataset (50 classes)
        model = create_acdnet(num_classes=50)
        
        # For ESC-10 dataset (10 classes)
        model = create_acdnet(num_classes=10)
    """
    return ACDNet(
        num_classes=num_classes,
        input_length=input_length,
        sample_rate=sample_rate,
        **kwargs
    )


if __name__ == "__main__":
    # Test the model
    batch_size = 4
    num_classes = 50
    input_length = 30225  # ~1.51s @ 20kHz
    
    model = create_acdnet(num_classes=num_classes, input_length=input_length)
    
    # Create random input
    x = torch.randn(batch_size, 1, input_length)
    
    # Forward pass
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {model.get_num_parameters():,}")
    print(f"Trainable parameters: {model.get_num_trainable_parameters():,}")
    
    # Calculate model size in MB (FP32)
    model_size_mb = (model.get_num_parameters() * 4) / (1024 ** 2)
    print(f"Model size (FP32): {model_size_mb:.2f} MB")

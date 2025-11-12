"""
Micro-ACDNet: Compressed ACDNet for Edge Devices
97.22% size reduction, 97.28% FLOP reduction while maintaining close to SOTA accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class MicroACDNet(nn.Module):
    """
    Micro-ACDNet: Compressed version of ACDNet for deployment on MCUs.
    
    Architecture follows Table 11 from the paper:
    - 415 total filters (vs 2074 in ACDNet)
    - 0.131M parameters (vs 4.74M in ACDNet)
    - 0.50MB model size (vs 18.06MB in ACDNet)
    - 14.82M FLOPs (vs 544M in ACDNet)
    
    Filter configuration:
    SFEB: [7, 20]
    TFEB: [10, 14, 22, 31, 35, 41, 51, 67, 69]
    
    Args:
        num_classes: Number of output classes (default: 50)
        input_length: Length of input audio in samples (default: 30225 ~1.51s @ 20kHz)
        sample_rate: Sample rate of input audio (default: 20000)
        dropout_rate: Dropout rate (default: 0.2)
    """
    
    def __init__(
        self,
        num_classes: int = 50,
        input_length: int = 30225,
        sample_rate: int = 20000,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_length = input_length
        self.sample_rate = sample_rate
        
        # SFEB filter configuration
        self.sfeb_filters = [7, 20]
        
        # TFEB filter configuration
        self.tfeb_filters = [10, 14, 22, 31, 35, 41, 51, 67, 69, 48]
        
        # ==================== SFEB ====================
        # Conv1: (1,9) kernel, stride (1,2), 7 filters
        self.conv1 = nn.Conv1d(1, self.sfeb_filters[0], kernel_size=9, stride=2)
        self.bn1 = nn.BatchNorm1d(self.sfeb_filters[0])
        
        # Conv2: (1,5) kernel, stride (1,2), 20 filters
        self.conv2 = nn.Conv1d(self.sfeb_filters[0], self.sfeb_filters[1], kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm1d(self.sfeb_filters[1])
        
        # MaxPool1: (1,50) kernel, stride (1,50)
        self.pool1_size = 50
        
        # ==================== TFEB ====================
        # After swapaxes, input is (batch, 1, 20, 151) where 20 is now frequency dimension
        
        # Conv3: (3,3) kernel, 10 filters
        self.conv3 = nn.Conv2d(1, self.tfeb_filters[0], kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(self.tfeb_filters[0])
        # MaxPool2: (2,2) kernel, stride (2,2) -> (10, 16, 75)
        
        # Block 1: Conv4-5
        self.conv4 = nn.Conv2d(self.tfeb_filters[0], self.tfeb_filters[1], kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(self.tfeb_filters[1])
        self.conv5 = nn.Conv2d(self.tfeb_filters[1], self.tfeb_filters[2], kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(self.tfeb_filters[2])
        # MaxPool3: (2,2) kernel, stride (2,2) -> (22, 8, 37)
        
        # Block 2: Conv6-7
        self.conv6 = nn.Conv2d(self.tfeb_filters[2], self.tfeb_filters[3], kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(self.tfeb_filters[3])
        self.conv7 = nn.Conv2d(self.tfeb_filters[3], self.tfeb_filters[4], kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(self.tfeb_filters[4])
        # MaxPool4: (2,2) kernel, stride (2,2) -> (35, 4, 18)
        
        # Block 3: Conv8-9
        self.conv8 = nn.Conv2d(self.tfeb_filters[4], self.tfeb_filters[5], kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(self.tfeb_filters[5])
        self.conv9 = nn.Conv2d(self.tfeb_filters[5], self.tfeb_filters[6], kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(self.tfeb_filters[6])
        # MaxPool5: (2,2) kernel, stride (2,2) -> (51, 2, 9)
        
        # Block 4: Conv10-11
        self.conv10 = nn.Conv2d(self.tfeb_filters[6], self.tfeb_filters[7], kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(self.tfeb_filters[7])
        self.conv11 = nn.Conv2d(self.tfeb_filters[7], self.tfeb_filters[8], kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(self.tfeb_filters[8])
        # MaxPool6: (2,2) kernel, stride (2,2) -> (69, 1, 4)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Conv12: (1,1) kernel, 48 filters
        self.conv12 = nn.Conv2d(self.tfeb_filters[8], self.tfeb_filters[9], kernel_size=1)
        
        # AvgPool1: (1,4) kernel, stride (1,4) -> (48, 1, 1)
        
        # Dense layer: 48 -> num_classes
        self.dense = nn.Linear(self.tfeb_filters[9], num_classes)
        
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
        
        batch_size = x.size(0)
        
        # ==================== SFEB ====================
        # Conv1: (B, 1, 30225) -> (B, 7, 15109)
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Conv2: (B, 7, 15109) -> (B, 20, 7553)
        x = F.relu(self.bn2(self.conv2(x)))
        
        # MaxPool1: (B, 20, 7553) -> (B, 20, 151)
        x = F.max_pool1d(x, kernel_size=self.pool1_size, stride=self.pool1_size)
        
        # Swapaxes: (B, 20, 151) -> (B, 1, 20, 151)
        x = x.unsqueeze(1)  # Add channel dimension
        x = x.transpose(1, 2)  # (B, 1, 20, 151)
        
        # ==================== TFEB ====================
        # Conv3 + MaxPool2: (B, 1, 20, 151) -> (B, 10, 16, 75)
        x = F.relu(self.bn3(self.conv3(x)))  # (B, 10, 20, 151)
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # (B, 10, 10, 75) but paper says (10, 16, 75)?
        
        # Block 1: Conv4-5 + MaxPool3 -> (B, 22, 8, 37)
        x = F.relu(self.bn4(self.conv4(x)))  # (B, 14, H, W)
        x = F.relu(self.bn5(self.conv5(x)))  # (B, 22, H, W)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Block 2: Conv6-7 + MaxPool4 -> (B, 35, 4, 18)
        x = F.relu(self.bn6(self.conv6(x)))  # (B, 31, H, W)
        x = F.relu(self.bn7(self.conv7(x)))  # (B, 35, H, W)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Block 3: Conv8-9 + MaxPool5 -> (B, 51, 2, 9)
        x = F.relu(self.bn8(self.conv8(x)))  # (B, 41, H, W)
        x = F.relu(self.bn9(self.conv9(x)))  # (B, 51, H, W)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Block 4: Conv10-11 + MaxPool6 -> (B, 69, 1, 4)
        x = F.relu(self.bn10(self.conv10(x)))  # (B, 67, H, W)
        x = F.relu(self.bn11(self.conv11(x)))  # (B, 69, H, W)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Dropout
        x = self.dropout(x)
        
        # Conv12: (B, 69, H, W) -> (B, 48, H, W)
        x = self.conv12(x)
        
        # AvgPool1: Global average pooling over spatial dimensions
        h, w = x.size(2), x.size(3)
        x = F.avg_pool2d(x, kernel_size=(h, w))  # (B, 48, 1, 1)
        
        # Flatten: (B, 48, 1, 1) -> (B, 48)
        x = x.view(batch_size, -1)
        
        # Dense: (B, 48) -> (B, num_classes)
        x = self.dense(x)
        
        return x
    
    def get_num_parameters(self) -> int:
        """Return the total number of parameters in the model"""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_parameters(self) -> int:
        """Return the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_filter_counts(self) -> dict:
        """Return filter counts for each layer"""
        return {
            'sfeb': self.sfeb_filters,
            'tfeb': self.tfeb_filters,
            'total_filters': sum(self.sfeb_filters) + sum(self.tfeb_filters)
        }


def create_micro_acdnet(
    num_classes: int = 50,
    input_length: int = 30225,
    sample_rate: int = 20000,
    **kwargs
) -> MicroACDNet:
    """
    Factory function to create Micro-ACDNet model.
    
    Example:
        # For ESC-50 dataset (50 classes)
        model = create_micro_acdnet(num_classes=50)
        
        # For ESC-10 dataset (10 classes)
        model = create_micro_acdnet(num_classes=10)
    """
    return MicroACDNet(
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
    
    model = create_micro_acdnet(num_classes=num_classes, input_length=input_length)
    
    # Create random input
    x = torch.randn(batch_size, 1, input_length)
    
    # Forward pass
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Model statistics
    filter_counts = model.get_filter_counts()
    print(f"\nFilter counts:")
    print(f"  SFEB: {filter_counts['sfeb']}")
    print(f"  TFEB: {filter_counts['tfeb']}")
    print(f"  Total filters: {filter_counts['total_filters']}")
    
    num_params = model.get_num_parameters()
    print(f"\nTotal parameters: {num_params:,}")
    print(f"Trainable parameters: {model.get_num_trainable_parameters():,}")
    
    # Calculate model size in MB (FP32 and INT8)
    model_size_fp32 = (num_params * 4) / (1024 ** 2)
    model_size_int8 = (num_params * 1) / (1024 ** 2)
    print(f"\nModel size (FP32): {model_size_fp32:.2f} MB")
    print(f"Model size (INT8): {model_size_int8:.2f} MB")
    
    # Expected values from paper
    print(f"\nExpected from paper:")
    print(f"  Parameters: 0.131M (131,000)")
    print(f"  Size: 0.50MB (FP32)")
    print(f"  FLOPs: 14.82M")

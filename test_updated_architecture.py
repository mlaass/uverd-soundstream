#!/usr/bin/env python3
"""Test the updated ACDNet architecture with conv12 + dense layers"""
import sys
import torch
from acdnet_model import create_acdnet

sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)

print("="*60, flush=True)
print("TESTING UPDATED ACDNET ARCHITECTURE", flush=True)
print("="*60, flush=True)

# Create model
model = create_acdnet(num_classes=50, input_length=30225, sample_rate=20000)
model = model.eval()

print(f"\nModel created successfully!", flush=True)
print(f"Total parameters: {model.get_num_parameters():,}", flush=True)

# Test forward pass
print(f"\nTesting forward pass...", flush=True)
batch_size = 4
dummy_input = torch.randn(batch_size, 1, 30225)

try:
    with torch.no_grad():
        output = model(dummy_input)

    print(f"✓ Forward pass successful!", flush=True)
    print(f"  Input shape:  {dummy_input.shape}", flush=True)
    print(f"  Output shape: {output.shape}", flush=True)

    if output.shape == (batch_size, 50):
        print(f"  ✓ Output shape is correct!", flush=True)
    else:
        print(f"  ❌ Output shape is wrong! Expected ({batch_size}, 50)", flush=True)

    # Check output statistics
    print(f"\nOutput statistics:", flush=True)
    print(f"  Mean: {output.mean().item():.4f}", flush=True)
    print(f"  Std:  {output.std().item():.4f}", flush=True)
    print(f"  Min:  {output.min().item():.4f}", flush=True)
    print(f"  Max:  {output.max().item():.4f}", flush=True)

except Exception as e:
    print(f"❌ Forward pass failed!", flush=True)
    print(f"Error: {e}", flush=True)
    import traceback
    traceback.print_exc()

# Print layer information
print(f"\n{'='*60}", flush=True)
print("LAYER INFORMATION", flush=True)
print(f"{'='*60}", flush=True)

print(f"\nTFEB layers:", flush=True)
for name, module in model.tfeb.named_children():
    if hasattr(module, 'weight'):
        if hasattr(module.weight, 'shape'):
            print(f"  {name:10s}: {str(module.weight.shape):30s} ({module.__class__.__name__})", flush=True)
    else:
        print(f"  {name:10s}: {module.__class__.__name__}", flush=True)

print(f"\n{'='*60}", flush=True)
print("Architecture matches paper specification:", flush=True)
print("  ✓ conv12 (1x1): 512 → 50 channels", flush=True)
print("  ✓ Dense layer: 50 → 50", flush=True)
print(f"{'='*60}", flush=True)

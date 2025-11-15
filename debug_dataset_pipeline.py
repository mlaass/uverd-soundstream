#!/usr/bin/env python3
"""Debug the full dataset pipeline"""
import sys
import torch
from dataset import create_dataloaders

sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)

print("="*60, flush=True)
print("DATASET PIPELINE DEBUG", flush=True)
print("="*60, flush=True)

# Create dataloaders
train_loader, test_loader = create_dataloaders(
    'esc50', './datasets/ESC-50-master',
    fold=1, batch_size=4, num_workers=0,
    augment=False, mixup=False
)

print(f"Train loader: {len(train_loader)} batches", flush=True)
print(f"Test loader: {len(test_loader)} batches", flush=True)

# Get first batch from test loader
audio_batch, label_batch = next(iter(test_loader))

print(f"\nBatch from test_loader:", flush=True)
print(f"  Audio shape: {audio_batch.shape}", flush=True)
print(f"  Labels shape: {label_batch.shape}", flush=True)
print(f"  Audio dtype: {audio_batch.dtype}", flush=True)
print(f"  Labels: {label_batch.tolist()}", flush=True)

for i in range(min(4, audio_batch.shape[0])):
    audio = audio_batch[i]
    print(f"\n  Sample {i}:", flush=True)
    print(f"    Mean: {audio.mean().item():.6f}", flush=True)
    print(f"    Std: {audio.std().item():.6f}", flush=True)
    print(f"    Min: {audio.min().item():.6f}", flush=True)
    print(f"    Max: {audio.max().item():.6f}", flush=True)

    if audio.std().item() < 0.001:
        print(f"    ❌ SILENT/ZERO AUDIO!", flush=True)
    else:
        print(f"    ✓ Has variance", flush=True)

# Check if all samples are identical
print(f"\nChecking if all samples are identical:", flush=True)
for i in range(1, min(4, audio_batch.shape[0])):
    diff = (audio_batch[0] - audio_batch[i]).abs().max().item()
    print(f"  Sample 0 vs {i}: max_diff = {diff:.6f}", flush=True)
    if diff < 1e-6:
        print(f"    ❌ IDENTICAL!", flush=True)

print("="*60, flush=True)

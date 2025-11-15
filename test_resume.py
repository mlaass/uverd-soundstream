#!/usr/bin/env python3
"""Test resume functionality"""
import torch
from pathlib import Path

checkpoint_path = "checkpoints_acdnet/acdnet_esc50_fold1_20251115_125936/checkpoint_latest.pt"

print("="*60)
print("TESTING RESUME FUNCTIONALITY")
print("="*60)

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location="cpu")

print(f"\nCheckpoint contents:")
for key in checkpoint.keys():
    if "state_dict" in key:
        print(f"  {key}: <state_dict>")
    else:
        print(f"  {key}: {checkpoint[key]}")

print(f"\nCurrent training state:")
print(f"  Last completed epoch: {checkpoint['epoch']}")
print(f"  Best accuracy: {checkpoint['best_acc']:.2f}%")
print(f"  Best epoch: {checkpoint['best_epoch']}")

print(f"\nResume will start from:")
print(f"  Epoch: {checkpoint['epoch'] + 1}")
print(f"  Continuing to epoch 2000: {2000 - checkpoint['epoch']} more epochs")

print("\n" + "="*60)
print("✓ Checkpoint loads successfully!")
print("="*60)

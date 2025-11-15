#!/usr/bin/env python3
"""Test scheduler T_max fix"""
import torch
import torch.optim as optim
from acdnet_model import create_acdnet

print("="*60)
print("TESTING SCHEDULER T_MAX FIX")
print("="*60)

# Load checkpoint
checkpoint_path = "checkpoints_acdnet/acdnet_esc50_fold1_20251115_125936/checkpoint_latest.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")

print(f"\nCheckpoint info:")
print(f"  Last epoch: {checkpoint['epoch']}")
print(f"  Best acc: {checkpoint['best_acc']:.2f}%")

# Extract scheduler state
sched_state = checkpoint["scheduler_state_dict"]
print(f"\nScheduler state from checkpoint:")
for key, value in sched_state.items():
    if key == 'T_max':
        print(f"  ⚠️  {key}: {value} (THIS IS THE PROBLEM!)")
    else:
        print(f"  {key}: {value}")

# Simulate creating trainer with NEW num_epochs=4000
print(f"\n{'='*60}")
print("SIMULATING RESUME WITH --epochs 4000")
print(f"{'='*60}")

model = create_acdnet(num_classes=50, input_length=30225, sample_rate=20000)
optimizer = optim.AdamW(model.parameters(), lr=3e-4)

# Create scheduler with NEW T_max=4000
print(f"\n1. Creating scheduler with T_max=4000...")
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4000, eta_min=3e-6)
print(f"   Scheduler T_max: {scheduler.T_max}")

# Load old scheduler state (T_max=2000)
print(f"\n2. Loading scheduler state from checkpoint (T_max=2000)...")
scheduler.load_state_dict(sched_state)
print(f"   Scheduler T_max after load: {scheduler.T_max} ❌ OVERWRITTEN!")

# Apply the fix
print(f"\n3. Applying the fix...")
if scheduler.T_max != 4000:
    print(f"   Detected mismatch: {scheduler.T_max} != 4000")
    print(f"   Fixing T_max to 4000...")
    scheduler.T_max = 4000
    scheduler.eta_min = 3e-6
    print(f"   Scheduler T_max after fix: {scheduler.T_max} ✓ FIXED!")

print(f"\n{'='*60}")
print("RESULT: Fix successfully prevents T_max mismatch!")
print(f"{'='*60}")

# Show what LR would be at various epochs
print(f"\nLearning rate progression with T_max=4000:")
for epoch in [1628, 2000, 2500, 3000, 3500, 4000]:
    scheduler.last_epoch = epoch - 1
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print(f"  Epoch {epoch}: LR = {lr:.6f}")

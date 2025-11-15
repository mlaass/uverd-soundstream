#!/usr/bin/env python3
"""Debug feature extraction to find where collapse happens"""
import sys
import torch
from acdnet_model import create_acdnet
from dataset import create_dataloaders

sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)

print("="*60, flush=True)
print("FEATURE DEBUGGING", flush=True)
print("="*60, flush=True)

train_loader, test_loader = create_dataloaders(
    'esc50', './datasets/ESC-50-master',
    fold=1, batch_size=8, num_workers=0,
    augment=False, mixup=False
)

model = create_acdnet(num_classes=50, input_length=30225, sample_rate=20000)
model = model.cuda()
model.eval()

# Get one batch
audio, labels = next(iter(test_loader))
audio = audio.cuda()

print(f"\nInput audio shape: {audio.shape}", flush=True)
print(f"Audio mean: {audio.mean().item():.4f}, std: {audio.std().item():.4f}", flush=True)

# Hook to capture activations
activations = {}

def hook_fn(name):
    def hook(module, input, output):
        activations[name] = output.detach()
    return hook

# Register hooks
model.sfeb.register_forward_hook(hook_fn('sfeb_out'))
model.tfeb.conv3.register_forward_hook(hook_fn('tfeb_conv3'))
model.tfeb.conv11.register_forward_hook(hook_fn('tfeb_conv11'))
model.tfeb.fc.register_forward_hook(hook_fn('fc_out'))

# Forward pass
with torch.no_grad():
    outputs = model(audio)

print(f"\n{'='*60}", flush=True)
print("ACTIVATION STATISTICS", flush=True)
print(f"{'='*60}", flush=True)

for name, act in activations.items():
    mean = act.mean().item()
    std = act.std().item()
    min_val = act.min().item()
    max_val = act.max().item()

    # Check if all samples have same features (collapsed)
    if len(act.shape) == 4:  # Conv output (B, C, H, W)
        # Average over spatial dims
        act_pooled = act.mean(dim=[2, 3])  # (B, C)
        sample_var = act_pooled.var(dim=0).mean().item()  # Variance across samples
        print(f"\n{name}: {act.shape}", flush=True)
        print(f"  mean={mean:.4f}, std={std:.4f}, min={min_val:.4f}, max={max_val:.4f}", flush=True)
        print(f"  Variance across samples: {sample_var:.6f}", flush=True)
        if sample_var < 0.01:
            print(f"  ❌ COLLAPSED! All samples have nearly identical features", flush=True)
    elif len(act.shape) == 2:  # FC output (B, C)
        sample_var = act.var(dim=0).mean().item()
        print(f"\n{name}: {act.shape}", flush=True)
        print(f"  mean={mean:.4f}, std={std:.4f}", flush=True)
        print(f"  Variance across samples: {sample_var:.6f}", flush=True)
        if sample_var < 0.01:
            print(f"  ❌ COLLAPSED! All samples have nearly identical features", flush=True)

print(f"\nFinal outputs shape: {outputs.shape}", flush=True)
print(f"Output mean: {outputs.mean().item():.4f}, std: {outputs.std().item():.4f}", flush=True)

# Check predictions
preds = outputs.argmax(dim=1)
unique_preds = preds.unique()
print(f"\nPredictions: {preds.tolist()}", flush=True)
print(f"Unique predictions: {len(unique_preds)} classes", flush=True)

print("="*60, flush=True)

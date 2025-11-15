#!/usr/bin/env python3
"""Minimal unbuffered test"""
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from acdnet_model import create_acdnet
from dataset import create_dataloaders

# Force unbuffered output
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)

print("="*60, flush=True)
print("MINIMAL TEST - 3 epochs, no mixup", flush=True)
print("="*60, flush=True)

print("\nLoading data...", flush=True)
train_loader, test_loader = create_dataloaders(
    'esc50', './datasets/ESC-50-master',
    fold=1, batch_size=64, num_workers=0,
    augment=False, mixup=False
)
print(f"Train batches: {len(train_loader)}", flush=True)
print(f"Test samples: {len(test_loader.dataset)}", flush=True)

print("\nCreating model...", flush=True)
model = create_acdnet(num_classes=50, input_length=30225, sample_rate=20000)
model = model.cuda()
print(f"Params: {model.get_num_parameters():,}", flush=True)

print("\nTesting AdamW LR=3e-4...", flush=True)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

losses = []
for epoch in range(1, 4):
    print(f"\n  Epoch {epoch}/3:", flush=True)
    model.train()
    epoch_loss = 0.0

    for batch_idx, (audio, labels) in enumerate(train_loader):
        audio = audio.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        outputs = model(audio)
        loss = nn.functional.cross_entropy(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"    Batch {batch_idx+1}/{len(train_loader)}: loss={loss.item():.4f}", flush=True)

    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)
    print(f"  → Epoch loss: {avg_loss:.4f}", flush=True)

# Quick test
print("\nEvaluating...", flush=True)
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for audio, labels in test_loader:
        audio = audio.cuda()
        labels = labels.cuda()
        outputs = model(audio)
        pred = outputs.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

test_acc = 100.0 * correct / total
loss_change = losses[0] - losses[-1]

print("\n" + "="*60, flush=True)
print("RESULT:", flush=True)
print(f"  Loss: {losses[0]:.4f} → {losses[-1]:.4f} (change: {loss_change:.4f})", flush=True)
print(f"  Test acc: {test_acc:.2f}%", flush=True)

if loss_change > 0.05:
    print("  ✓ Loss is decreasing - MODEL IS LEARNING!", flush=True)
elif loss_change > 0.01:
    print("  ⚠️  Loss decreasing slowly - weak learning", flush=True)
else:
    print("  ❌ Loss NOT decreasing - MODEL STUCK!", flush=True)

if test_acc > 10.0:
    print("  ✓ Test accuracy > 10% - GOOD!", flush=True)
elif test_acc > 5.0:
    print("  ⚠️  Test accuracy 5-10% - starting to learn", flush=True)
else:
    print("  ❌ Test accuracy < 5% - near random (2%)", flush=True)

print("="*60, flush=True)

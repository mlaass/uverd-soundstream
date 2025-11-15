#!/usr/bin/env python3
"""Test 50 epochs without mixup to see when accuracy improves"""
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from acdnet_model import create_acdnet
from dataset import create_dataloaders

sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)

print("="*60, flush=True)
print("50 EPOCH TEST - No Mixup, AdamW LR=3e-4", flush=True)
print("="*60, flush=True)

print("\nLoading data...", flush=True)
train_loader, test_loader = create_dataloaders(
    'esc50', './datasets/ESC-50-master',
    fold=1, batch_size=64, num_workers=0,
    augment=False, mixup=False
)

print("\nCreating model...", flush=True)
model = create_acdnet(num_classes=50, input_length=30225, sample_rate=20000)
model = model.cuda()

optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

print("\nTraining for 50 epochs...", flush=True)
print("Epoch | Train Loss | Test Acc", flush=True)
print("-" * 40, flush=True)

for epoch in range(1, 51):
    # Train
    model.train()
    epoch_loss = 0.0

    for audio, labels in train_loader:
        audio = audio.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        outputs = model(audio)
        loss = nn.functional.cross_entropy(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)

    # Evaluate every 5 epochs
    if epoch % 5 == 0:
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
        print(f"{epoch:5d} | {avg_loss:10.4f} | {test_acc:8.2f}%", flush=True)
    else:
        print(f"{epoch:5d} | {avg_loss:10.4f} | (skipped)", flush=True)

print("\n" + "="*60, flush=True)
print("Training complete!", flush=True)
print("="*60, flush=True)

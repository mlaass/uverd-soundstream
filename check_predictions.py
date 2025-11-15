#!/usr/bin/env python3
"""Check what the model is actually predicting"""
import sys
import torch
import torch.nn as nn
from acdnet_model import create_acdnet
from dataset import create_dataloaders
from collections import Counter

sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)

print("="*60, flush=True)
print("PREDICTION ANALYSIS", flush=True)
print("="*60, flush=True)

# Train a model for 10 epochs
print("\nTraining model for 10 epochs...", flush=True)
train_loader, test_loader = create_dataloaders(
    'esc50', './datasets/ESC-50-master',
    fold=1, batch_size=64, num_workers=0,
    augment=False, mixup=False
)

model = create_acdnet(num_classes=50, input_length=30225, sample_rate=20000)
model = model.cuda()

import torch.optim as optim
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

for epoch in range(1, 11):
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

    print(f"Epoch {epoch}: Loss = {epoch_loss / len(train_loader):.4f}", flush=True)

# Analyze predictions
print("\n" + "="*60, flush=True)
print("ANALYZING PREDICTIONS", flush=True)
print("="*60, flush=True)

model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for audio, labels in test_loader:
        audio = audio.cuda()
        labels = labels.cuda()

        outputs = model(audio)
        pred = outputs.argmax(dim=1)

        all_predictions.extend(pred.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

pred_counter = Counter(all_predictions)
label_counter = Counter(all_labels)

print(f"\nUnique classes predicted: {len(pred_counter)}/50", flush=True)
print(f"Unique classes in labels: {len(label_counter)}/50", flush=True)

if len(pred_counter) == 1:
    only_class = list(pred_counter.keys())[0]
    print(f"\n❌ MODEL COLLAPSED! Always predicts class {only_class}", flush=True)
elif len(pred_counter) < 10:
    print(f"\n❌ PARTIAL COLLAPSE! Only {len(pred_counter)} classes predicted", flush=True)
else:
    print(f"\n✓ Model predicting {len(pred_counter)} different classes", flush=True)

print(f"\nTop 10 predicted classes:", flush=True)
for class_idx, count in pred_counter.most_common(10):
    percentage = 100.0 * count / len(all_predictions)
    print(f"  Class {class_idx:2d}: {count:4d} ({percentage:5.2f}%)", flush=True)

print(f"\nTop 10 true label classes:", flush=True)
for class_idx, count in label_counter.most_common(10):
    percentage = 100.0 * count / len(all_labels)
    print(f"  Class {class_idx:2d}: {count:4d} ({percentage:5.2f}%)", flush=True)

# Calculate accuracy manually
correct = sum(1 for p, l in zip(all_predictions, all_labels) if p == l)
total = len(all_predictions)
acc = 100.0 * correct / total

print(f"\nManual accuracy calculation: {correct}/{total} = {acc:.2f}%", flush=True)

print("\n" + "="*60, flush=True)

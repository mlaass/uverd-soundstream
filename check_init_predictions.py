#!/usr/bin/env python3
"""Check predictions before and during training"""
import sys
import torch
import torch.nn as nn
from acdnet_model import create_acdnet
from dataset import create_dataloaders
from collections import Counter
import torch.optim as optim

sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)

print("="*60, flush=True)
print("INITIALIZATION vs TRAINING PREDICTIONS", flush=True)
print("="*60, flush=True)

train_loader, test_loader = create_dataloaders(
    'esc50', './datasets/ESC-50-master',
    fold=1, batch_size=64, num_workers=0,
    augment=False, mixup=False
)

model = create_acdnet(num_classes=50, input_length=30225, sample_rate=20000)
model = model.cuda()

def analyze_predictions(model, test_loader, desc):
    """Analyze what classes the model predicts"""
    model.eval()
    all_predictions = []
    all_logits = []

    with torch.no_grad():
        for audio, labels in test_loader:
            audio = audio.cuda()
            outputs = model(audio)
            pred = outputs.argmax(dim=1)
            all_predictions.extend(pred.cpu().tolist())
            all_logits.append(outputs.cpu())

    pred_counter = Counter(all_predictions)
    all_logits_tensor = torch.cat(all_logits, dim=0)

    print(f"\n{desc}:", flush=True)
    print(f"  Unique classes predicted: {len(pred_counter)}/50", flush=True)

    if len(pred_counter) == 1:
        print(f"  ❌ COLLAPSED to class {list(pred_counter.keys())[0]}", flush=True)
    elif len(pred_counter) < 10:
        print(f"  ⚠️  Only {len(pred_counter)} classes", flush=True)
    else:
        print(f"  ✓ Predicting {len(pred_counter)} classes", flush=True)

    print(f"  Top 5 predictions:", flush=True)
    for class_idx, count in pred_counter.most_common(5):
        pct = 100.0 * count / len(all_predictions)
        print(f"    Class {class_idx:2d}: {count:3d} ({pct:5.1f}%)", flush=True)

    # Check logit statistics
    logit_mean = all_logits_tensor.mean().item()
    logit_std = all_logits_tensor.std().item()
    print(f"  Logit mean: {logit_mean:.4f}, std: {logit_std:.4f}", flush=True)

# Epoch 0 (untrained)
analyze_predictions(model, test_loader, "EPOCH 0 (untrained model)")

# Train and check after each epoch
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

for epoch in [1, 2, 3, 5, 10]:
    # Train to this epoch
    while True:
        model.train()
        for audio, labels in train_loader:
            audio = audio.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(audio)
            loss = nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        analyze_predictions(model, test_loader, f"EPOCH {epoch}")
        break

print("\n" + "="*60, flush=True)

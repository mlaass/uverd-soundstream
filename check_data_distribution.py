#!/usr/bin/env python3
"""Check training and test data distribution"""
import sys
from dataset import create_dataloaders
from collections import Counter

sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)

print("="*60, flush=True)
print("DATA DISTRIBUTION ANALYSIS", flush=True)
print("="*60, flush=True)

train_loader, test_loader = create_dataloaders(
    'esc50', './datasets/ESC-50-master',
    fold=1, batch_size=1, num_workers=0,
    augment=False, mixup=False
)

# Check training set
print("\nTRAINING SET:", flush=True)
train_labels = []
for _, labels in train_loader:
    train_labels.append(labels.item())

train_counter = Counter(train_labels)
print(f"Total training samples: {len(train_labels)}", flush=True)
print(f"Unique classes: {len(train_counter)}/50", flush=True)

print(f"\nClass distribution:", flush=True)
for class_idx, count in sorted(train_counter.items()):
    print(f"  Class {class_idx:2d}: {count:3d} samples", flush=True)

# Check for imbalance
counts = list(train_counter.values())
min_count = min(counts)
max_count = max(counts)
print(f"\nMin samples per class: {min_count}", flush=True)
print(f"Max samples per class: {max_count}", flush=True)

if max_count > min_count * 1.5:
    print("❌ IMBALANCED! Some classes have >1.5x more samples", flush=True)
else:
    print("✓ Balanced - all classes have similar counts", flush=True)

# Check test set
print(f"\n{'='*60}", flush=True)
print("TEST SET:", flush=True)
test_labels = []
for _, labels in test_loader:
    test_labels.append(labels.item())

test_counter = Counter(test_labels)
print(f"Total test samples: {len(test_labels)}", flush=True)
print(f"Unique classes: {len(test_counter)}/50", flush=True)

print(f"\nClass distribution (first 20):", flush=True)
for class_idx, count in sorted(test_counter.items())[:20]:
    print(f"  Class {class_idx:2d}: {count:3d} samples", flush=True)

# Check which class appears most
most_common_train = train_counter.most_common(1)[0]
print(f"\n{'='*60}", flush=True)
print(f"Most common training class: {most_common_train[0]} ({most_common_train[1]} samples)", flush=True)
print(f"Model keeps collapsing to class 10", flush=True)

print("="*60, flush=True)

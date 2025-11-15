#!/usr/bin/env python3
"""
Diagnose why test accuracy is stuck at exactly 2.0% with zero fluctuation.
This is NOT normal - should see at least some variance if model is learning.
"""

import torch
import torch.nn as nn
from pathlib import Path
from acdnet_model import create_acdnet
from dataset import create_dataloaders
from collections import Counter
import numpy as np

def main():
    print("=" * 80)
    print("MODEL COLLAPSE DIAGNOSIS")
    print("=" * 80)

    # Find latest checkpoint
    checkpoint_dir = Path("./checkpoints_acdnet")
    checkpoint_paths = list(checkpoint_dir.glob("*/checkpoint_latest.pt"))

    if not checkpoint_paths:
        print("ERROR: No checkpoints found")
        print("Creating fresh model for testing...")
        model = create_acdnet(num_classes=50, input_length=30225, sample_rate=20000)
        checkpoint_epoch = 0
    else:
        checkpoint_path = checkpoint_paths[0]
        print(f"Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        checkpoint_epoch = checkpoint.get('epoch', 'unknown')
        best_acc = checkpoint.get('best_acc', 'unknown')

        print(f"Checkpoint epoch: {checkpoint_epoch}")
        print(f"Checkpoint best_acc: {best_acc}")

        model = create_acdnet(num_classes=50, input_length=30225, sample_rate=20000)
        model.load_state_dict(checkpoint['model_state_dict'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Load test data
    print("\n" + "=" * 80)
    print("LOADING TEST DATA")
    print("=" * 80)

    train_loader, test_loader = create_dataloaders(
        dataset_name='esc50',
        root='./datasets/ESC-50-master',
        fold=1,
        batch_size=32,
        num_workers=0,
        target_length=30225,
        target_sr=20000,
        augment=False,
        mixup=False,
    )

    print(f"Test batches: {len(test_loader)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Run evaluation with detailed logging
    print("\n" + "=" * 80)
    print("RUNNING DETAILED EVALUATION")
    print("=" * 80)

    all_predictions = []
    all_labels = []
    all_logits = []
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, (audio, labels) in enumerate(test_loader):
            audio = audio.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(audio)

            # Loss
            loss = nn.functional.cross_entropy(outputs, labels)
            total_loss += loss.item()

            # Predictions
            pred = outputs.argmax(dim=1)

            # Collect data
            all_predictions.extend(pred.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_logits.append(outputs.cpu())

            # Accuracy
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            # Print first batch details
            if batch_idx == 0:
                print(f"\nFIRST BATCH DETAILS:")
                print(f"  Audio shape: {audio.shape}")
                print(f"  Labels shape: {labels.shape}")
                print(f"  Outputs shape: {outputs.shape}")
                print(f"  Labels (first 10): {labels[:10].tolist()}")
                print(f"  Predictions (first 10): {pred[:10].tolist()}")
                print(f"  Logits (first sample, first 10 classes):")
                print(f"    {outputs[0][:10].tolist()}")
                print(f"  Max logit value: {outputs[0].max().item():.3f}")
                print(f"  Min logit value: {outputs[0].min().item():.3f}")

    # Overall metrics
    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total

    print(f"\n" + "=" * 80)
    print("OVERALL METRICS")
    print("=" * 80)
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.4f}%")
    print(f"Average loss: {avg_loss:.4f}")

    # Analyze predictions
    print(f"\n" + "=" * 80)
    print("PREDICTION ANALYSIS")
    print("=" * 80)

    pred_counter = Counter(all_predictions)
    label_counter = Counter(all_labels)

    print(f"Unique classes predicted: {len(pred_counter)}/50")
    print(f"Unique classes in labels: {len(label_counter)}/50")

    if len(pred_counter) < 5:
        print("\n⚠️  MODEL HAS COLLAPSED! Only predicting {len(pred_counter)} classes")

    print(f"\nTop 10 most predicted classes:")
    for class_idx, count in pred_counter.most_common(10):
        percentage = 100.0 * count / len(all_predictions)
        print(f"  Class {class_idx:2d}: {count:4d} times ({percentage:5.2f}%)")

    print(f"\nTrue label distribution (first 10):")
    for class_idx, count in label_counter.most_common(10):
        percentage = 100.0 * count / len(all_labels)
        print(f"  Class {class_idx:2d}: {count:4d} samples ({percentage:5.2f}%)")

    # Check if predictions are all the same
    if len(pred_counter) == 1:
        only_class = list(pred_counter.keys())[0]
        print(f"\n❌ CRITICAL: Model ALWAYS predicts class {only_class}!")
        print(f"   This explains the constant 2.0% accuracy (random chance for 50 classes)")

    # Analyze logits
    print(f"\n" + "=" * 80)
    print("LOGIT ANALYSIS")
    print("=" * 80)

    all_logits_tensor = torch.cat(all_logits, dim=0)  # (N, 50)

    # Check if all logits are similar
    logit_means = all_logits_tensor.mean(dim=0)
    logit_stds = all_logits_tensor.std(dim=0)

    print(f"Logit statistics across all samples:")
    print(f"  Mean of means: {logit_means.mean().item():.4f}")
    print(f"  Std of means: {logit_means.std().item():.4f}")
    print(f"  Mean of stds: {logit_stds.mean().item():.4f}")

    # Check if one class always has highest logit
    max_indices = all_logits_tensor.argmax(dim=1)
    max_counter = Counter(max_indices.tolist())

    print(f"\nClass with highest logit (distribution):")
    for class_idx, count in max_counter.most_common(5):
        percentage = 100.0 * count / len(max_indices)
        print(f"  Class {class_idx:2d}: {count:4d} times ({percentage:5.2f}%)")

    if max_counter.most_common(1)[0][1] > 0.9 * len(max_indices):
        dominant_class = max_counter.most_common(1)[0][0]
        print(f"\n❌ Class {dominant_class} dominates predictions (>{90}%)")

    # Sample predictions vs labels
    print(f"\n" + "=" * 80)
    print("SAMPLE PREDICTIONS vs LABELS (first 20)")
    print("=" * 80)
    print("Index | Label | Prediction | Correct?")
    print("-" * 45)
    for i in range(min(20, len(all_predictions))):
        is_correct = "✓" if all_predictions[i] == all_labels[i] else "✗"
        print(f"{i:5d} | {all_labels[i]:5d} | {all_predictions[i]:10d} | {is_correct}")

    # DIAGNOSIS
    print(f"\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    if len(pred_counter) == 1:
        print("❌ MODEL COLLAPSED: Predicting only ONE class")
        print("   → This causes constant 2.0% accuracy")
        print("\n   CAUSE: Model stuck in local minimum")
        print("   FIX: Restart with lower learning rate (0.01) or add gradient clipping")

    elif len(pred_counter) < 10:
        print(f"❌ MODEL PARTIALLY COLLAPSED: Only {len(pred_counter)} classes predicted")
        print("   → Accuracy will be constant at 2-4%")
        print("\n   CAUSE: Model not exploring enough classes")
        print("   FIX: Lower learning rate or reduce momentum")

    elif accuracy < 5.0 and checkpoint_epoch > 100:
        print(f"❌ MODEL NOT LEARNING: Accuracy {accuracy:.2f}% after {checkpoint_epoch} epochs")
        print("   → Should be >20% by epoch 100")
        print("\n   CAUSE: Model architecture or data issue")
        print("   FIX: Check model output shape, verify data pipeline")

    else:
        print(f"✓ Model predictions look reasonable")
        print(f"  Predicting {len(pred_counter)} different classes")
        print(f"  Accuracy: {accuracy:.2f}%")

        if checkpoint_epoch < 50:
            print(f"\n  → Epoch {checkpoint_epoch} is early, continue training")
        elif accuracy < 30.0:
            print(f"\n  ⚠️  Accuracy should be higher by epoch {checkpoint_epoch}")
            print("     Expected: ~30-50% by epoch 100")

if __name__ == "__main__":
    main()

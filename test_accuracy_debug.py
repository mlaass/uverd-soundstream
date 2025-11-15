#!/usr/bin/env python3
"""
Quick debug script to check if test accuracy calculation is working.
"""

import torch
import torch.nn as nn
from acdnet_model import create_acdnet
from dataset import create_dataloaders

def main():
    print("=" * 80)
    print("Test Accuracy Debug")
    print("=" * 80)

    # Create model
    model = create_acdnet(num_classes=50, input_length=30225, sample_rate=20000)
    model = model.cuda()
    model.eval()

    # Load data
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

    # Evaluate
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (audio, labels) in enumerate(test_loader):
            audio = audio.cuda()
            labels = labels.cuda()

            print(f"\nBatch {batch_idx}:")
            print(f"  Audio shape: {audio.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Labels dtype: {labels.dtype}")
            print(f"  Labels (first 5): {labels[:5].tolist()}")

            # Forward
            outputs = model(audio)
            print(f"  Output shape: {outputs.shape}")
            print(f"  Output dtype: {outputs.dtype}")

            # Loss
            loss = nn.functional.cross_entropy(outputs, labels)
            total_loss += loss.item()

            # Predictions
            pred = outputs.argmax(dim=1)
            print(f"  Predictions (first 5): {pred[:5].tolist()}")

            # Check if all predictions are the same
            unique_preds = pred.unique()
            print(f"  Unique predictions in batch: {len(unique_preds)} - {unique_preds.tolist()}")

            # Accuracy
            batch_correct = (pred == labels).sum().item()
            correct += batch_correct
            total += labels.size(0)

            batch_acc = 100.0 * batch_correct / labels.size(0)
            print(f"  Batch accuracy: {batch_acc:.2f}%")

            if batch_idx >= 2:  # Just check first 3 batches
                break

    overall_acc = 100.0 * correct / total
    avg_loss = total_loss / 3

    print(f"\n" + "=" * 80)
    print(f"Overall (first 3 batches):")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Accuracy: {overall_acc:.2f}%")
    print(f"  Correct: {correct}/{total}")
    print("=" * 80)

    # Check if model is completely untrained
    if overall_acc < 5.0:
        print("\n⚠️  Accuracy < 5% suggests model is untrained or broken")
        print("   Expected: Random chance = 2% (1/50 classes)")
        print("   This is normal for a fresh model")
    elif overall_acc > 50.0:
        print("\n✓ Accuracy > 50% - model is learning!")
    else:
        print(f"\n→ Accuracy = {overall_acc:.2f}% - model is starting to learn")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Debug script to identify the ACDNet checkpoint bug.

This script verifies that the dynamic dense layer in TFEB is NOT being saved
in checkpoints, causing random predictions during evaluation.
"""

import torch
from pathlib import Path
from acdnet_model import create_acdnet
from dataset import create_dataloaders
from collections import Counter

def main():
    print("=" * 80)
    print("ACDNet Checkpoint Debug Script")
    print("=" * 80)

    # Find latest checkpoint
    checkpoint_dir = Path("./checkpoints_acdnet")
    checkpoint_paths = list(checkpoint_dir.glob("*/checkpoint_latest.pt"))

    if not checkpoint_paths:
        print("ERROR: No checkpoints found in ./checkpoints_acdnet/")
        print("Please train the model first.")
        return

    checkpoint_path = checkpoint_paths[0]
    print(f"\nUsing checkpoint: {checkpoint_path}")

    # Create model
    print("\n" + "=" * 80)
    print("STEP 1: Create fresh model")
    print("=" * 80)
    model = create_acdnet(num_classes=50, input_length=30225, sample_rate=20000)

    print(f"Model has 'dense' attribute: {hasattr(model.tfeb, 'dense')}")
    print(f"Dense layer value: {model.tfeb.dense}")
    dense_keys = [k for k in model.state_dict().keys() if 'dense' in k]
    print(f"State dict keys with 'dense': {dense_keys if dense_keys else 'NONE'}")

    # Load checkpoint
    print("\n" + "=" * 80)
    print("STEP 2: Load checkpoint")
    print("=" * 80)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Best accuracy: {checkpoint.get('best_acc', 'unknown'):.2f}%")

    ckpt_dense_keys = [k for k in checkpoint['model_state_dict'].keys() if 'dense' in k]
    print(f"Checkpoint state dict keys with 'dense': {ckpt_dense_keys if ckpt_dense_keys else 'NONE ❌'}")

    if not ckpt_dense_keys:
        print("\n⚠️  BUG CONFIRMED: Dense layer weights NOT in checkpoint!")

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"\nAfter loading:")
    print(f"Dense layer value: {model.tfeb.dense}")

    # Run a forward pass to trigger dynamic creation
    print("\n" + "=" * 80)
    print("STEP 3: Run forward pass (triggers dynamic dense creation)")
    print("=" * 80)
    dummy_input = torch.randn(2, 30225)  # Batch of 2
    model.eval()

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Output shape: {output.shape}")
    print(f"Output logits (first sample, first 5 classes): {output[0][:5].tolist()}")

    print(f"\nAfter forward pass:")
    print(f"Dense layer created: {model.tfeb.dense is not None}")
    if model.tfeb.dense is not None:
        print(f"Dense layer weight shape: {model.tfeb.dense.weight.shape}")
        print(f"Dense layer bias shape: {model.tfeb.dense.bias.shape}")
        dense_in_state = 'tfeb.dense.weight' in model.state_dict()
        print(f"Dense layer in state_dict NOW: {dense_in_state}")
        if not dense_in_state:
            print("⚠️  Dense layer still NOT registered in state_dict!")

    # Test on real data
    print("\n" + "=" * 80)
    print("STEP 4: Test on real ESC-50 data")
    print("=" * 80)

    try:
        train_loader, test_loader = create_dataloaders(
            dataset_name='esc50',
            root='./datasets/ESC-50-master',
            fold=1,
            batch_size=8,
            num_workers=0,
            target_length=30225,
            target_sr=20000,
            augment=False,
            mixup=False,
        )
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return

    # Get one batch
    audio, labels = next(iter(test_loader))

    with torch.no_grad():
        outputs = model(audio)
        predictions = outputs.argmax(dim=1)

    accuracy = (predictions == labels).float().mean().item()

    print(f"Batch size: {audio.shape[0]}")
    print(f"True labels:  {labels.tolist()}")
    print(f"Predictions:  {predictions.tolist()}")
    print(f"Accuracy: {accuracy:.2%}")

    # Check if predictions are all the same
    unique_preds = set(predictions.tolist())
    print(f"\nUnique predictions in batch: {len(unique_preds)}")
    if len(unique_preds) == 1:
        print("⚠️  All predictions are the SAME class - model is broken!")

    # Check prediction distribution over multiple batches
    print("\n" + "=" * 80)
    print("STEP 5: Prediction distribution (100 batches)")
    print("=" * 80)

    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    for i, (audio, labels) in enumerate(test_loader):
        if i >= 100:
            break
        with torch.no_grad():
            outputs = model(audio)
            predictions = outputs.argmax(dim=1)
            all_preds.extend(predictions.tolist())
            all_labels.extend(labels.tolist())
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    overall_accuracy = 100.0 * correct / total

    pred_counts = Counter(all_preds)
    label_counts = Counter(all_labels)

    print(f"Total samples: {len(all_preds)}")
    print(f"Overall accuracy: {overall_accuracy:.2f}%")
    print(f"Unique classes predicted: {len(pred_counts)}/50")
    print(f"Unique classes in labels: {len(label_counts)}/50")
    print(f"\nTop 10 predicted classes:")
    for class_idx, count in pred_counts.most_common(10):
        percentage = 100.0 * count / len(all_preds)
        print(f"  Class {class_idx:2d}: {count:4d} predictions ({percentage:5.2f}%)")

    # Diagnosis
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    if not ckpt_dense_keys:
        print("❌ BUG CONFIRMED: Dense layer NOT in checkpoint")
        print("   → Dense layer gets random weights on every evaluation")
        print("   → This causes ~2% random accuracy")

    if overall_accuracy < 5.0:
        print(f"❌ Accuracy ({overall_accuracy:.2f}%) is near random (2% = 1/50)")
        print("   → Model predictions are essentially random")

    if len(pred_counts) < 10:
        print(f"❌ Only {len(pred_counts)} classes predicted out of 50")
        print("   → Model is biased towards a few classes")

    print("\nRECOMMENDED FIX:")
    print("1. Remove the dynamic dense layer from acdnet_model.py")
    print("2. Delete all corrupted checkpoints")
    print("3. Retrain from scratch")
    print("4. Expect accuracy to improve to 80%+ after 2000 epochs")

if __name__ == "__main__":
    main()

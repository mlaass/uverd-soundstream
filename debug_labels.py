"""
Debug script to diagnose ACDNet label alignment issue.
"""

import torch
import torch.nn as nn
from pathlib import Path
from dataset import create_dataloaders
from acdnet_model import create_acdnet
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='datasets/ESC-50-master')
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    
    print("=" * 80)
    print("DEBUGGING ACDNET LABEL ALIGNMENT")
    print("=" * 80)
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        dataset_name='esc50',
        root=args.data_root,
        fold=1,
        batch_size=32,
        num_workers=0,
        augment=False,
        mixup=False
    )
    
    # Check dataset properties
    print("\n=== DATASET PROPERTIES ===")
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")
    print(f"Train class_names: {train_loader.dataset.class_names[:5]}...")
    print(f"Test class_names: {test_loader.dataset.class_names[:5]}...")
    print(f"Num classes (train): {len(train_loader.dataset.class_names)}")
    print(f"Num classes (test): {len(test_loader.dataset.class_names)}")
    
    # Get one batch from each
    print("\n=== TRAIN BATCH ===")
    train_audio, train_labels = next(iter(train_loader))
    print(f"Audio shape: {train_audio.shape}")
    print(f"Labels shape: {train_labels.shape}")
    print(f"Labels dtype: {train_labels.dtype}")
    print(f"Labels range: [{train_labels.min().item()}, {train_labels.max().item()}]")
    print(f"Unique labels in batch: {torch.unique(train_labels).tolist()}")
    print(f"First 10 labels: {train_labels[:10].tolist()}")
    
    print("\n=== TEST BATCH ===")
    test_audio, test_labels = next(iter(test_loader))
    print(f"Audio shape: {test_audio.shape}")
    print(f"Labels shape: {test_labels.shape}")
    print(f"Labels dtype: {test_labels.dtype}")
    print(f"Labels range: [{test_labels.min().item()}, {test_labels.max().item()}]")
    print(f"Unique labels in batch: {torch.unique(test_labels).tolist()}")
    print(f"First 10 labels: {test_labels[:10].tolist()}")
    
    # Create model
    num_classes = len(train_loader.dataset.class_names)
    model = create_acdnet(num_classes=num_classes, input_length=30225, sample_rate=20000)
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"\n=== LOADING CHECKPOINT: {args.checkpoint} ===")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    model.eval()
    
    # Test on train batch
    print("\n=== MODEL INFERENCE ON TRAIN BATCH ===")
    with torch.no_grad():
        train_outputs = model(train_audio)
        print(f"Output shape: {train_outputs.shape}")
        print(f"Output range: [{train_outputs.min().item():.3f}, {train_outputs.max().item():.3f}]")
        train_preds = train_outputs.argmax(dim=1)
        print(f"Predictions: {train_preds[:10].tolist()}")
        print(f"Ground truth: {train_labels[:10].tolist()}")
        train_acc = (train_preds == train_labels).float().mean().item() * 100
        print(f"Batch accuracy: {train_acc:.2f}%")
    
    # Test on test batch
    print("\n=== MODEL INFERENCE ON TEST BATCH ===")
    with torch.no_grad():
        test_outputs = model(test_audio)
        print(f"Output shape: {test_outputs.shape}")
        print(f"Output range: [{test_outputs.min().item():.3f}, {test_outputs.max().item():.3f}]")
        test_preds = test_outputs.argmax(dim=1)
        print(f"Predictions: {test_preds[:10].tolist()}")
        print(f"Ground truth: {test_labels[:10].tolist()}")
        test_acc = (test_preds == test_labels).float().mean().item() * 100
        print(f"Batch accuracy: {test_acc:.2f}%")
    
    # Check prediction distribution
    print("\n=== PREDICTION DISTRIBUTION (TEST BATCH) ===")
    pred_counts = torch.bincount(test_preds, minlength=num_classes)
    print(f"Predictions per class (first 10): {pred_counts[:10].tolist()}")
    print(f"Total predictions: {pred_counts.sum().item()}")
    print(f"Classes predicted: {(pred_counts > 0).sum().item()} / {num_classes}")
    
    # Check if model is stuck predicting same class
    if (pred_counts > 0).sum().item() == 1:
        stuck_class = pred_counts.argmax().item()
        print(f"WARNING: Model only predicting class {stuck_class}")
    
    # Full test set evaluation
    print("\n=== FULL TEST SET EVALUATION ===")
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for audio, labels in test_loader:
            outputs = model(audio)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
    
    accuracy = 100.0 * correct / total
    print(f"Test accuracy: {accuracy:.2f}%")
    print(f"Correct: {correct} / {total}")
    
    # Check prediction distribution
    all_preds_tensor = torch.tensor(all_preds)
    all_labels_tensor = torch.tensor(all_labels)
    pred_dist = torch.bincount(all_preds_tensor, minlength=num_classes)
    label_dist = torch.bincount(all_labels_tensor, minlength=num_classes)
    
    print(f"\nPrediction distribution (first 10): {pred_dist[:10].tolist()}")
    print(f"Label distribution (first 10): {label_dist[:10].tolist()}")
    
    # Check class name alignment
    print("\n=== CLASS NAME VERIFICATION ===")
    print("Checking if train and test use same class_names order...")
    if train_loader.dataset.class_names == test_loader.dataset.class_names:
        print("✓ Class names are identical")
    else:
        print("✗ Class names DIFFER!")
        print(f"Train: {train_loader.dataset.class_names}")
        print(f"Test: {test_loader.dataset.class_names}")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()

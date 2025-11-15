#!/usr/bin/env python3
"""
Diagnose why loss is stuck at 3.2 and not decreasing.
Test different scenarios to find the root cause.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from acdnet_model import create_acdnet
from dataset import create_dataloaders
from tqdm import tqdm

def test_scenario(name, train_loader, test_loader, epochs=10, **kwargs):
    """Test a training scenario and report results."""
    print(f"\n{'='*80}")
    print(f"SCENARIO: {name}")
    print(f"{'='*80}")

    model = create_acdnet(num_classes=50, input_length=30225, sample_rate=20000)
    model = model.cuda()

    # Get config
    lr = kwargs.get('lr', 3e-4)
    optimizer_type = kwargs.get('optimizer', 'adamw')

    # Create optimizer
    if optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"Optimizer: {optimizer_type}, LR: {lr}")
    print(f"Train batches: {len(train_loader)}, Test samples: {len(test_loader.dataset)}")

    # Get one batch to check
    audio, labels = next(iter(train_loader))
    print(f"Batch shape: {audio.shape}")
    print(f"Label shape: {labels.shape}")
    print(f"Label dtype: {labels.dtype}")

    if labels.dim() > 1:
        print(f"Soft labels detected (shape {labels.shape})")
        print(f"Sample label (first 10 classes): {labels[0][:10]}")
    else:
        print(f"Hard labels detected")
        print(f"Sample labels: {labels[:10]}")

    # Training loop
    losses = []
    test_accs = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for audio, labels in train_loader:
            audio = audio.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(audio)

            # Loss calculation
            if labels.dim() > 1:
                # Soft labels (mixup) - use KL divergence
                log_pred = torch.log_softmax(outputs, dim=1)
                target_dist = torch.softmax(labels, dim=1) if labels.dtype == torch.float else labels
                loss = nn.functional.kl_div(log_pred, target_dist, reduction='batchmean')
            else:
                # Hard labels - use cross entropy
                loss = nn.functional.cross_entropy(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        # Quick test
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
        test_accs.append(test_acc)

        print(f"Epoch {epoch:2d}: Loss = {avg_loss:.4f}, Test Acc = {test_acc:.2f}%")

    # Summary
    print(f"\nRESULTS:")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Loss change: {losses[0] - losses[-1]:.4f}")
    print(f"  Final test acc: {test_accs[-1]:.2f}%")

    if losses[0] - losses[-1] < 0.1:
        print(f"  ❌ BARELY LEARNING (loss change < 0.1)")
    elif test_accs[-1] < 5.0:
        print(f"  ❌ NOT LEARNING (test acc near random 2%)")
    elif test_accs[-1] > 10.0:
        print(f"  ✓ LEARNING! (test acc > 10%)")
    else:
        print(f"  ⚠️  SLOW LEARNING (some progress but weak)")

    return losses, test_accs


def main():
    print("="*80)
    print("LEARNING DIAGNOSTIC")
    print("="*80)

    # Test 1: No mixup, standard settings
    print("\n\nTEST 1: NO MIXUP (to verify basic learning works)")
    train_loader, test_loader = create_dataloaders(
        'esc50', './datasets/ESC-50-master',
        fold=1, batch_size=64, num_workers=0,
        augment=False, mixup=False
    )
    test_scenario("No Mixup + AdamW (LR=3e-4)", train_loader, test_loader,
                  epochs=20, lr=3e-4, optimizer='adamw')

    # Test 2: Higher learning rate
    print("\n\nTEST 2: NO MIXUP + HIGHER LR")
    test_scenario("No Mixup + AdamW (LR=1e-3)", train_loader, test_loader,
                  epochs=20, lr=1e-3, optimizer='adamw')

    # Test 3: SGD
    print("\n\nTEST 3: NO MIXUP + SGD")
    test_scenario("No Mixup + SGD (LR=0.01)", train_loader, test_loader,
                  epochs=20, lr=0.01, optimizer='sgd')

    # Test 4: With mixup
    print("\n\nTEST 4: WITH MIXUP")
    train_loader_mixup, _ = create_dataloaders(
        'esc50', './datasets/ESC-50-master',
        fold=1, batch_size=64, num_workers=0,
        augment=False, mixup=True
    )
    test_scenario("Mixup + AdamW (LR=3e-4)", train_loader_mixup, test_loader,
                  epochs=20, lr=3e-4, optimizer='adamw')

    # Test 5: Smaller batch size
    print("\n\nTEST 5: SMALLER BATCH SIZE")
    train_loader_small, test_loader_small = create_dataloaders(
        'esc50', './datasets/ESC-50-master',
        fold=1, batch_size=16, num_workers=0,
        augment=False, mixup=False
    )
    test_scenario("No Mixup + Small Batch (16) + AdamW", train_loader_small, test_loader_small,
                  epochs=20, lr=3e-4, optimizer='adamw')

    print("\n\n" + "="*80)
    print("DIAGNOSIS COMPLETE")
    print("="*80)
    print("\nCheck which scenario shows the best learning.")
    print("If ALL scenarios fail, there's a fundamental model/data issue.")
    print("If some work, we can identify the right hyperparameters.")

if __name__ == "__main__":
    main()

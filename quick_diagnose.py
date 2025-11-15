#!/usr/bin/env python3
"""
Quick 5-epoch diagnostic to rapidly identify learning issues.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from acdnet_model import create_acdnet
from dataset import create_dataloaders

def quick_test(name, train_loader, test_loader, **kwargs):
    """Quick 5-epoch test."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")

    model = create_acdnet(num_classes=50, input_length=30225, sample_rate=20000)
    model = model.cuda()

    lr = kwargs.get('lr', 3e-4)
    optimizer_type = kwargs.get('optimizer', 'adamw')

    if optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    print(f"Optimizer: {optimizer_type}, LR: {lr}")

    losses = []
    for epoch in range(1, 6):
        model.train()
        epoch_loss = 0.0

        for audio, labels in train_loader:
            audio = audio.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(audio)

            if labels.dim() > 1:
                log_pred = torch.log_softmax(outputs, dim=1)
                target_dist = torch.softmax(labels, dim=1) if labels.dtype == torch.float else labels
                loss = nn.functional.kl_div(log_pred, target_dist, reduction='batchmean')
            else:
                loss = nn.functional.cross_entropy(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

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
    loss_change = losses[0] - losses[-1]

    print(f"\nRESULT:")
    print(f"  Loss: {losses[0]:.4f} → {losses[-1]:.4f} (change: {loss_change:.4f})")
    print(f"  Test acc: {test_acc:.2f}%")

    if loss_change > 0.05 and test_acc > 5.0:
        print(f"  ✓ LEARNING!")
    elif loss_change < 0.02:
        print(f"  ❌ NOT LEARNING (loss barely changing)")
    else:
        print(f"  ⚠️  WEAK LEARNING")

    return losses, test_acc

def main():
    print("="*60)
    print("QUICK DIAGNOSTIC (5 epochs per test)")
    print("="*60)

    # Test 1: No mixup baseline
    print("\n\nTEST 1: NO MIXUP + AdamW (LR=3e-4)")
    train_loader, test_loader = create_dataloaders(
        'esc50', './datasets/ESC-50-master',
        fold=1, batch_size=64, num_workers=0,
        augment=False, mixup=False
    )
    quick_test("No Mixup Baseline", train_loader, test_loader,
               lr=3e-4, optimizer='adamw')

    # Test 2: Higher LR
    print("\n\nTEST 2: NO MIXUP + AdamW (LR=1e-3)")
    quick_test("Higher LR", train_loader, test_loader,
               lr=1e-3, optimizer='adamw')

    # Test 3: With mixup
    print("\n\nTEST 3: WITH MIXUP + AdamW (LR=3e-4)")
    train_loader_mixup, _ = create_dataloaders(
        'esc50', './datasets/ESC-50-master',
        fold=1, batch_size=64, num_workers=0,
        augment=False, mixup=True
    )
    quick_test("With Mixup", train_loader_mixup, test_loader,
               lr=3e-4, optimizer='adamw')

    print("\n\n" + "="*60)
    print("QUICK DIAGNOSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()

"""
Training script for ACDNet and Micro-ACDNet
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime

from acdnet_model import create_acdnet
from acdnet_micro import create_micro_acdnet
from acdnet_dataset import create_dataloaders


class KLDivLossWithLogits(nn.Module):
    """
    KL Divergence Loss for mixup training.
    Combines log_softmax with KLDivLoss for numerical stability.
    """
    
    def __init__(self):
        super().__init__()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, pred_logits: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_logits: Model predictions (logits), shape (B, num_classes)
            target_dist: Target distribution, shape (B, num_classes)
        """
        log_pred = torch.log_softmax(pred_logits, dim=1)
        return self.kl_div(log_pred, target_dist)


class Trainer:
    """
    Trainer for ACDNet models.
    
    Implements the training procedure from the paper:
    - 2000 epochs
    - Initial LR: 0.1
    - LR decay at epochs [600, 1200, 1800] by factor of 10
    - Warm-up: first 10 epochs with 0.1x LR
    - SGD with Nesterov momentum 0.9
    - Weight decay: 5e-4
    - Batch size: 64
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
        output_dir: Path,
        num_epochs: int = 2000,
        initial_lr: float = 0.1,
        weight_decay: float = 5e-4,
        momentum: float = 0.9,
        mixup: bool = True
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_epochs = num_epochs
        self.initial_lr = initial_lr
        self.mixup = mixup
        
        # Setup optimizer
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=initial_lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True
        )
        
        # Setup learning rate scheduler
        # Milestones at [600, 1200, 1800], decay by factor of 10
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[600, 1200, 1800],
            gamma=0.1
        )
        
        # Loss function
        if mixup:
            self.criterion = KLDivLossWithLogits()
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Tensorboard
        self.writer = SummaryWriter(self.output_dir / 'logs')
        
        # Tracking
        self.best_acc = 0.0
        self.best_epoch = 0
        self.global_step = 0
        
    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Warm-up: first 10 epochs with 0.1x learning rate
        if epoch < 10:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.initial_lr * 0.1
        elif epoch == 10:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.initial_lr
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.num_epochs}')
        
        for batch_idx, (audio, labels) in enumerate(pbar):
            audio = audio.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(audio)
            
            # Calculate loss
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            
            # Calculate accuracy
            if self.mixup:
                # For mixup, take argmax of both pred and target
                pred = outputs.argmax(dim=1)
                target = labels.argmax(dim=1)
            else:
                pred = outputs.argmax(dim=1)
                target = labels
            
            correct += (pred == target).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # Log to tensorboard
            self.writer.add_scalar('train/loss', loss.item(), self.global_step)
            self.writer.add_scalar('train/acc', 100. * correct / total, self.global_step)
            self.global_step += 1
        
        avg_loss = total_loss / len(self.train_loader)
        acc = 100. * correct / total
        
        return {
            'loss': avg_loss,
            'acc': acc
        }
    
    @torch.no_grad()
    def evaluate(self) -> dict:
        """Evaluate on test set."""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for audio, labels in tqdm(self.test_loader, desc='Evaluating'):
            audio = audio.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(audio)
            
            # Calculate loss (always use CrossEntropyLoss for evaluation)
            if self.mixup:
                # For mixup training, labels might be soft, convert to hard labels
                if labels.dim() > 1:
                    labels = labels.argmax(dim=1)
            
            loss = nn.functional.cross_entropy(outputs, labels)
            
            # Statistics
            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(self.test_loader)
        acc = 100. * correct / total
        
        return {
            'loss': avg_loss,
            'acc': acc
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_acc': self.best_acc,
            'best_epoch': self.best_epoch
        }
        
        # Save latest checkpoint
        checkpoint_path = self.output_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model with accuracy {self.best_acc:.2f}%")
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.num_epochs} epochs")
        print(f"Output directory: {self.output_dir}")
        print(f"Device: {self.device}")
        print(f"Mixup: {self.mixup}")
        
        # Training loop
        for epoch in range(1, self.num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Evaluate
            test_metrics = self.evaluate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Log to tensorboard
            self.writer.add_scalar('epoch/train_loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('epoch/train_acc', train_metrics['acc'], epoch)
            self.writer.add_scalar('epoch/test_loss', test_metrics['loss'], epoch)
            self.writer.add_scalar('epoch/test_acc', test_metrics['acc'], epoch)
            self.writer.add_scalar('epoch/lr', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print summary
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.2f}%")
            print(f"Test  - Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['acc']:.2f}%")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            is_best = test_metrics['acc'] > self.best_acc
            if is_best:
                self.best_acc = test_metrics['acc']
                self.best_epoch = epoch
            
            # Save every 100 epochs and at the end
            if epoch % 100 == 0 or epoch == self.num_epochs or is_best:
                self.save_checkpoint(epoch, is_best)
        
        print(f"\nTraining completed!")
        print(f"Best accuracy: {self.best_acc:.2f}% at epoch {self.best_epoch}")
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train ACDNet models')
    
    # Model
    parser.add_argument('--model', type=str, default='acdnet', choices=['acdnet', 'micro'],
                        help='Model architecture')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='esc50',
                        choices=['esc50', 'esc10', 'urbansound8k'],
                        help='Dataset name')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to dataset root directory')
    parser.add_argument('--fold', type=int, default=1,
                        help='Fold number for cross-validation')
    
    # Training
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    
    # Augmentation
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--no-mixup', action='store_true',
                        help='Disable mixup augmentation')
    
    # Audio
    parser.add_argument('--target-length', type=int, default=30225,
                        help='Target audio length in samples (~1.51s @ 20kHz)')
    parser.add_argument('--sample-rate', type=int, default=20000,
                        help='Target sample rate')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./checkpoints',
                        help='Output directory for checkpoints')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of dataloader workers')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    print(f"Loading {args.dataset} dataset from {args.data_root}")
    train_loader, test_loader = create_dataloaders(
        dataset_name=args.dataset,
        root=args.data_root,
        fold=args.fold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_length=args.target_length,
        target_sr=args.sample_rate,
        augment=not args.no_augment,
        mixup=not args.no_mixup
    )
    
    # Determine number of classes
    num_classes = len(train_loader.dataset.class_names)
    print(f"Number of classes: {num_classes}")
    
    # Create model
    if args.model == 'acdnet':
        model = create_acdnet(
            num_classes=num_classes,
            input_length=args.target_length,
            sample_rate=args.sample_rate
        )
    else:
        model = create_micro_acdnet(
            num_classes=num_classes,
            input_length=args.target_length,
            sample_rate=args.sample_rate
        )
    
    print(f"\nModel: {args.model}")
    print(f"Parameters: {model.get_num_parameters():,}")
    print(f"Model size: {(model.get_num_parameters() * 4) / (1024**2):.2f} MB")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.model}_{args.dataset}_fold{args.fold}_{timestamp}"
    
    # Save config
    config = vars(args)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        output_dir=output_dir,
        num_epochs=args.epochs,
        initial_lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        mixup=not args.no_mixup
    )
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()

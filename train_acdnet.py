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
from dataset import create_dataloaders


class KLDivLossWithLogits(nn.Module):
    """
    KL Divergence Loss for mixup training.
    Combines log_softmax with KLDivLoss for numerical stability.
    """

    def __init__(self):
        super().__init__()
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

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

    Uses modern optimization techniques for stable training:
    - 2000 epochs
    - AdamW optimizer with LR 3e-4
    - Cosine annealing LR schedule (smooth decay from 3e-4 to 3e-6)
    - Gradient clipping (max_norm=1.0) for stability
    - Weight decay: 1e-4
    - Batch size: 64
    - Mixup augmentation with soft labels
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
        output_dir: Path,
        log_dir: Path,
        num_epochs: int = 2000,
        initial_lr: float = 0.1,
        weight_decay: float = 5e-4,
        mixup: bool = True,
        accumulation_steps: int = 1,
        optimizer_type: str = "sgd",
        momentum: float = 0.9,
        lr_scheduler: str = "multistep",
        lr_milestones: list = None,
        lr_gamma: float = 0.1,
        lr_min_factor: float = 0.01,
        lr_t_max: int = None,
        lr_patience: int = 50,
        lr_factor: float = 0.1,
        warmup_epochs: int = 10,
        warmup_factor: float = 0.1,
        save_interval: int = 250,
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
        self.accumulation_steps = accumulation_steps
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        self.save_interval = save_interval
        self.lr_milestones = lr_milestones or [600, 1200, 1800]

        # Setup optimizer
        if optimizer_type == "sgd":
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=initial_lr,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=True
            )
        elif optimizer_type == "adamw":
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=initial_lr,
                betas=(0.9, 0.999),
                weight_decay=weight_decay,
                eps=1e-8
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        # Setup learning rate scheduler
        if lr_scheduler == "multistep":
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.lr_milestones,
                gamma=lr_gamma
            )
        elif lr_scheduler == "cosine":
            t_max = lr_t_max if lr_t_max is not None else num_epochs
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=t_max,
                eta_min=initial_lr * lr_min_factor
            )
        elif lr_scheduler == "exponential":
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=lr_gamma
            )
        elif lr_scheduler == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',  # Maximize accuracy
                factor=lr_factor,
                patience=lr_patience,
                verbose=True
            )
        elif lr_scheduler == "none":
            self.scheduler = None
        else:
            raise ValueError(f"Unknown LR scheduler: {lr_scheduler}")

        self.lr_scheduler_type = lr_scheduler

        # Loss function
        if mixup:
            self.criterion = KLDivLossWithLogits()
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Tensorboard
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

        # Tracking
        self.best_acc = 0.0
        self.best_epoch = 0
        self.global_step = 0
        self.start_epoch = 1  # Can be updated when resuming from checkpoint

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs}")

        for batch_idx, (audio, labels) in enumerate(pbar):
            audio = audio.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(audio)

            # Calculate loss
            loss = self.criterion(outputs, labels)

            # Scale loss by accumulation steps
            loss = loss / self.accumulation_steps

            # Backward pass (accumulates gradients)
            loss.backward()

            # Only step optimizer every accumulation_steps
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Gradient clipping to prevent instability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Statistics (use unscaled loss for reporting)
            total_loss += loss.item() * self.accumulation_steps

            # Calculate accuracy
            # NOTE: With mixup, this compares argmax(pred) vs argmax(soft_label)
            # which isn't truly meaningful, but gives a rough signal
            pred = outputs.argmax(dim=1)
            if self.mixup:
                # For mixup, compare with dominant class in soft label
                target = labels.argmax(dim=1)
            else:
                target = labels

            correct += (pred == target).sum().item()
            total += labels.size(0)

            # Update progress bar with accuracy and lr
            pbar.set_postfix(
                {
                    "loss": total_loss / (batch_idx + 1),
                    "acc": 100.0 * correct / total,
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
            )

            # Log to tensorboard (every accumulation cycle)
            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.writer.add_scalar("train/loss", total_loss / (batch_idx + 1), self.global_step)
                self.writer.add_scalar("train/acc", 100.0 * correct / total, self.global_step)
                self.global_step += 1

        avg_loss = total_loss / len(self.train_loader)
        acc = 100.0 * correct / total

        return {"loss": avg_loss, "acc": acc}

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Evaluate on test set."""
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for audio, labels in self.test_loader:
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
        acc = 100.0 * correct / total

        return {"loss": avg_loss, "acc": acc}

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_acc": self.best_acc,
            "best_epoch": self.best_epoch,
        }

        # Save latest checkpoint
        checkpoint_path = self.output_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume training."""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model and optimizer states
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Fix T_max mismatch if resuming with different number of epochs
        # This prevents the learning rate from cycling back up after reaching the original T_max
        if self.scheduler.T_max != self.num_epochs:
            print(f"⚠️  WARNING: Checkpoint scheduler has T_max={self.scheduler.T_max}, but training with {self.num_epochs} epochs")
            print(f"   Adjusting scheduler T_max from {self.scheduler.T_max} to {self.num_epochs}")
            self.scheduler.T_max = self.num_epochs
            # Also update eta_min to match the current initial_lr
            self.scheduler.eta_min = self.initial_lr * 0.01

        # Restore training state
        self.start_epoch = checkpoint["epoch"] + 1  # Start from next epoch
        self.best_acc = checkpoint["best_acc"]
        self.best_epoch = checkpoint["best_epoch"]

        print(f"Resumed from epoch {checkpoint['epoch']}")
        print(f"Best accuracy so far: {self.best_acc:.2f}% at epoch {self.best_epoch}")
        print(f"Will continue training from epoch {self.start_epoch} to {self.num_epochs}")
        print(f"Current learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")

    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.num_epochs} epochs")
        print(f"Output directory: {self.output_dir}")
        print(f"Device: {self.device}")
        print(f"Mixup: {self.mixup}")

        # Training loop
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)

            # Evaluate
            test_metrics = self.evaluate()

            # Update learning rate (with warmup support)
            if epoch <= self.warmup_epochs and self.warmup_epochs > 0:
                # Warmup phase: use reduced LR
                warmup_lr = self.initial_lr * self.warmup_factor
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            else:
                # Normal training: use scheduler
                if self.scheduler is not None:
                    if self.lr_scheduler_type == "plateau":
                        # ReduceLROnPlateau needs metrics
                        self.scheduler.step(test_metrics["acc"])
                    else:
                        self.scheduler.step()

            # Log to tensorboard
            self.writer.add_scalar("epoch/train_loss", train_metrics["loss"], epoch)
            self.writer.add_scalar("epoch/train_acc", train_metrics["acc"], epoch)
            self.writer.add_scalar("epoch/test_loss", test_metrics["loss"], epoch)
            self.writer.add_scalar("epoch/test_acc", test_metrics["acc"], epoch)
            self.writer.add_scalar("epoch/lr", self.optimizer.param_groups[0]["lr"], epoch)

            # Save checkpoint
            is_best = test_metrics["acc"] > self.best_acc
            if is_best:
                self.best_acc = test_metrics["acc"]
                self.best_epoch = epoch

            # Save checkpoints at configured interval, at the end, or when best
            if epoch % self.save_interval == 0 or epoch == self.num_epochs or is_best:
                self.save_checkpoint(epoch, is_best)

        print(f"\nTraining completed!")
        print(f"Best accuracy: {self.best_acc:.2f}% at epoch {self.best_epoch}")

        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train ACDNet models")

    # Model
    parser.add_argument("--model", type=str, default="acdnet", choices=["acdnet", "micro"], help="Model architecture")

    # Dataset
    parser.add_argument(
        "--dataset", type=str, default="esc50", choices=["esc50", "esc10", "urbansound8k"], help="Dataset name"
    )
    parser.add_argument("--data-root", type=str, required=True, help="Path to dataset root directory")
    parser.add_argument("--fold", type=int, default=1, help="Fold number for cross-validation")

    # Training
    parser.add_argument("--epochs", type=int, default=2000, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (effective_batch_size = batch_size * accumulation_steps)",
    )

    # Optimizer
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adamw"],
                       help="Optimizer (default: sgd as in paper)")
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate (default: 0.1 for SGD)")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay (default: 5e-4 for SGD)")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD (default: 0.9)")

    # Learning Rate Scheduler
    parser.add_argument("--lr-scheduler", type=str, default="multistep",
                       choices=["multistep", "cosine", "exponential", "plateau", "none"],
                       help="LR scheduler (default: multistep as in paper)")
    parser.add_argument("--lr-milestones", type=int, nargs="+", default=[600, 1200, 1800],
                       help="Epochs at which to decay LR for multistep (default: [600, 1200, 1800])")
    parser.add_argument("--lr-gamma", type=float, default=0.1,
                       help="LR decay factor for multistep/exponential (default: 0.1)")
    parser.add_argument("--lr-min-factor", type=float, default=0.01,
                       help="Min LR as fraction of initial LR for cosine (default: 0.01)")
    parser.add_argument("--lr-t-max", type=int, default=None,
                       help="T_max for cosine scheduler (default: num_epochs)")
    parser.add_argument("--lr-patience", type=int, default=50,
                       help="Patience for plateau scheduler (default: 50)")
    parser.add_argument("--lr-factor", type=float, default=0.1,
                       help="Reduction factor for plateau scheduler (default: 0.1)")

    # Warmup
    parser.add_argument("--warmup-epochs", type=int, default=10,
                       help="Number of warmup epochs (default: 10 as in paper, 0 to disable)")
    parser.add_argument("--warmup-factor", type=float, default=0.1,
                       help="LR multiplier during warmup (default: 0.1)")

    # Checkpoints
    parser.add_argument("--save-interval", type=int, default=250,
                       help="Save checkpoint every N epochs (default: 250)")

    # Augmentation
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    parser.add_argument("--no-mixup", action="store_true", help="Disable mixup augmentation")

    # Audio
    parser.add_argument(
        "--target-length", type=int, default=30225, help="Target audio length in samples (~1.51s @ 20kHz)"
    )
    parser.add_argument("--sample-rate", type=int, default=20000, help="Target sample rate")

    # Output
    parser.add_argument("--output-dir", type=str, default="./checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Tensorboard log directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    # Device
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers")

    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
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
        mixup=not args.no_mixup,
    )

    # Determine number of classes
    num_classes = len(train_loader.dataset.class_names)
    print(f"Number of classes: {num_classes}")

    # Create model
    if args.model == "acdnet":
        model = create_acdnet(num_classes=num_classes, input_length=args.target_length, sample_rate=args.sample_rate)
    else:
        model = create_micro_acdnet(
            num_classes=num_classes, input_length=args.target_length, sample_rate=args.sample_rate
        )

    print(f"\nModel: {args.model}")
    print(f"Parameters: {model.get_num_parameters():,}")
    print(f"Model size: {(model.get_num_parameters() * 4) / (1024**2):.2f} MB")

    # Create output and log directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.model}_{args.dataset}_fold{args.fold}_{timestamp}"
    output_dir = Path(args.output_dir) / run_name
    log_dir = Path(args.log_dir) / run_name

    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint directory: {output_dir}")
    print(f"Log directory: {log_dir}")

    # Save config
    config = vars(args)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        output_dir=output_dir,
        log_dir=log_dir,
        num_epochs=args.epochs,
        initial_lr=args.lr,
        weight_decay=args.weight_decay,
        mixup=not args.no_mixup,
        accumulation_steps=args.accumulation_steps,
        optimizer_type=args.optimizer,
        momentum=args.momentum,
        lr_scheduler=args.lr_scheduler,
        lr_milestones=args.lr_milestones,
        lr_gamma=args.lr_gamma,
        lr_min_factor=args.lr_min_factor,
        lr_t_max=args.lr_t_max,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        warmup_epochs=args.warmup_epochs,
        warmup_factor=args.warmup_factor,
        save_interval=args.save_interval,
    )

    # Load checkpoint if resuming
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()

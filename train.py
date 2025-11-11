"""
Training script for SoundStream
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
from tqdm import tqdm
import random
import numpy as np
from datetime import datetime
import json
import socket
import subprocess
import sys

from model import SoundStream
from discriminator import CombinedDiscriminator
from losses import GeneratorLoss, DiscriminatorLoss
from dataset import create_dataloader


def get_git_commit_hash():
    """Get current git commit hash, or None if not in a git repo"""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5,
            check=False
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def collect_metadata():
    """Collect system and environment metadata"""
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'hostname': socket.gethostname(),
        'python_version': sys.version,
        'pytorch_version': torch.__version__,
    }

    # Add CUDA version if available
    if torch.cuda.is_available():
        metadata['cuda_version'] = torch.version.cuda
        metadata['cudnn_version'] = torch.backends.cudnn.version()

    # Add git commit hash if available
    git_hash = get_git_commit_hash()
    if git_hash:
        metadata['git_commit'] = git_hash

    return metadata


def save_config_json(config_dict, run_name, checkpoint_dir):
    """Save training configuration to JSON file"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config_path = checkpoint_dir / f'{run_name}_config.json'

    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

    print(f"Saved config to: {config_path}")


class SoundStreamTrainer:
    """Trainer for SoundStream"""
    def __init__(
        self,
        model: SoundStream,
        discriminator: CombinedDiscriminator,
        train_loader,
        config: dict,
        run_name: str,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.discriminator = discriminator.to(device)
        self.train_loader = train_loader
        self.config = config
        self.run_name = run_name
        self.device = device

        # Optimizers
        self.g_optimizer = optim.AdamW(
            model.parameters(),
            lr=config['g_lr'],
            betas=(0.8, 0.9)
        )

        self.d_optimizer = optim.AdamW(
            discriminator.parameters(),
            lr=config['d_lr'],
            betas=(0.8, 0.9)
        )

        # Learning rate schedulers (exponential decay)
        self.g_scheduler = optim.lr_scheduler.ExponentialLR(
            self.g_optimizer,
            gamma=config.get('lr_decay', 0.999996)
        )

        self.d_scheduler = optim.lr_scheduler.ExponentialLR(
            self.d_optimizer,
            gamma=config.get('lr_decay', 0.999996)
        )

        # Loss functions
        self.gen_loss_fn = GeneratorLoss(
            sample_rate=config['sample_rate'],
            lambda_adv=config.get('lambda_adv', 1.0),
            lambda_feat=config.get('lambda_feat', 100.0),
            lambda_rec=config.get('lambda_rec', 1.0)
        ).to(device)

        self.disc_loss_fn = DiscriminatorLoss().to(device)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=config['log_dir'])

        # Training state
        self.global_step = 0
        self.epoch = 0

        # Discriminator warmup (start training discriminator after N steps)
        self.disc_warmup_steps = config.get('disc_warmup_steps', 0)

    def train_step(self, batch):
        """Single training step"""
        real_audio = batch.to(self.device)  # (batch, 1, time)
        batch_size = real_audio.shape[0]

        # Sample number of quantizers for quantizer dropout
        if self.model.training and random.random() < 0.5:  # 50% chance to use dropout
            num_quantizers = random.randint(1, self.model.num_quantizers)
        else:
            num_quantizers = self.model.num_quantizers

        # ============ Train Generator ============
        self.g_optimizer.zero_grad()

        # Forward pass through generator
        fake_audio, indices, commitment_loss = self.model(real_audio, num_quantizers)

        # Get discriminator outputs
        with torch.no_grad():
            disc_outputs = self.discriminator(real_audio, fake_audio.detach())

        # Generator wants to fool discriminator
        disc_outputs_for_gen = self.discriminator(real_audio, fake_audio)

        # Combine all logits and features
        fake_logits_list = (
            disc_outputs_for_gen['fake_wave_logits'] +
            [disc_outputs_for_gen['fake_stft_logits']]
        )
        real_features_list = (
            disc_outputs_for_gen['real_wave_features'] +
            [disc_outputs_for_gen['real_stft_features']]
        )
        fake_features_list = (
            disc_outputs_for_gen['fake_wave_features'] +
            [disc_outputs_for_gen['fake_stft_features']]
        )

        # Compute generator loss
        gen_loss, gen_loss_dict = self.gen_loss_fn(
            fake_logits_list,
            real_features_list,
            fake_features_list,
            fake_audio,
            real_audio
        )

        # Add commitment loss
        total_gen_loss = gen_loss + commitment_loss

        # Backward and optimize
        total_gen_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.g_optimizer.step()
        self.g_scheduler.step()

        # ============ Train Discriminator ============
        d_loss_value = 0.0

        # Only train discriminator after warmup period
        if self.global_step >= self.disc_warmup_steps:
            self.d_optimizer.zero_grad()

            # Get fresh discriminator outputs
            disc_outputs = self.discriminator(real_audio, fake_audio.detach())

            # Combine all logits
            real_logits_list = (
                disc_outputs['real_wave_logits'] +
                [disc_outputs['real_stft_logits']]
            )
            fake_logits_list = (
                disc_outputs['fake_wave_logits'] +
                [disc_outputs['fake_stft_logits']]
            )

            # Compute discriminator loss
            d_loss = self.disc_loss_fn(real_logits_list, fake_logits_list)

            # Backward and optimize
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)

            self.d_optimizer.step()
            self.d_scheduler.step()

            d_loss_value = d_loss.item()

        # Collect losses
        losses = {
            'g_total': total_gen_loss.item(),
            'g_adv': gen_loss_dict['adv_loss'],
            'g_feat': gen_loss_dict['feat_loss'],
            'g_rec': gen_loss_dict['rec_loss'],
            'commitment': commitment_loss.item(),
            'd_loss': d_loss_value,
            'num_quantizers': num_quantizers
        }

        return losses

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        self.discriminator.train()

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')

        for batch in pbar:
            losses = self.train_step(batch)

            # Log to tensorboard
            for k, v in losses.items():
                self.writer.add_scalar(f'train/{k}', v, self.global_step)

            # Update progress bar
            pbar.set_postfix({
                'g_loss': f"{losses['g_total']:.3f}",
                'd_loss': f"{losses['d_loss']:.3f}",
                'rec': f"{losses['g_rec']:.3f}"
            })

            self.global_step += 1

            # Save checkpoint periodically
            if self.global_step % self.config['save_interval'] == 0:
                self.save_checkpoint()

        self.epoch += 1

    def save_checkpoint(self):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'run_name': self.run_name,
            'model_state_dict': self.model.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_scheduler_state_dict': self.g_scheduler.state_dict(),
            'd_scheduler_state_dict': self.d_scheduler.state_dict(),
            'config': self.config
        }

        checkpoint_path = checkpoint_dir / f'soundstream_{self.run_name}_step_{self.global_step}.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"\nCheckpoint saved: {checkpoint_path}")

        # Keep only last N checkpoints for this run
        checkpoints = sorted(checkpoint_dir.glob(f'soundstream_{self.run_name}_step_*.pt'))
        if len(checkpoints) > self.config.get('max_checkpoints', 5):
            for old_ckpt in checkpoints[:-self.config.get('max_checkpoints', 5)]:
                old_ckpt.unlink()

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        self.g_scheduler.load_state_dict(checkpoint['g_scheduler_state_dict'])
        self.d_scheduler.load_state_dict(checkpoint['d_scheduler_state_dict'])

        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']

        # Restore run_name if available (for backwards compatibility with old checkpoints)
        if 'run_name' in checkpoint:
            self.run_name = checkpoint['run_name']
            print(f"Loaded checkpoint from run '{self.run_name}' at step {self.global_step}")
        else:
            print(f"Loaded checkpoint from step {self.global_step} (no run_name in checkpoint)")

    @torch.no_grad()
    def test_reconstruction(self, audio_path: str, output_path: str):
        """Test reconstruction on a single audio file"""
        import torchaudio

        self.model.eval()

        # Load audio
        waveform, sr = torchaudio.load(audio_path)

        # Resample if needed
        if sr != self.config['sample_rate']:
            resampler = torchaudio.transforms.Resample(sr, self.config['sample_rate'])
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Add batch dimension and move to device
        waveform = waveform.unsqueeze(0).to(self.device)

        # Encode and decode
        indices = self.model.encode(waveform)
        reconstructed = self.model.decode(indices)

        # Save reconstructed audio
        reconstructed = reconstructed.cpu()
        torchaudio.save(output_path, reconstructed[0], self.config['sample_rate'])

        print(f"Saved reconstruction to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train SoundStream or TinyStream')
    parser.add_argument('--audio_dir', type=str, required=True, help='Directory containing audio files')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Tensorboard log directory')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')

    # Model selection
    parser.add_argument('--model', type=str, default='soundstream', choices=['soundstream', 'tinystream'],
                       help='Model to train: soundstream (full) or tinystream (ESP32)')

    # Model config
    parser.add_argument('--C', type=int, default=32, help='Base number of channels')
    parser.add_argument('--D', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--num_quantizers', type=int, default=8, help='Number of quantizers')
    parser.add_argument('--codebook_size', type=int, default=1024, help='Codebook size')
    parser.add_argument('--sample_rate', type=int, default=24000, help='Audio sample rate')

    # Training config
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--audio_length', type=float, default=2.0, help='Audio length in seconds')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--g_lr', type=float, default=1e-4, help='Generator learning rate')
    parser.add_argument('--d_lr', type=float, default=1e-4, help='Discriminator learning rate')
    parser.add_argument('--disc_warmup_steps', type=int, default=0, help='Discriminator warmup steps')
    parser.add_argument('--save_interval', type=int, default=5000, help='Save checkpoint every N steps')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Generate run name (timestamp)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Run name: {run_name}")

    # Create config dict with run_name in log_dir
    log_dir_with_run = str(Path(args.log_dir) / run_name)
    config = {
        'sample_rate': args.sample_rate,
        'g_lr': args.g_lr,
        'd_lr': args.d_lr,
        'disc_warmup_steps': args.disc_warmup_steps,
        'save_interval': args.save_interval,
        'checkpoint_dir': args.checkpoint_dir,
        'log_dir': log_dir_with_run,
        'lambda_adv': 1.0,
        'lambda_feat': 100.0,
        'lambda_rec': 1.0,
        'lr_decay': 0.999996,
        'max_checkpoints': 5
    }

    # Create model
    print(f"Creating model: {args.model.upper()}...")
    if args.model == 'tinystream':
        from model_tiny import TinyStream
        model = TinyStream(
            C=args.C,
            D=args.D,
            strides=[4, 4, 4, 4],  # 256x downsampling for ESP32
            num_quantizers=args.num_quantizers,
            codebook_size=args.codebook_size,
            sample_rate=args.sample_rate
        )
        print("Note: TinyStream uses fixed number of quantizers (no dropout)")
        print(f"Encoder+Quantizer for ESP32: {model.get_model_size_mb():.3f} MB")
    else:
        model = SoundStream(
            C=args.C,
            D=args.D,
            strides=[2, 4, 5, 8],  # 320x downsampling
            num_quantizers=args.num_quantizers,
            codebook_size=args.codebook_size,
            sample_rate=args.sample_rate
        )
    print(f"Total model parameters: {model.get_num_params():,}")

    # Create discriminator
    discriminator = CombinedDiscriminator()

    # Create data loader
    print("Creating data loader...")
    train_loader = create_dataloader(
        audio_dir=args.audio_dir,
        batch_size=args.batch_size,
        sample_rate=args.sample_rate,
        audio_length_seconds=args.audio_length,
        num_workers=args.num_workers
    )
    print(f"Training batches per epoch: {len(train_loader)}")

    # Collect metadata and save full config to JSON
    metadata = collect_metadata()
    full_config = {
        'run_name': run_name,
        'metadata': metadata,
        'args': vars(args),
        'config': config,
        'model_params': model.get_num_params(),
        'device': device
    }
    save_config_json(full_config, run_name, args.checkpoint_dir)

    # Create trainer
    trainer = SoundStreamTrainer(
        model=model,
        discriminator=discriminator,
        train_loader=train_loader,
        config=config,
        run_name=run_name,
        device=device
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Training loop
    print(f"\nStarting training for {args.num_epochs} epochs...")
    try:
        for epoch in range(args.num_epochs):
            trainer.train_epoch()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    # Save final checkpoint
    trainer.save_checkpoint()
    print("Training complete!")


if __name__ == "__main__":
    main()

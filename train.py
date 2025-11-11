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

from model import SoundStream
from discriminator import CombinedDiscriminator
from losses import GeneratorLoss, DiscriminatorLoss
from dataset import create_dataloader


class SoundStreamTrainer:
    """Trainer for SoundStream"""
    def __init__(
        self,
        model: SoundStream,
        discriminator: CombinedDiscriminator,
        train_loader,
        config: dict,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.discriminator = discriminator.to(device)
        self.train_loader = train_loader
        self.config = config
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
        if self.training and random.random() < 0.5:  # 50% chance to use dropout
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
            'model_state_dict': self.model.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_scheduler_state_dict': self.g_scheduler.state_dict(),
            'd_scheduler_state_dict': self.d_scheduler.state_dict(),
            'config': self.config
        }

        checkpoint_path = checkpoint_dir / f'soundstream_step_{self.global_step}.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"\nCheckpoint saved: {checkpoint_path}")

        # Keep only last N checkpoints
        checkpoints = sorted(checkpoint_dir.glob('soundstream_step_*.pt'))
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

        print(f"Loaded checkpoint from step {self.global_step}")

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
    parser = argparse.ArgumentParser(description='Train SoundStream')
    parser.add_argument('--audio_dir', type=str, required=True, help='Directory containing audio files')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Tensorboard log directory')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')

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

    # Create config dict
    config = {
        'sample_rate': args.sample_rate,
        'g_lr': args.g_lr,
        'd_lr': args.d_lr,
        'disc_warmup_steps': args.disc_warmup_steps,
        'save_interval': args.save_interval,
        'checkpoint_dir': args.checkpoint_dir,
        'log_dir': args.log_dir,
        'lambda_adv': 1.0,
        'lambda_feat': 100.0,
        'lambda_rec': 1.0,
        'lr_decay': 0.999996,
        'max_checkpoints': 5
    }

    # Create model
    print("Creating model...")
    model = SoundStream(
        C=args.C,
        D=args.D,
        strides=[2, 4, 5, 8],
        num_quantizers=args.num_quantizers,
        codebook_size=args.codebook_size,
        sample_rate=args.sample_rate
    )
    print(f"Model parameters: {model.get_num_params():,}")

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

    # Create trainer
    trainer = SoundStreamTrainer(
        model=model,
        discriminator=discriminator,
        train_loader=train_loader,
        config=config,
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

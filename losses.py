"""
Loss functions for SoundStream training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import torchaudio.transforms as T


class MultiScaleSpectralLoss(nn.Module):
    """Multi-scale mel-spectrogram reconstruction loss"""
    def __init__(
        self,
        sample_rate: int = 24000,
        window_lengths: List[int] = [2048, 1024, 512, 256, 128, 64],
        n_mels: int = 64,
        alpha: float = 1.0
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.alpha = alpha
        
        # Create mel-spectrogram transforms for each window length
        self.mel_specs = nn.ModuleList()
        for win_len in window_lengths:
            hop_len = win_len // 4
            self.mel_specs.append(
                T.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=win_len,
                    hop_length=hop_len,
                    n_mels=n_mels,
                    f_min=0.0,
                    f_max=sample_rate / 2.0
                )
            )
    
    def forward(self, pred, target):
        """
        Args:
            pred: (batch, 1, time) - predicted waveform
            target: (batch, 1, time) - target waveform
        Returns:
            loss: scalar
        """
        total_loss = 0
        
        pred = pred.squeeze(1)  # (batch, time)
        target = target.squeeze(1)  # (batch, time)
        
        for mel_spec in self.mel_specs:
            # Compute mel-spectrograms
            pred_mel = mel_spec(pred)
            target_mel = mel_spec(target)
            
            # L1 loss on magnitude
            mag_loss = F.l1_loss(pred_mel, target_mel)
            
            # L2 loss on log magnitude
            pred_log = torch.log(pred_mel + 1e-5)
            target_log = torch.log(target_mel + 1e-5)
            log_loss = F.mse_loss(pred_log, target_log)
            
            # Weighted combination
            scale_loss = mag_loss + self.alpha * log_loss
            total_loss += scale_loss
        
        return total_loss / len(self.mel_specs)


class GeneratorLoss(nn.Module):
    """Combined generator loss"""
    def __init__(
        self,
        sample_rate: int = 24000,
        lambda_adv: float = 1.0,
        lambda_feat: float = 100.0,
        lambda_rec: float = 1.0
    ):
        super().__init__()
        self.lambda_adv = lambda_adv
        self.lambda_feat = lambda_feat
        self.lambda_rec = lambda_rec
        
        self.spectral_loss = MultiScaleSpectralLoss(sample_rate=sample_rate)
    
    def forward(
        self,
        fake_logits_list: List[torch.Tensor],
        real_features_list: List[List[torch.Tensor]],
        fake_features_list: List[List[torch.Tensor]],
        pred_audio: torch.Tensor,
        target_audio: torch.Tensor
    ):
        """
        Args:
            fake_logits_list: list of discriminator logits for fake audio
            real_features_list: list of lists of discriminator features for real audio
            fake_features_list: list of lists of discriminator features for fake audio
            pred_audio: (batch, 1, time) - predicted audio
            target_audio: (batch, 1, time) - target audio
        Returns:
            total_loss: scalar
            loss_dict: dictionary of individual loss components
        """
        # Adversarial loss (generator wants fake to be classified as real)
        adv_loss = 0
        for fake_logits in fake_logits_list:
            adv_loss += torch.mean(F.relu(1 - fake_logits))
        adv_loss = adv_loss / len(fake_logits_list)
        
        # Feature matching loss
        feat_loss = 0
        count = 0
        for real_features, fake_features in zip(real_features_list, fake_features_list):
            for real_feat, fake_feat in zip(real_features, fake_features):
                feat_loss += F.l1_loss(real_feat, fake_feat)
                count += 1
        feat_loss = feat_loss / count if count > 0 else feat_loss
        
        # Spectral reconstruction loss
        rec_loss = self.spectral_loss(pred_audio, target_audio)
        
        # Combined loss
        total_loss = (
            self.lambda_adv * adv_loss +
            self.lambda_feat * feat_loss +
            self.lambda_rec * rec_loss
        )
        
        loss_dict = {
            'adv_loss': adv_loss.item(),
            'feat_loss': feat_loss.item(),
            'rec_loss': rec_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict


class DiscriminatorLoss(nn.Module):
    """Discriminator loss"""
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        real_logits_list: List[torch.Tensor],
        fake_logits_list: List[torch.Tensor]
    ):
        """
        Args:
            real_logits_list: list of discriminator logits for real audio
            fake_logits_list: list of discriminator logits for fake audio
        Returns:
            loss: scalar
        """
        loss = 0
        for real_logits, fake_logits in zip(real_logits_list, fake_logits_list):
            # Hinge loss: want real > 1, fake < -1
            loss += torch.mean(F.relu(1 - real_logits)) + torch.mean(F.relu(1 + fake_logits))
        return loss / len(real_logits_list)


if __name__ == "__main__":
    # Test losses
    batch_size = 2
    time_steps = 24000
    
    # Create dummy audio
    pred_audio = torch.randn(batch_size, 1, time_steps)
    target_audio = torch.randn(batch_size, 1, time_steps)
    
    # Create dummy discriminator outputs
    fake_logits_list = [torch.randn(batch_size, 1, 100) for _ in range(4)]
    real_logits_list = [torch.randn(batch_size, 1, 100) for _ in range(4)]
    
    # Dummy features
    real_features_list = [[torch.randn(batch_size, 64, 200) for _ in range(3)] for _ in range(4)]
    fake_features_list = [[torch.randn(batch_size, 64, 200) for _ in range(3)] for _ in range(4)]
    
    # Test generator loss
    gen_loss_fn = GeneratorLoss(sample_rate=24000)
    total_loss, loss_dict = gen_loss_fn(
        fake_logits_list,
        real_features_list,
        fake_features_list,
        pred_audio,
        target_audio
    )
    
    print("Generator losses:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")
    
    # Test discriminator loss
    disc_loss_fn = DiscriminatorLoss()
    disc_loss = disc_loss_fn(real_logits_list, fake_logits_list)
    print(f"\nDiscriminator loss: {disc_loss.item():.4f}")
    
    # Test spectral loss alone
    spectral_loss = MultiScaleSpectralLoss(sample_rate=24000)
    spec_loss = spectral_loss(pred_audio, target_audio)
    print(f"Spectral reconstruction loss: {spec_loss.item():.4f}")

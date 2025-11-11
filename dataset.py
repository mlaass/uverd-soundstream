"""
Audio dataset and data loading utilities
"""

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
from typing import Optional, List


class AudioDataset(Dataset):
    """Dataset for loading audio files"""
    def __init__(
        self,
        audio_dir: str,
        sample_rate: int = 24000,
        audio_length_seconds: float = 2.0,
        extensions: List[str] = ['.wav', '.mp3', '.flac', '.ogg']
    ):
        """
        Args:
            audio_dir: directory containing audio files
            sample_rate: target sample rate
            audio_length_seconds: length of audio chunks in seconds
            extensions: list of audio file extensions to include
        """
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.audio_length = int(audio_length_seconds * sample_rate)
        
        # Find all audio files
        self.audio_files = []
        for ext in extensions:
            self.audio_files.extend(list(self.audio_dir.rglob(f'*{ext}')))
        
        print(f"Found {len(self.audio_files)} audio files in {audio_dir}")
        
        if len(self.audio_files) == 0:
            raise ValueError(f"No audio files found in {audio_dir}")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        """Load and process audio file"""
        audio_path = self.audio_files[idx]
        
        try:
            # Load audio
            waveform, sr = torchaudio.load(audio_path)
            
            # Resample if necessary
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Get random chunk if audio is longer than target length
            if waveform.shape[1] > self.audio_length:
                start_idx = random.randint(0, waveform.shape[1] - self.audio_length)
                waveform = waveform[:, start_idx:start_idx + self.audio_length]
            
            # Pad if audio is shorter than target length
            elif waveform.shape[1] < self.audio_length:
                pad_amount = self.audio_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
            
            # Normalize to [-1, 1]
            if waveform.abs().max() > 0:
                waveform = waveform / waveform.abs().max()
            
            return waveform
        
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return random noise as fallback
            return torch.randn(1, self.audio_length)


def create_dataloader(
    audio_dir: str,
    batch_size: int,
    sample_rate: int = 24000,
    audio_length_seconds: float = 2.0,
    num_workers: int = 4,
    shuffle: bool = True
):
    """Create a data loader for audio files"""
    dataset = AudioDataset(
        audio_dir=audio_dir,
        sample_rate=sample_rate,
        audio_length_seconds=audio_length_seconds
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


class AudioNormalizer:
    """Normalize audio to have specific statistics"""
    @staticmethod
    def peak_normalize(audio, target_peak=0.95):
        """Normalize audio to target peak amplitude"""
        max_val = audio.abs().max()
        if max_val > 0:
            return audio * (target_peak / max_val)
        return audio
    
    @staticmethod
    def rms_normalize(audio, target_rms=0.1):
        """Normalize audio to target RMS"""
        rms = torch.sqrt(torch.mean(audio ** 2))
        if rms > 0:
            return audio * (target_rms / rms)
        return audio


# Data augmentation functions
class AudioAugmentation:
    """Simple audio augmentation"""
    @staticmethod
    def add_noise(audio, noise_level=0.005):
        """Add Gaussian noise"""
        noise = torch.randn_like(audio) * noise_level
        return audio + noise
    
    @staticmethod
    def time_stretch(audio, rate_range=(0.9, 1.1)):
        """Simple time stretching (naive resampling)"""
        rate = random.uniform(*rate_range)
        if rate == 1.0:
            return audio
        
        # This is a simple version - could use torchaudio's speed for better quality
        length = int(audio.shape[-1] * rate)
        return torch.nn.functional.interpolate(
            audio.unsqueeze(0),
            size=length,
            mode='linear',
            align_corners=False
        ).squeeze(0)
    
    @staticmethod
    def pitch_shift(audio, semitones_range=(-2, 2)):
        """Pitch shift (simple resampling - not true pitch shift)"""
        # Note: For real pitch shifting, use torchaudio.transforms.PitchShift
        # This is a placeholder
        return audio


if __name__ == "__main__":
    # Test the dataset
    import sys
    
    if len(sys.argv) > 1:
        audio_dir = sys.argv[1]
    else:
        audio_dir = "/path/to/audio/files"
        print(f"Usage: python dataset.py <audio_dir>")
        print(f"Using default: {audio_dir}")
    
    try:
        # Create dataset
        dataset = AudioDataset(
            audio_dir=audio_dir,
            sample_rate=24000,
            audio_length_seconds=2.0
        )
        
        print(f"\nDataset size: {len(dataset)}")
        
        # Test loading a sample
        audio = dataset[0]
        print(f"Audio shape: {audio.shape}")
        print(f"Audio range: [{audio.min():.3f}, {audio.max():.3f}]")
        
        # Create data loader
        dataloader = create_dataloader(
            audio_dir=audio_dir,
            batch_size=4,
            sample_rate=24000,
            audio_length_seconds=2.0,
            num_workers=2
        )
        
        # Test batch loading
        for batch in dataloader:
            print(f"\nBatch shape: {batch.shape}")
            print(f"Batch range: [{batch.min():.3f}, {batch.max():.3f}]")
            break
        
        print("\nDataset test passed!")
        
    except Exception as e:
        print(f"Dataset test failed: {e}")
        import traceback
        traceback.print_exc()

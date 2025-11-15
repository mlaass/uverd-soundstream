"""
Unified audio dataset and data loading utilities
Supports both unlabeled audio (codec training) and labeled audio (classification)
"""

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
from typing import Optional, List, Tuple, Union
import warnings

# Try to set torchaudio backend to avoid torchcodec dependency
try:
    torchaudio.set_audio_backend("soundfile")
except (AttributeError, RuntimeError):
    # Newer versions of torchaudio don't use set_audio_backend
    # Instead, they auto-select based on available backends
    pass


class AudioNormalizer:
    """Normalize audio to have specific statistics"""

    @staticmethod
    def peak_normalize(audio: torch.Tensor, target_peak: float = 0.95) -> torch.Tensor:
        """Normalize audio to target peak amplitude"""
        max_val = audio.abs().max()
        if max_val > 0:
            return audio * (target_peak / max_val)
        return audio

    @staticmethod
    def rms_normalize(audio: torch.Tensor, target_rms: float = 0.1) -> torch.Tensor:
        """Normalize audio to target RMS"""
        rms = torch.sqrt(torch.mean(audio ** 2))
        if rms > 0:
            return audio * (target_rms / rms)
        return audio

    @staticmethod
    def fixed_normalize(audio: torch.Tensor, divisor: float = 32768.0) -> torch.Tensor:
        """Normalize by fixed divisor (e.g., 16-bit audio range)"""
        return audio / divisor


class AudioAugmentation:
    """Audio augmentation utilities"""

    @staticmethod
    def add_noise(audio: torch.Tensor, noise_level: float = 0.005) -> torch.Tensor:
        """Add Gaussian noise"""
        noise = torch.randn_like(audio) * noise_level
        return audio + noise

    @staticmethod
    def time_stretch(audio: torch.Tensor, rate_range: Tuple[float, float] = (0.9, 1.1)) -> torch.Tensor:
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
    def time_stretch_resample(
        audio: torch.Tensor,
        sample_rate: int,
        rate_range: Tuple[float, float] = (0.8, 1.2)
    ) -> torch.Tensor:
        """
        Time stretching using resampling (better quality).
        Used in ACDNet training.
        """
        rate = random.uniform(*rate_range)
        if abs(rate - 1.0) < 0.01:
            return audio

        # Resample to stretched rate
        stretched_sr = int(sample_rate * rate)
        audio = torchaudio.functional.resample(audio, sample_rate, stretched_sr)
        # Resample back to original rate
        audio = torchaudio.functional.resample(audio, stretched_sr, sample_rate)
        return audio

    @staticmethod
    def pitch_shift(
        audio: torch.Tensor,
        sample_rate: int,
        semitones_range: Tuple[int, int] = (-2, 2)
    ) -> torch.Tensor:
        """
        Pitch shifting using resampling.
        Note: This is not true pitch shifting (which preserves tempo).
        """
        shift = random.randint(*semitones_range)
        if shift == 0:
            return audio

        # Convert semitones to rate
        rate = 2 ** (shift / 12)
        shifted_sr = int(sample_rate * rate)
        audio = torchaudio.functional.resample(audio, sample_rate, shifted_sr)
        audio = torchaudio.functional.resample(audio, shifted_sr, sample_rate)
        return audio


class AudioDataset(Dataset):
    """
    Unified audio dataset supporting both labeled and unlabeled data.

    Supports two modes:
    1. Directory mode: Scan directory for audio files (for codec training)
    2. Metadata mode: Load from provided file lists with labels (for classification)

    Args:
        audio_dir: Directory containing audio files
        sample_rate: Target sample rate (24000 for SoundStream, 20000 for ACDNet)
        audio_length_seconds: Length of audio chunks in seconds
        extensions: List of audio file extensions to include
        return_labels: If True, return (audio, label) tuples; if False, return audio only
        files: Optional list of file paths (for metadata mode)
        labels: Optional list of labels (for metadata mode)
        class_names: Optional list of class names
        split: 'train' or 'test' (affects cropping behavior)
        augment: Enable augmentation
        mixup: Enable mixup augmentation (requires return_labels=True)
        mixup_alpha: Alpha parameter for mixup
        normalization: 'peak' (max normalize), 'rms', or 'fixed' (divide by 32768)
    """

    def __init__(
        self,
        audio_dir: Union[str, Path],
        sample_rate: int = 24000,
        audio_length_seconds: float = 2.0,
        extensions: List[str] = ['.wav', '.mp3', '.flac', '.ogg'],
        return_labels: bool = False,
        files: Optional[List[Path]] = None,
        labels: Optional[List[int]] = None,
        class_names: Optional[List[str]] = None,
        split: str = 'train',
        augment: bool = False,
        mixup: bool = False,
        mixup_alpha: float = 0.5,
        normalization: str = 'peak'
    ):
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.audio_length = int(audio_length_seconds * sample_rate)
        self.return_labels = return_labels
        self.split = split
        self.augment = augment and (split == 'train')
        self.mixup = mixup and (split == 'train') and return_labels
        self.mixup_alpha = mixup_alpha
        self.normalization = normalization

        # Load files
        if files is not None:
            # Metadata mode: use provided files
            self.audio_files = files
            self.labels = labels if labels is not None else [0] * len(files)
            self.class_names = class_names if class_names is not None else []
        else:
            # Directory mode: scan for files
            self.audio_files = []
            for ext in extensions:
                self.audio_files.extend(list(self.audio_dir.rglob(f'*{ext}')))
            self.labels = [0] * len(self.audio_files)  # Dummy labels
            self.class_names = []

        if len(self.audio_files) == 0:
            raise ValueError(f"No audio files found in {audio_dir}")

        if return_labels and labels is None and files is not None:
            warnings.warn("return_labels=True but no labels provided")

        print(f"Loaded {len(self.audio_files)} audio files from {audio_dir}")

    def __len__(self) -> int:
        return len(self.audio_files)

    def _load_audio(self, path: Path) -> torch.Tensor:
        """Load and preprocess audio file"""
        try:
            # Load audio
            waveform, sr = torchaudio.load(str(path))

            # Resample if necessary
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Pad or crop to target length
            current_length = waveform.shape[1]

            if current_length < self.audio_length:
                # Pad with zeros
                pad_length = self.audio_length - current_length
                waveform = torch.nn.functional.pad(waveform, (0, pad_length))
            elif current_length > self.audio_length:
                # Crop: random for train, center for test
                if self.split == 'train':
                    start = random.randint(0, current_length - self.audio_length)
                else:
                    start = (current_length - self.audio_length) // 2
                waveform = waveform[:, start:start + self.audio_length]

            # Normalize
            if self.normalization == 'peak':
                waveform = AudioNormalizer.peak_normalize(waveform)
            elif self.normalization == 'rms':
                waveform = AudioNormalizer.rms_normalize(waveform)
            elif self.normalization == 'fixed':
                waveform = AudioNormalizer.fixed_normalize(waveform)
            else:
                # Fallback to peak normalization
                if waveform.abs().max() > 0:
                    waveform = waveform / waveform.abs().max()

            return waveform

        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return random noise as fallback
            return torch.randn(1, self.audio_length)

    def _apply_augmentation(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply data augmentation.

        Uses fast variants:
        - time_stretch: Fast linear interpolation (not resampling-based)
        - Gaussian noise: No resampling needed
        - Pitch shift: Disabled (too slow with resampling)

        This keeps augmentation fast while still providing variation.
        """
        # Time stretching (fast linear interpolation variant)
        if random.random() < 0.3:
            waveform = AudioAugmentation.time_stretch(
                waveform, rate_range=(0.9, 1.1)
            )

        # Pitch shifting - DISABLED for performance
        # The resampling-based pitch shift is too slow (4 resample ops per sample)
        # Can be re-enabled if needed: AudioAugmentation.pitch_shift()

        # Add Gaussian noise (fast, no resampling)
        if random.random() < 0.3:
            noise_factor = random.uniform(0.001, 0.01)
            waveform = AudioAugmentation.add_noise(waveform, noise_factor)

        # Ensure correct length after augmentation
        if waveform.shape[1] != self.audio_length:
            if waveform.shape[1] < self.audio_length:
                pad_length = self.audio_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_length))
            else:
                waveform = waveform[:, :self.audio_length]

        return waveform

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get a single sample.

        Returns:
            If return_labels=False: audio tensor of shape (1, target_length)
            If return_labels=True: (audio, label) where label is:
                - Tensor of shape (num_classes,) for mixup
                - Scalar tensor for regular training
        """
        # Load audio
        audio = self._load_audio(self.audio_files[idx])
        label = self.labels[idx]

        # Apply augmentation
        if self.augment:
            audio = self._apply_augmentation(audio)

        # Apply mixup (following EnvNet-v2 approach)
        if self.mixup:
            # Pick another random sample
            idx2 = random.randint(0, len(self) - 1)
            audio2 = self._load_audio(self.audio_files[idx2])
            label2 = self.labels[idx2]

            if self.augment:
                audio2 = self._apply_augmentation(audio2)

            # Pad both with T/2 zeros on each side
            T = self.audio_length
            audio = torch.nn.functional.pad(audio, (T // 2, T // 2))
            audio2 = torch.nn.functional.pad(audio2, (T // 2, T // 2))

            # Randomly crop T-length sections
            start1 = random.randint(0, audio.shape[1] - T)
            start2 = random.randint(0, audio2.shape[1] - T)
            audio = audio[:, start1:start1 + T]
            audio2 = audio2[:, start2:start2 + T]

            # Calculate mixing ratio according to EnvNet-v2 paper
            g1 = torch.max(torch.abs(audio))
            g2 = torch.max(torch.abs(audio2))
            # Avoid r=0 or r=1 to prevent division by zero or degenerate cases
            r = random.uniform(0.01, 0.99)

            # p = 1 / (1 + 10^((g1-g2)/20 * (1-r)/r))
            # No epsilon needed since r is bounded away from 0
            p = 1.0 / (1.0 + 10 ** ((g1 - g2) / 20 * (1 - r) / r))

            # Mix audio
            audio_mix = (p * audio + (1 - p) * audio2) / torch.sqrt(p**2 + (1 - p)**2)

            # Create soft labels
            num_classes = len(self.class_names)
            label_mix = torch.zeros(num_classes, dtype=torch.float32)

            # Handle edge case: both samples from same class
            if label == label2:
                label_mix[label] = 1.0  # p + (1-p) = 1.0
            else:
                label_mix[label] = p
                label_mix[label2] = 1 - p

            return audio_mix, label_mix

        # Return based on mode
        if self.return_labels:
            return audio, torch.tensor(label, dtype=torch.long)
        else:
            return audio


class ESC50Dataset(AudioDataset):
    """
    ESC-50 Dataset (and ESC-10 subset).

    2000 samples, 50 classes, 5-fold CV
    ESC-10 is first 10 classes of ESC-50

    Args:
        root: Path to ESC-50 directory containing 'audio' folder and 'meta/esc50.csv'
        split: 'train' or 'test'
        fold: Fold number for cross-validation (1-5)
        subset: Either 'esc50' or 'esc10'
        target_length: Target audio length in samples (default: 30225 ~1.51s @ 20kHz)
        target_sr: Target sample rate (default: 20000)
        augment: Enable augmentation
        mixup: Enable mixup
        mixup_alpha: Mixup alpha parameter
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: str = 'train',
        fold: int = 1,
        subset: str = 'esc50',
        target_length: int = 30225,
        target_sr: int = 20000,
        augment: bool = False,
        mixup: bool = False,
        mixup_alpha: float = 0.5
    ):
        import pandas as pd

        root = Path(root)

        # Load metadata
        meta_file = root / 'meta' / 'esc50.csv'
        if not meta_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_file}")

        df = pd.read_csv(meta_file)

        # Filter by subset (ESC-10 or ESC-50)
        if subset == 'esc10':
            df = df[df['esc10'] == True]

        # Split by fold
        if split == 'train':
            df = df[df['fold'] != fold]
        else:
            df = df[df['fold'] == fold]

        # Get files and labels
        audio_dir = root / 'audio'
        files = [audio_dir / filename for filename in df['filename']]

        # Get class names and create label mapping
        class_names = sorted(df['category'].unique())
        label_to_idx = {name: idx for idx, name in enumerate(class_names)}

        # Remap labels to continuous indices
        categories = df['category'].tolist()
        labels = [label_to_idx[cat] for cat in categories]

        # Initialize parent class
        super().__init__(
            audio_dir=root,
            sample_rate=target_sr,
            audio_length_seconds=target_length / target_sr,
            return_labels=True,
            files=files,
            labels=labels,
            class_names=class_names,
            split=split,
            augment=augment,
            mixup=mixup,
            mixup_alpha=mixup_alpha,
            normalization='peak'  # ESC-50/UrbanSound8K files are float32, use peak norm
        )


class UrbanSound8KDataset(AudioDataset):
    """
    UrbanSound8K Dataset.

    8732 samples, 10 classes, 10-fold CV

    Args:
        root: Path to UrbanSound8K directory containing 'audio' and 'metadata' folders
        split: 'train' or 'test'
        fold: Fold number for cross-validation (1-10)
        target_length: Target audio length in samples (default: 30225 ~1.51s @ 20kHz)
        target_sr: Target sample rate (default: 20000)
        augment: Enable augmentation
        mixup: Enable mixup
        mixup_alpha: Mixup alpha parameter
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: str = 'train',
        fold: int = 1,
        target_length: int = 30225,
        target_sr: int = 20000,
        augment: bool = False,
        mixup: bool = False,
        mixup_alpha: float = 0.5
    ):
        import pandas as pd

        root = Path(root)

        # Load metadata
        meta_file = root / 'metadata' / 'UrbanSound8K.csv'
        if not meta_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_file}")

        df = pd.read_csv(meta_file)

        # Split by fold
        if split == 'train':
            df = df[df['fold'] != fold]
        else:
            df = df[df['fold'] == fold]

        # Get files and labels
        files = []
        for _, row in df.iterrows():
            file_path = root / 'audio' / f"fold{row['fold']}" / row['slice_file_name']
            files.append(file_path)

        labels = df['classID'].tolist()

        # Get class names
        class_names = sorted(df['class'].unique())

        # Initialize parent class
        super().__init__(
            audio_dir=root,
            sample_rate=target_sr,
            audio_length_seconds=target_length / target_sr,
            return_labels=True,
            files=files,
            labels=labels,
            class_names=class_names,
            split=split,
            augment=augment,
            mixup=mixup,
            mixup_alpha=mixup_alpha,
            normalization='peak'  # ESC-50/UrbanSound8K files are float32, use peak norm
        )


def create_dataloader(
    audio_dir: str,
    batch_size: int,
    sample_rate: int = 24000,
    audio_length_seconds: float = 2.0,
    num_workers: int = 4,
    shuffle: bool = True
) -> DataLoader:
    """
    Create a data loader for unlabeled audio files (for codec training).

    Args:
        audio_dir: Directory containing audio files
        batch_size: Batch size
        sample_rate: Target sample rate
        audio_length_seconds: Audio length in seconds
        num_workers: Number of dataloader workers
        shuffle: Whether to shuffle

    Returns:
        DataLoader yielding audio tensors of shape (batch, 1, time)
    """
    dataset = AudioDataset(
        audio_dir=audio_dir,
        sample_rate=sample_rate,
        audio_length_seconds=audio_length_seconds,
        return_labels=False,
        normalization='peak'
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return dataloader


def create_dataloaders(
    dataset_name: str,
    root: Union[str, Path],
    fold: int = 1,
    batch_size: int = 64,
    num_workers: int = 4,
    target_length: int = 30225,
    target_sr: int = 20000,
    augment: bool = True,
    mixup: bool = True,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders for classification tasks.

    Args:
        dataset_name: One of 'esc50', 'esc10', 'urbansound8k'
        root: Root directory of the dataset
        fold: Fold number for cross-validation
        batch_size: Batch size
        num_workers: Number of dataloader workers
        target_length: Target audio length in samples
        target_sr: Target sample rate
        augment: Whether to apply augmentation
        mixup: Whether to apply mixup

    Returns:
        (train_loader, test_loader) tuple
        Each yields (audio, label) where:
            audio: (batch, 1, time)
            label: (batch,) for regular training or (batch, num_classes) for mixup
    """
    if dataset_name.lower() in ['esc50', 'esc10']:
        dataset_class = ESC50Dataset
        subset = 'esc10' if dataset_name.lower() == 'esc10' else 'esc50'
        extra_kwargs = {'subset': subset}
    elif dataset_name.lower() == 'urbansound8k':
        dataset_class = UrbanSound8KDataset
        extra_kwargs = {}
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Create datasets
    train_dataset = dataset_class(
        root=root,
        split='train',
        fold=fold,
        target_length=target_length,
        target_sr=target_sr,
        augment=augment,
        mixup=mixup,
        **extra_kwargs,
        **kwargs
    )

    test_dataset = dataset_class(
        root=root,
        split='test',
        fold=fold,
        target_length=target_length,
        target_sr=target_sr,
        augment=False,
        mixup=False,
        **extra_kwargs,
        **kwargs
    )

    # Create dataloaders with optimizations for GPU utilization
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,  # Keep workers alive between epochs
        prefetch_factor=2 if num_workers > 0 else None,  # Prefetch 2 batches per worker
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return train_loader, test_loader


if __name__ == "__main__":
    # Test the unified dataset
    import sys

    print("="*60)
    print("Testing Unified Dataset")
    print("="*60)

    # Test 1: Directory mode (codec training)
    if len(sys.argv) > 1:
        audio_dir = sys.argv[1]
        print(f"\nTest 1: Directory mode (codec training)")
        print(f"Audio dir: {audio_dir}")

        try:
            dataloader = create_dataloader(
                audio_dir=audio_dir,
                batch_size=4,
                sample_rate=24000,
                audio_length_seconds=2.0,
                num_workers=2
            )

            for batch in dataloader:
                print(f"Batch shape: {batch.shape}")
                print(f"Batch range: [{batch.min():.3f}, {batch.max():.3f}]")
                break

            print("✓ Directory mode test passed!")
        except Exception as e:
            print(f"✗ Directory mode test failed: {e}")

    # Test 2: ESC-50 mode (classification)
    esc50_root = Path("/path/to/ESC-50")
    if esc50_root.exists():
        print(f"\nTest 2: ESC-50 mode (classification)")
        try:
            train_loader, test_loader = create_dataloaders(
                dataset_name='esc50',
                root=esc50_root,
                fold=1,
                batch_size=4,
                num_workers=0
            )

            for audio, labels in train_loader:
                print(f"Audio batch shape: {audio.shape}")
                print(f"Labels batch shape: {labels.shape}")
                print(f"Audio range: [{audio.min():.3f}, {audio.max():.3f}]")
                break

            print("✓ ESC-50 mode test passed!")
        except Exception as e:
            print(f"✗ ESC-50 mode test failed: {e}")
    else:
        print(f"\nTest 2: Skipped (ESC-50 not found at {esc50_root})")

    print("\n" + "="*60)
    print("Dataset tests complete!")
    print("="*60)

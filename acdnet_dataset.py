"""
Dataset module for Environmental Sound Classification
Supports ESC-10, ESC-50, UrbanSound8K, and AudioEvent datasets
"""

import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional, List, Union
import random


class EnvironmentalSoundDataset(Dataset):
    """
    Dataset for Environmental Sound Classification.
    
    Supports:
    - ESC-10: 400 samples, 10 classes, 5-fold CV
    - ESC-50: 2000 samples, 50 classes, 5-fold CV
    - UrbanSound8K: 8732 samples, 10 classes, 10-fold CV
    
    Args:
        root: Root directory containing audio files
        split: Either 'train' or 'test'
        fold: Fold number for cross-validation
        target_length: Target length of audio in samples (default: 30225 ~1.51s @ 20kHz)
        target_sr: Target sample rate (default: 20000 Hz)
        augment: Whether to apply data augmentation (default: False)
        mixup: Whether to apply mixup augmentation (default: False)
        mixup_alpha: Alpha parameter for mixup (default: 0.5)
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
        super().__init__()
        
        self.root = Path(root)
        self.split = split
        self.fold = fold
        self.target_length = target_length
        self.target_sr = target_sr
        self.augment = augment and (split == 'train')
        self.mixup = mixup and (split == 'train')
        self.mixup_alpha = mixup_alpha
        
        # Load file list and labels
        self.files, self.labels, self.class_names = self._load_dataset()
        
        print(f"Loaded {len(self.files)} files for {split} (fold {fold})")
        print(f"Classes: {len(self.class_names)}")
        
    def _load_dataset(self) -> Tuple[List[Path], List[int], List[str]]:
        """Load dataset files and labels. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _load_dataset")
    
    def _load_audio(self, path: Path) -> torch.Tensor:
        """
        Load and preprocess audio file.
        
        Returns:
            Audio tensor of shape (1, target_length)
        """
        # Load audio
        waveform, sr = torchaudio.load(str(path))
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
        
        # Pad or crop to target length
        current_length = waveform.shape[1]
        
        if current_length < self.target_length:
            # Pad with zeros
            pad_length = self.target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        elif current_length > self.target_length:
            # Crop from center or random position during training
            if self.split == 'train':
                start = random.randint(0, current_length - self.target_length)
            else:
                start = (current_length - self.target_length) // 2
            waveform = waveform[:, start:start + self.target_length]
        
        # Normalize to [-1, 1]
        waveform = waveform / 32768.0  # Assuming 16-bit audio
        
        return waveform
    
    def _apply_augmentation(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation techniques."""
        # Time stretching
        if random.random() < 0.3:
            rate = random.uniform(0.8, 1.2)
            waveform = torchaudio.functional.resample(
                waveform, 
                self.target_sr, 
                int(self.target_sr * rate)
            )
            # Resample back to original rate
            waveform = torchaudio.functional.resample(
                waveform,
                int(self.target_sr * rate),
                self.target_sr
            )
        
        # Pitch shifting (simple version using resampling)
        if random.random() < 0.3:
            shift = random.randint(-2, 2)
            if shift != 0:
                rate = 2 ** (shift / 12)  # Semitone shift
                waveform = torchaudio.functional.resample(
                    waveform,
                    self.target_sr,
                    int(self.target_sr * rate)
                )
                waveform = torchaudio.functional.resample(
                    waveform,
                    int(self.target_sr * rate),
                    self.target_sr
                )
        
        # Add Gaussian noise
        if random.random() < 0.3:
            noise_factor = random.uniform(0.001, 0.01)
            noise = torch.randn_like(waveform) * noise_factor
            waveform = waveform + noise
        
        # Ensure still correct length
        if waveform.shape[1] != self.target_length:
            if waveform.shape[1] < self.target_length:
                pad_length = self.target_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_length))
            else:
                waveform = waveform[:, :self.target_length]
        
        return waveform
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            audio: Tensor of shape (1, target_length)
            label: Tensor of shape (num_classes,) for mixup or scalar for regular training
        """
        # Load audio
        audio = self._load_audio(self.files[idx])
        label = self.labels[idx]
        
        # Apply augmentation
        if self.augment:
            audio = self._apply_augmentation(audio)
        
        # Apply mixup during training (following EnvNet-v2 approach)
        if self.mixup:
            # Pick another random sample
            idx2 = random.randint(0, len(self) - 1)
            audio2 = self._load_audio(self.files[idx2])
            label2 = self.labels[idx2]
            
            if self.augment:
                audio2 = self._apply_augmentation(audio2)
            
            # Pad both with T/2 zeros on each side
            T = self.target_length
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
            r = random.random()
            
            # p = 1 / (1 + 10^((g1-g2)/20 * (1-r)/r))
            p = 1.0 / (1.0 + 10 ** ((g1 - g2) / 20 * (1 - r) / (r + 1e-8)))
            
            # Mix audio
            audio_mix = (p * audio + (1 - p) * audio2) / torch.sqrt(p**2 + (1 - p)**2)
            
            # Create soft labels
            num_classes = len(self.class_names)
            label_mix = torch.zeros(num_classes)
            label_mix[label] = p
            label_mix[label2] = 1 - p
            
            return audio_mix, label_mix
        else:
            return audio, torch.tensor(label, dtype=torch.long)


class ESC50Dataset(EnvironmentalSoundDataset):
    """
    ESC-50 Dataset (and ESC-10 subset).
    
    2000 samples, 50 classes, 5-fold CV
    ESC-10 is first 10 classes of ESC-50
    
    Args:
        root: Path to ESC-50 directory containing 'audio' folder and 'meta/esc50.csv'
        subset: Either 'esc50' or 'esc10'
    """
    
    def __init__(
        self,
        root: Union[str, Path],
        split: str = 'train',
        fold: int = 1,
        subset: str = 'esc50',
        **kwargs
    ):
        self.subset = subset
        super().__init__(root, split, fold, **kwargs)
    
    def _load_dataset(self) -> Tuple[List[Path], List[int], List[str]]:
        import pandas as pd
        
        # Load metadata
        meta_file = self.root / 'meta' / 'esc50.csv'
        if not meta_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_file}")
        
        df = pd.read_csv(meta_file)
        
        # Filter by subset (ESC-10 or ESC-50)
        if self.subset == 'esc10':
            df = df[df['esc10'] == True]
        
        # Split by fold
        if self.split == 'train':
            df = df[df['fold'] != self.fold]
        else:
            df = df[df['fold'] == self.fold]
        
        # Get files and labels
        audio_dir = self.root / 'audio'
        files = [audio_dir / filename for filename in df['filename']]
        labels = df['target'].tolist()
        
        # Get class names
        class_names = sorted(df['category'].unique())
        
        # Create label mapping
        label_to_idx = {name: idx for idx, name in enumerate(class_names)}
        
        # Remap labels to continuous indices
        categories = df['category'].tolist()
        labels = [label_to_idx[cat] for cat in categories]
        
        return files, labels, class_names


class UrbanSound8KDataset(EnvironmentalSoundDataset):
    """
    UrbanSound8K Dataset.
    
    8732 samples, 10 classes, 10-fold CV
    
    Args:
        root: Path to UrbanSound8K directory containing 'audio' and 'metadata' folders
    """
    
    def _load_dataset(self) -> Tuple[List[Path], List[int], List[str]]:
        import pandas as pd
        
        # Load metadata
        meta_file = self.root / 'metadata' / 'UrbanSound8K.csv'
        if not meta_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_file}")
        
        df = pd.read_csv(meta_file)
        
        # Split by fold
        if self.split == 'train':
            df = df[df['fold'] != self.fold]
        else:
            df = df[df['fold'] == self.fold]
        
        # Get files and labels
        files = []
        for _, row in df.iterrows():
            file_path = self.root / 'audio' / f"fold{row['fold']}" / row['slice_file_name']
            files.append(file_path)
        
        labels = df['classID'].tolist()
        
        # Get class names
        class_names = sorted(df['class'].unique())
        
        return files, labels, class_names


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
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and test dataloaders for a given dataset.
    
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
        train_loader, test_loader
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
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    root = Path("/path/to/ESC-50")
    
    if root.exists():
        train_loader, test_loader = create_dataloaders(
            dataset_name='esc50',
            root=root,
            fold=1,
            batch_size=4,
            num_workers=0
        )
        
        # Test loading a batch
        for audio, labels in train_loader:
            print(f"Audio batch shape: {audio.shape}")
            print(f"Labels batch shape: {labels.shape}")
            print(f"Audio range: [{audio.min():.3f}, {audio.max():.3f}]")
            break
    else:
        print(f"Dataset root not found: {root}")
        print("Please update the path to test the dataset.")

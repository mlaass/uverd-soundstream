"""
Main evaluation script for SoundStream and TinyStream models
"""

import sys
from pathlib import Path

# Add parent directory to path to import model modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torchaudio
import argparse
from tqdm import tqdm
import random
from typing import List, Tuple, Dict

from model import SoundStream
from model_tiny import TinyStream
from evaluation.metrics import compute_all_metrics
from evaluation.visualizations import create_all_visualizations
from evaluation.report_generator import generate_html_report


def load_model_from_checkpoint(checkpoint_path: str, device: str = 'cuda') -> Tuple[torch.nn.Module, Dict, Dict]:
    """
    Load model from checkpoint and auto-detect type

    Returns:
        model: Loaded model
        model_info: Dict with model metadata
        checkpoint_info: Dict with checkpoint metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    model_state = checkpoint['model_state_dict']

    # Detect model type by checking state_dict keys
    # TinyStream has depthwise separable convolutions
    state_keys = list(model_state.keys())
    is_tiny = any('depthwise' in key for key in state_keys)

    # Infer parameters from actual state_dict shapes
    if 'quantizer.quantizers.0.embeddings.weight' in model_state:
        codebook_size, D = model_state['quantizer.quantizers.0.embeddings.weight'].shape
    else:
        codebook_size = config.get('codebook_size', 1024)
        D = config.get('D', 512 if not is_tiny else 128)

    # Count number of quantizers
    num_quantizers = sum(1 for key in state_keys if 'quantizer.quantizers.' in key and '.embeddings.weight' in key)
    if num_quantizers == 0:
        num_quantizers = config.get('num_quantizers', 8)

    # Infer C from encoder initial conv
    if is_tiny and 'encoder.initial_conv.weight' in model_state:
        # TinyStream: (out_channels, in_channels, kernel)
        C_inferred = model_state['encoder.initial_conv.weight'].shape[0]
    elif 'encoder.initial_conv.conv.weight' in model_state:
        # SoundStream: has CausalConv1d wrapper
        C_inferred = model_state['encoder.initial_conv.conv.weight'].shape[0]
    else:
        C_inferred = config.get('C', 32)

    C = C_inferred

    # Sample rate from config or default
    sample_rate = config.get('sample_rate', 16000 if is_tiny else 24000)

    print(f"Detected model type: {'TinyStream' if is_tiny else 'SoundStream'}")
    print(f"Inferred from checkpoint: C={C}, D={D}, num_quantizers={num_quantizers}, codebook_size={codebook_size}")

    # Create appropriate model
    if is_tiny:
        model = TinyStream(
            C=C,
            D=D,
            strides=[4, 4, 4, 4],
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            sample_rate=sample_rate
        )
        model_name = "TinyStream"
    else:
        model = SoundStream(
            C=C,
            D=D,
            strides=[2, 4, 5, 8],
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            sample_rate=sample_rate
        )
        model_name = "SoundStream"

    # Load weights
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()

    # Model info
    model_info = {
        'model_name': model_name,
        'model_params': model.get_num_params(),
        'model_size_mb': model.get_num_params() * 4 / (1024 * 1024),  # FP32
        'config': {
            'C': C,
            'D': D,
            'num_quantizers': num_quantizers,
            'codebook_size': codebook_size,
            'sample_rate': sample_rate,
            'downsample_factor': model.downsample_factor
        }
    }

    # Checkpoint info
    checkpoint_info = {
        'checkpoint_path': checkpoint_path,
        'global_step': checkpoint.get('global_step', 0),
        'epoch': checkpoint.get('epoch', 0),
        'run_name': checkpoint.get('run_name', 'unknown')
    }

    print(f"Loaded {model_name} with {model_info['model_params']:,} parameters")
    print(f"Checkpoint: step {checkpoint_info['global_step']}, epoch {checkpoint_info['epoch']}")

    return model, model_info, checkpoint_info


def select_test_samples(audio_dir: str, num_samples: int, seed: int = 42) -> List[str]:
    """
    Select diverse test samples from audio directory

    Attempts to select samples from different categories if available
    (e.g., ESC-50 has categories encoded in filenames)
    """
    audio_dir = Path(audio_dir)
    audio_files = []

    # Collect all audio files
    for ext in ['.wav', '.mp3', '.flac', '.ogg']:
        audio_files.extend(list(audio_dir.rglob(f'*{ext}')))

    if len(audio_files) == 0:
        raise ValueError(f"No audio files found in {audio_dir}")

    print(f"Found {len(audio_files)} audio files")

    # Try to do smart sampling by detecting patterns
    # ESC-50 format: fold-clip-category.wav (e.g., 1-100032-A-0.wav)
    categories = {}
    for f in audio_files:
        # Try to extract category from filename
        parts = f.stem.split('-')
        if len(parts) >= 3:
            category = parts[2]  # Use third part as category
        else:
            category = 'unknown'

        if category not in categories:
            categories[category] = []
        categories[category].append(f)

    # If we have multiple categories, sample from each
    if len(categories) > 1 and 'unknown' not in categories:
        print(f"Detected {len(categories)} categories, sampling diversely")
        selected = []
        samples_per_category = max(1, num_samples // len(categories))

        random.seed(seed)
        for category, files in categories.items():
            selected.extend(random.sample(files, min(samples_per_category, len(files))))

        # Fill remaining slots if needed
        while len(selected) < num_samples and len(selected) < len(audio_files):
            remaining = [f for f in audio_files if f not in selected]
            selected.append(random.choice(remaining))

        selected = selected[:num_samples]
    else:
        # Random sampling
        print(f"Sampling {num_samples} files randomly")
        random.seed(seed)
        selected = random.sample(audio_files, min(num_samples, len(audio_files)))

    print(f"Selected {len(selected)} samples for evaluation")
    return [str(f) for f in selected]


def evaluate_sample(model: torch.nn.Module, audio_path: str, sample_rate: int,
                   downsample_factor: int, codebook_size: int, device: str) -> Dict:
    """
    Evaluate a single audio sample

    Returns dictionary with metrics and visualizations
    """
    # Load audio
    waveform, sr = torchaudio.load(audio_path)

    # Resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Add batch dimension and move to device
    original = waveform.unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        reconstructed, indices, _ = model(original)

    # Compute all metrics
    metrics = compute_all_metrics(
        original, reconstructed, indices,
        sample_rate, downsample_factor, codebook_size
    )

    # Create all visualizations
    visualizations = create_all_visualizations(
        original, reconstructed, indices, metrics, sample_rate
    )

    return {
        'filename': Path(audio_path).name,
        'metrics': metrics,
        'visualizations': visualizations
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate SoundStream/TinyStream model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--audio_dir', type=str, required=True,
                       help='Directory containing test audio files')
    parser.add_argument('--num_samples', type=int, default=8,
                       help='Number of samples to evaluate (default: 8)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output HTML report path (default: evaluation/outputs/report_<step>.html)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu, default: cuda)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sample selection (default: 42)')

    args = parser.parse_args()

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = 'cpu'

    print("="*60)
    print("SoundStream/TinyStream Model Evaluation")
    print("="*60)
    print()

    # Load model
    print("Loading model...")
    model, model_info, checkpoint_info = load_model_from_checkpoint(
        args.checkpoint, args.device
    )
    print()

    # Select test samples
    print("Selecting test samples...")
    test_files = select_test_samples(args.audio_dir, args.num_samples, args.seed)
    print()

    # Evaluate each sample
    print("Evaluating samples...")
    samples_data = []

    for audio_path in tqdm(test_files, desc="Evaluating"):
        try:
            sample_data = evaluate_sample(
                model, audio_path,
                model_info['config']['sample_rate'],
                model_info['config']['downsample_factor'],
                model_info['config']['codebook_size'],
                args.device
            )
            samples_data.append(sample_data)
        except Exception as e:
            print(f"Error evaluating {audio_path}: {e}")
            continue

    print(f"Successfully evaluated {len(samples_data)}/{len(test_files)} samples")
    print()

    if len(samples_data) == 0:
        print("Error: No samples were successfully evaluated!")
        print("Check the error messages above for details.")
        return

    # Generate output path if not specified
    if args.output is None:
        output_dir = Path(__file__).parent / 'outputs'
        output_dir.mkdir(exist_ok=True)

        # Create descriptive filename with model name and run name
        model_name_lower = model_info['model_name'].lower()
        run_name = checkpoint_info.get('run_name', 'unknown')
        step = checkpoint_info['global_step']

        # Format: <model>_<runname>_step_<step>.html
        # e.g., tinystream_20251111_162700_step_5000.html
        filename = f"{model_name_lower}_{run_name}_step_{step}.html"
        args.output = str(output_dir / filename)

    # Generate HTML report
    print("Generating HTML report...")
    generate_html_report(samples_data, model_info, checkpoint_info, args.output)
    print()

    print("="*60)
    print(f"✓ Evaluation complete!")
    print(f"  Report saved to: {args.output}")
    print(f"  Open in browser to view results")
    print("="*60)


if __name__ == "__main__":
    main()

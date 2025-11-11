"""
Inference script for SoundStream
"""

import torch
import torchaudio
import argparse
from pathlib import Path

from model import SoundStream


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Get model config from checkpoint if available
    # Otherwise use defaults
    model = SoundStream(
        C=config.get('C', 32),
        D=config.get('D', 512),
        strides=[2, 4, 5, 8],
        num_quantizers=config.get('num_quantizers', 8),
        codebook_size=config.get('codebook_size', 1024),
        sample_rate=config.get('sample_rate', 24000)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from step {checkpoint['global_step']}")
    return model, config['sample_rate']


def reconstruct_audio(
    model: SoundStream,
    audio_path: str,
    output_path: str,
    target_sr: int,
    num_quantizers: int = None,
    device: str = 'cuda'
):
    """Reconstruct audio through encode-decode"""
    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    print(f"Loaded audio: {waveform.shape}, sample rate: {sr}Hz")
    
    # Resample if needed
    if sr != target_sr:
        print(f"Resampling from {sr}Hz to {target_sr}Hz...")
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        print("Converting stereo to mono...")
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Add batch dimension
    waveform = waveform.unsqueeze(0).to(device)
    
    # Encode
    print("Encoding...")
    with torch.no_grad():
        indices = model.encode(waveform)
    
    print(f"Encoded to indices: {indices.shape}")
    print(f"Original bitrate: {model.num_quantizers} quantizers")
    
    # Optionally use fewer quantizers for lower bitrate
    if num_quantizers is not None and num_quantizers < indices.shape[1]:
        print(f"Using only {num_quantizers} quantizers (reduced bitrate)")
        indices = indices[:, :num_quantizers, :]
    
    # Calculate bitrate
    embedding_rate = target_sr / model.downsample_factor  # Hz
    bits_per_frame = indices.shape[1] * 10  # num_quantizers * log2(1024)
    bitrate_kbps = (bits_per_frame * embedding_rate) / 1000
    print(f"Bitrate: {bitrate_kbps:.1f} kbps")
    
    # Decode
    print("Decoding...")
    with torch.no_grad():
        reconstructed = model.decode(indices)
    
    # Save
    reconstructed = reconstructed.cpu()
    torchaudio.save(output_path, reconstructed[0], target_sr)
    print(f"Saved reconstruction to: {output_path}")
    
    # Compute metrics
    waveform_cpu = waveform.cpu()
    mse = torch.mean((reconstructed - waveform_cpu) ** 2).item()
    print(f"MSE: {mse:.6f}")


def main():
    parser = argparse.ArgumentParser(description='SoundStream Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Input audio file')
    parser.add_argument('--output', type=str, default='reconstructed.wav', help='Output audio file')
    parser.add_argument('--num_quantizers', type=int, default=None, help='Number of quantizers to use (for lower bitrate)')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, sample_rate = load_model(args.checkpoint, args.device)
    
    # Reconstruct audio
    reconstruct_audio(
        model=model,
        audio_path=args.input,
        output_path=args.output,
        target_sr=sample_rate,
        num_quantizers=args.num_quantizers,
        device=args.device
    )


if __name__ == "__main__":
    main()

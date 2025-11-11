"""
Metric computation functions for audio codec evaluation
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


def compute_mse(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Compute Mean Squared Error"""
    return F.mse_loss(original, reconstructed).item()


def compute_mae(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Compute Mean Absolute Error"""
    return F.l1_loss(original, reconstructed).item()


def compute_snr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """
    Compute Signal-to-Noise Ratio in dB

    SNR = 10 * log10(signal_power / noise_power)
    """
    noise = original - reconstructed
    signal_power = torch.mean(original ** 2)
    noise_power = torch.mean(noise ** 2)

    if noise_power < 1e-10:
        return float('inf')

    snr = 10 * torch.log10(signal_power / noise_power)
    return snr.item()


def compute_si_snr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """
    Compute Scale-Invariant Signal-to-Noise Ratio in dB

    More robust than SNR for audio signals with different scales
    """
    # Remove batch and channel dimensions for computation
    original = original.squeeze()
    reconstructed = reconstructed.squeeze()

    # Zero-mean signals
    original = original - torch.mean(original)
    reconstructed = reconstructed - torch.mean(reconstructed)

    # Compute projection
    alpha = torch.sum(original * reconstructed) / (torch.sum(original ** 2) + 1e-8)
    target = alpha * original
    noise = reconstructed - target

    # Compute SI-SNR
    si_snr = 10 * torch.log10(torch.sum(target ** 2) / (torch.sum(noise ** 2) + 1e-8))
    return si_snr.item()


def compute_spectral_convergence(original: torch.Tensor, reconstructed: torch.Tensor,
                                 n_fft: int = 1024) -> float:
    """
    Compute Spectral Convergence (lower is better)

    Measures the difference between magnitude spectrograms
    """
    # Compute STFTs
    orig_stft = torch.stft(
        original.squeeze(), n_fft=n_fft, hop_length=n_fft//4,
        window=torch.hann_window(n_fft).to(original.device),
        return_complex=True
    )
    recon_stft = torch.stft(
        reconstructed.squeeze(), n_fft=n_fft, hop_length=n_fft//4,
        window=torch.hann_window(n_fft).to(reconstructed.device),
        return_complex=True
    )

    # Get magnitudes
    orig_mag = torch.abs(orig_stft)
    recon_mag = torch.abs(recon_stft)

    # Compute spectral convergence
    numerator = torch.norm(orig_mag - recon_mag, p='fro')
    denominator = torch.norm(orig_mag, p='fro')

    sc = numerator / (denominator + 1e-8)
    return sc.item()


def compute_log_spectral_distance(original: torch.Tensor, reconstructed: torch.Tensor,
                                   n_fft: int = 1024) -> float:
    """
    Compute Log-Spectral Distance in dB

    Perceptually motivated frequency-domain metric
    """
    # Compute STFTs
    orig_stft = torch.stft(
        original.squeeze(), n_fft=n_fft, hop_length=n_fft//4,
        window=torch.hann_window(n_fft).to(original.device),
        return_complex=True
    )
    recon_stft = torch.stft(
        reconstructed.squeeze(), n_fft=n_fft, hop_length=n_fft//4,
        window=torch.hann_window(n_fft).to(reconstructed.device),
        return_complex=True
    )

    # Get power spectrograms
    orig_power = torch.abs(orig_stft) ** 2 + 1e-8
    recon_power = torch.abs(recon_stft) ** 2 + 1e-8

    # Compute log-spectral distance
    log_diff = torch.log10(orig_power) - torch.log10(recon_power)
    lsd = torch.sqrt(torch.mean(log_diff ** 2))

    return lsd.item()


def compute_pesq(original_np: np.ndarray, reconstructed_np: np.ndarray,
                 sample_rate: int) -> Optional[float]:
    """
    Compute PESQ (Perceptual Evaluation of Speech Quality)

    Returns score between -0.5 and 4.5 (higher is better)
    Requires 'pesq' library
    """
    try:
        from pesq import pesq

        # PESQ requires specific sample rates
        if sample_rate not in [8000, 16000]:
            # Resample if needed
            from scipy import signal
            if sample_rate > 16000:
                target_sr = 16000
            else:
                target_sr = 8000

            num_samples = int(len(original_np) * target_sr / sample_rate)
            original_resampled = signal.resample(original_np, num_samples)
            reconstructed_resampled = signal.resample(reconstructed_np, num_samples)
            sample_rate = target_sr
        else:
            original_resampled = original_np
            reconstructed_resampled = reconstructed_np

        # PESQ mode: 'wb' (wideband) for 16kHz, 'nb' (narrowband) for 8kHz
        mode = 'wb' if sample_rate == 16000 else 'nb'

        score = pesq(sample_rate, original_resampled, reconstructed_resampled, mode)
        return float(score)

    except ImportError:
        print("Warning: 'pesq' library not installed. Skipping PESQ metric.")
        return None
    except Exception as e:
        print(f"Warning: PESQ computation failed: {e}")
        return None


def compute_stoi(original_np: np.ndarray, reconstructed_np: np.ndarray,
                 sample_rate: int) -> Optional[float]:
    """
    Compute STOI (Short-Time Objective Intelligibility)

    Returns score between 0 and 1 (higher is better)
    Requires 'pystoi' library
    """
    try:
        from pystoi import stoi

        score = stoi(original_np, reconstructed_np, sample_rate, extended=False)
        return float(score)

    except ImportError:
        print("Warning: 'pystoi' library not installed. Skipping STOI metric.")
        return None
    except Exception as e:
        print(f"Warning: STOI computation failed: {e}")
        return None


def compute_compression_metrics(original_length: int, indices: torch.Tensor,
                                sample_rate: int, downsample_factor: int) -> Dict[str, float]:
    """
    Compute compression-related metrics

    Args:
        original_length: Number of samples in original audio
        indices: (batch, num_quantizers, time) - discrete codes
        sample_rate: Audio sample rate
        downsample_factor: Encoder downsampling factor

    Returns:
        Dictionary with bitrate, compression ratio, etc.
    """
    num_quantizers, num_frames = indices.shape[1], indices.shape[2]

    # Calculate bitrate (assuming 10 bits per quantizer = log2(1024))
    bits_per_code = 10  # log2(codebook_size=1024)
    embedding_rate = sample_rate / downsample_factor  # Hz
    bitrate_kbps = (num_quantizers * bits_per_code * embedding_rate) / 1000

    # Original size in bytes (assuming 16-bit PCM)
    original_bytes = original_length * 2  # 16-bit = 2 bytes per sample

    # Compressed size in bytes
    compressed_bytes = (num_quantizers * num_frames * bits_per_code) / 8

    # Compression ratio
    compression_ratio = original_bytes / compressed_bytes

    # Bytes saved
    bytes_saved = original_bytes - compressed_bytes
    percent_saved = (bytes_saved / original_bytes) * 100

    return {
        'bitrate_kbps': bitrate_kbps,
        'original_bytes': original_bytes,
        'compressed_bytes': compressed_bytes,
        'compression_ratio': compression_ratio,
        'bytes_saved': bytes_saved,
        'percent_saved': percent_saved,
        'num_frames': num_frames,
        'embedding_rate_hz': embedding_rate
    }


def compute_codebook_stats(indices: torch.Tensor, codebook_size: int) -> Dict[str, any]:
    """
    Analyze codebook utilization and entropy

    Args:
        indices: (batch, num_quantizers, time) - discrete codes
        codebook_size: Size of codebook

    Returns:
        Dictionary with utilization, perplexity, entropy per layer
    """
    batch_size, num_quantizers, num_frames = indices.shape

    stats = {
        'per_layer': [],
        'overall': {}
    }

    all_codes = []

    # Analyze each quantizer layer
    for q in range(num_quantizers):
        layer_indices = indices[:, q, :].flatten().cpu().numpy()
        all_codes.extend(layer_indices.tolist())

        # Count unique codes used
        unique_codes = np.unique(layer_indices)
        utilization = len(unique_codes) / codebook_size

        # Compute probability distribution
        hist, _ = np.histogram(layer_indices, bins=np.arange(codebook_size + 1))
        probs = hist / (hist.sum() + 1e-10)
        probs = probs[probs > 0]  # Remove zero probabilities

        # Compute entropy
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        # Compute perplexity (2^entropy)
        perplexity = 2 ** entropy

        stats['per_layer'].append({
            'layer': q,
            'unique_codes': len(unique_codes),
            'utilization': utilization,
            'entropy': entropy,
            'perplexity': perplexity,
            'most_common': int(np.argmax(hist)),
            'most_common_count': int(np.max(hist))
        })

    # Overall statistics
    all_codes = np.array(all_codes)
    unique_overall = len(np.unique(all_codes))
    hist_overall, _ = np.histogram(all_codes, bins=np.arange(codebook_size + 1))
    probs_overall = hist_overall / (hist_overall.sum() + 1e-10)
    probs_overall = probs_overall[probs_overall > 0]
    entropy_overall = -np.sum(probs_overall * np.log2(probs_overall + 1e-10))

    stats['overall'] = {
        'unique_codes': unique_overall,
        'utilization': unique_overall / codebook_size,
        'entropy': entropy_overall,
        'perplexity': 2 ** entropy_overall,
        'total_codes': len(all_codes)
    }

    return stats


def compute_all_metrics(original: torch.Tensor, reconstructed: torch.Tensor,
                       indices: torch.Tensor, sample_rate: int,
                       downsample_factor: int, codebook_size: int) -> Dict[str, any]:
    """
    Compute all metrics for a single audio sample

    Args:
        original: Original audio tensor (batch, channels, time)
        reconstructed: Reconstructed audio tensor (batch, channels, time)
        indices: Quantized indices (batch, num_quantizers, time)
        sample_rate: Audio sample rate
        downsample_factor: Encoder downsampling factor
        codebook_size: Codebook size

    Returns:
        Dictionary with all computed metrics
    """
    # Audio quality metrics
    metrics = {
        'mse': compute_mse(original, reconstructed),
        'mae': compute_mae(original, reconstructed),
        'snr_db': compute_snr(original, reconstructed),
        'si_snr_db': compute_si_snr(original, reconstructed),
        'spectral_convergence': compute_spectral_convergence(original, reconstructed),
        'log_spectral_distance_db': compute_log_spectral_distance(original, reconstructed),
    }

    # Perceptual metrics (if libraries available)
    original_np = original.squeeze().cpu().numpy()
    reconstructed_np = reconstructed.squeeze().cpu().numpy()

    pesq_score = compute_pesq(original_np, reconstructed_np, sample_rate)
    if pesq_score is not None:
        metrics['pesq'] = pesq_score

    stoi_score = compute_stoi(original_np, reconstructed_np, sample_rate)
    if stoi_score is not None:
        metrics['stoi'] = stoi_score

    # Compression metrics
    compression = compute_compression_metrics(
        original.shape[-1], indices, sample_rate, downsample_factor
    )
    metrics['compression'] = compression

    # Codebook statistics
    codebook = compute_codebook_stats(indices, codebook_size)
    metrics['codebook'] = codebook

    return metrics

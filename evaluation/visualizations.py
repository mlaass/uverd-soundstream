"""
Visualization functions for audio codec evaluation
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
from typing import Tuple, Dict
import plotly.graph_objects as go
import plotly.express as px


def fig_to_base64(fig: plt.Figure) -> str:
    """Convert matplotlib figure to base64 string for HTML embedding"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_str


def plot_waveform_comparison(original: torch.Tensor, reconstructed: torch.Tensor,
                             sample_rate: int) -> str:
    """
    Create waveform comparison plot

    Returns base64 encoded PNG
    """
    original_np = original.squeeze().cpu().numpy()
    reconstructed_np = reconstructed.squeeze().cpu().numpy()

    # Time axis
    time = np.arange(len(original_np)) / sample_rate

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    # Original waveform
    axes[0].plot(time, original_np, color='#2E86AB', linewidth=0.5)
    axes[0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, time[-1])

    # Reconstructed waveform
    axes[1].plot(time, reconstructed_np, color='#A23B72', linewidth=0.5)
    axes[1].set_title('Reconstructed', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, time[-1])

    # Difference (error)
    difference = original_np - reconstructed_np
    axes[2].plot(time, difference, color='#F18F01', linewidth=0.5)
    axes[2].set_title('Difference (Error)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Amplitude')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(0, time[-1])

    plt.tight_layout()
    return fig_to_base64(fig)


def plot_spectrogram_comparison(original: torch.Tensor, reconstructed: torch.Tensor,
                                sample_rate: int, n_fft: int = 1024,
                                hop_length: int = 256, spec_type: str = 'mel') -> str:
    """
    Create spectrogram comparison plot

    Args:
        spec_type: 'mel' or 'linear'

    Returns base64 encoded PNG
    """
    import librosa
    import librosa.display

    original_np = original.squeeze().cpu().numpy()
    reconstructed_np = reconstructed.squeeze().cpu().numpy()

    # Compute spectrograms
    if spec_type == 'mel':
        orig_spec = librosa.feature.melspectrogram(
            y=original_np, sr=sample_rate, n_fft=n_fft,
            hop_length=hop_length, n_mels=128
        )
        recon_spec = librosa.feature.melspectrogram(
            y=reconstructed_np, sr=sample_rate, n_fft=n_fft,
            hop_length=hop_length, n_mels=128
        )
        title_prefix = 'Mel'
    else:  # linear
        orig_stft = librosa.stft(original_np, n_fft=n_fft, hop_length=hop_length)
        recon_stft = librosa.stft(reconstructed_np, n_fft=n_fft, hop_length=hop_length)
        orig_spec = np.abs(orig_stft) ** 2
        recon_spec = np.abs(recon_stft) ** 2
        title_prefix = 'Linear'

    # Convert to dB
    orig_spec_db = librosa.power_to_db(orig_spec, ref=np.max)
    recon_spec_db = librosa.power_to_db(recon_spec, ref=np.max)

    # Compute difference
    diff_spec_db = orig_spec_db - recon_spec_db

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Original spectrogram
    img1 = librosa.display.specshow(
        orig_spec_db, sr=sample_rate, hop_length=hop_length,
        x_axis='time', y_axis='mel' if spec_type == 'mel' else 'hz',
        ax=axes[0], cmap='viridis'
    )
    axes[0].set_title(f'{title_prefix} Spectrogram - Original', fontsize=12, fontweight='bold')
    fig.colorbar(img1, ax=axes[0], format='%+2.0f dB')

    # Reconstructed spectrogram
    img2 = librosa.display.specshow(
        recon_spec_db, sr=sample_rate, hop_length=hop_length,
        x_axis='time', y_axis='mel' if spec_type == 'mel' else 'hz',
        ax=axes[1], cmap='viridis'
    )
    axes[1].set_title(f'{title_prefix} Spectrogram - Reconstructed', fontsize=12, fontweight='bold')
    fig.colorbar(img2, ax=axes[1], format='%+2.0f dB')

    # Difference spectrogram
    img3 = librosa.display.specshow(
        diff_spec_db, sr=sample_rate, hop_length=hop_length,
        x_axis='time', y_axis='mel' if spec_type == 'mel' else 'hz',
        ax=axes[2], cmap='coolwarm'
    )
    axes[2].set_title(f'{title_prefix} Spectrogram - Difference', fontsize=12, fontweight='bold')
    fig.colorbar(img3, ax=axes[2], format='%+2.0f dB')

    plt.tight_layout()
    return fig_to_base64(fig)


def create_codebook_heatmap(codebook_stats: Dict) -> str:
    """
    Create interactive Plotly heatmap of codebook utilization

    Returns HTML string with embedded Plotly chart
    """
    per_layer = codebook_stats['per_layer']

    # Extract data
    layers = [f"Layer {item['layer']}" for item in per_layer]
    utilizations = [item['utilization'] * 100 for item in per_layer]
    entropies = [item['entropy'] for item in per_layer]
    perplexities = [item['perplexity'] for item in per_layer]

    # Create heatmap data
    data = np.array([utilizations, entropies, perplexities])
    row_labels = ['Utilization (%)', 'Entropy (bits)', 'Perplexity']

    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=layers,
        y=row_labels,
        colorscale='Viridis',
        text=np.round(data, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))

    fig.update_layout(
        title='Codebook Statistics per Quantizer Layer',
        xaxis_title='Quantizer Layer',
        height=300,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn')


def create_metrics_bar_chart(metrics: Dict) -> str:
    """
    Create interactive Plotly bar chart of key metrics

    Returns HTML string with embedded Plotly chart
    """
    # Extract relevant metrics for display
    metric_names = []
    metric_values = []

    if 'snr_db' in metrics:
        metric_names.append('SNR (dB)')
        metric_values.append(metrics['snr_db'])

    if 'si_snr_db' in metrics:
        metric_names.append('SI-SNR (dB)')
        metric_values.append(metrics['si_snr_db'])

    if 'pesq' in metrics:
        metric_names.append('PESQ (0-4.5)')
        metric_values.append(metrics['pesq'])

    if 'stoi' in metrics:
        metric_names.append('STOI (0-1)')
        metric_values.append(metrics['stoi'])

    if 'spectral_convergence' in metrics:
        metric_names.append('Spectral Conv.')
        metric_values.append(metrics['spectral_convergence'])

    fig = go.Figure(data=[
        go.Bar(
            x=metric_names,
            y=metric_values,
            marker_color='#2E86AB',
            text=[f'{v:.3f}' for v in metric_values],
            textposition='outside'
        )
    ])

    fig.update_layout(
        title='Audio Quality Metrics',
        yaxis_title='Score',
        height=400,
        margin=dict(l=50, r=50, t=50, b=50),
        showlegend=False
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn')


def create_compression_pie_chart(compression_metrics: Dict) -> str:
    """
    Create Plotly pie chart showing compression breakdown

    Returns HTML string with embedded Plotly chart
    """
    original_bytes = compression_metrics['original_bytes']
    compressed_bytes = compression_metrics['compressed_bytes']
    bytes_saved = compression_metrics['bytes_saved']

    fig = go.Figure(data=[go.Pie(
        labels=['Compressed Data', 'Bytes Saved'],
        values=[compressed_bytes, bytes_saved],
        hole=0.3,
        marker_colors=['#A23B72', '#F18F01']
    )])

    fig.update_layout(
        title=f"Compression: {compression_metrics['compression_ratio']:.1f}x ratio "
              f"({compression_metrics['percent_saved']:.1f}% saved)",
        height=350,
        margin=dict(l=50, r=50, t=80, b=50),
        annotations=[dict(text=f"{compression_metrics['bitrate_kbps']:.1f} kbps",
                         x=0.5, y=0.5, font_size=16, showarrow=False)]
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn')


def audio_to_base64(audio: torch.Tensor, sample_rate: int) -> str:
    """
    Convert audio tensor to base64 encoded WAV for HTML embedding

    Args:
        audio: Audio tensor (batch, channels, time)
        sample_rate: Sample rate

    Returns:
        Base64 encoded WAV file
    """
    import scipy.io.wavfile as wavfile

    audio_np = audio.squeeze().cpu().numpy()

    # Normalize to int16 range
    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_np * 32767).astype(np.int16)

    # Write to buffer
    buf = io.BytesIO()
    wavfile.write(buf, sample_rate, audio_int16)
    buf.seek(0)

    # Encode to base64
    audio_b64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return audio_b64


def create_all_visualizations(original: torch.Tensor, reconstructed: torch.Tensor,
                              indices: torch.Tensor, metrics: Dict,
                              sample_rate: int) -> Dict[str, str]:
    """
    Generate all visualizations for a single audio sample

    Returns dictionary of base64/HTML encoded visualizations
    """
    visualizations = {}

    # Audio players (base64 WAV)
    visualizations['original_audio'] = audio_to_base64(original, sample_rate)
    visualizations['reconstructed_audio'] = audio_to_base64(reconstructed, sample_rate)

    # Waveform comparison
    visualizations['waveform'] = plot_waveform_comparison(original, reconstructed, sample_rate)

    # Mel spectrogram comparison
    visualizations['mel_spectrogram'] = plot_spectrogram_comparison(
        original, reconstructed, sample_rate, spec_type='mel'
    )

    # Linear spectrogram comparison
    visualizations['linear_spectrogram'] = plot_spectrogram_comparison(
        original, reconstructed, sample_rate, spec_type='linear'
    )

    # Codebook heatmap (Plotly)
    visualizations['codebook_heatmap'] = create_codebook_heatmap(metrics['codebook'])

    # Metrics bar chart (Plotly)
    visualizations['metrics_chart'] = create_metrics_bar_chart(metrics)

    # Compression pie chart (Plotly)
    visualizations['compression_chart'] = create_compression_pie_chart(metrics['compression'])

    return visualizations

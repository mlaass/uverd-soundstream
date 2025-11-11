"""
HTML report generation using Jinja2 templates
"""

from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import numpy as np


def compute_summary_statistics(samples_data: List[Dict]) -> Dict:
    """
    Compute aggregate statistics across all samples

    Args:
        samples_data: List of dictionaries with metrics for each sample

    Returns:
        Dictionary with average metrics
    """
    if not samples_data:
        return {}

    # Collect all metrics
    snr_values = []
    si_snr_values = []
    pesq_values = []
    stoi_values = []
    compression_ratios = []
    bitrates = []
    codebook_utilizations = []
    perplexities = []

    for sample in samples_data:
        metrics = sample['metrics']

        snr_values.append(metrics['snr_db'])
        si_snr_values.append(metrics['si_snr_db'])

        if 'pesq' in metrics:
            pesq_values.append(metrics['pesq'])

        if 'stoi' in metrics:
            stoi_values.append(metrics['stoi'])

        compression_ratios.append(metrics['compression']['compression_ratio'])
        bitrates.append(metrics['compression']['bitrate_kbps'])
        codebook_utilizations.append(metrics['codebook']['overall']['utilization'])
        perplexities.append(metrics['codebook']['overall']['perplexity'])

    summary = {
        'avg_snr_db': np.mean(snr_values),
        'std_snr_db': np.std(snr_values),
        'avg_si_snr_db': np.mean(si_snr_values),
        'std_si_snr_db': np.std(si_snr_values),
        'avg_compression_ratio': np.mean(compression_ratios),
        'avg_bitrate_kbps': np.mean(bitrates),
        'avg_codebook_utilization': np.mean(codebook_utilizations),
        'avg_perplexity': np.mean(perplexities),
    }

    if pesq_values:
        summary['avg_pesq'] = np.mean(pesq_values)
        summary['std_pesq'] = np.std(pesq_values)

    if stoi_values:
        summary['avg_stoi'] = np.mean(stoi_values)
        summary['std_stoi'] = np.std(stoi_values)

    return summary


def generate_html_report(
    samples_data: List[Dict],
    model_info: Dict,
    checkpoint_info: Dict,
    output_path: str
):
    """
    Generate complete HTML report

    Args:
        samples_data: List of dictionaries containing:
            - filename: str
            - metrics: Dict (from metrics.py)
            - visualizations: Dict (from visualizations.py)
        model_info: Dictionary with model information:
            - model_name: str (e.g., "SoundStream" or "TinyStream")
            - model_params: int
            - model_size_mb: float
            - config: Dict with C, D, num_quantizers, codebook_size, sample_rate
        checkpoint_info: Dictionary with checkpoint information:
            - checkpoint_path: str
            - global_step: int
            - epoch: int
        output_path: Path to save HTML report
    """
    # Set up Jinja2 environment
    template_dir = Path(__file__).parent / 'templates'
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template('report.html')

    # Compute summary statistics
    summary = compute_summary_statistics(samples_data)

    # Prepare template variables
    template_vars = {
        'model_name': model_info['model_name'],
        'model_params': f"{model_info['model_params']:,}",
        'model_size_mb': f"{model_info['model_size_mb']:.2f}",
        'config': model_info['config'],
        'checkpoint_path': checkpoint_info['checkpoint_path'],
        'global_step': checkpoint_info['global_step'],
        'epoch': checkpoint_info['epoch'],
        'num_samples': len(samples_data),
        'samples': samples_data,
        'summary': summary,
        'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Render template
    html_content = template.render(**template_vars)

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✓ HTML report generated: {output_path}")
    print(f"  - {len(samples_data)} samples evaluated")
    print(f"  - Average SNR: {summary['avg_snr_db']:.2f} dB")
    if 'avg_pesq' in summary:
        print(f"  - Average PESQ: {summary['avg_pesq']:.2f}")
    print(f"  - Average compression: {summary['avg_compression_ratio']:.1f}x "
          f"({summary['avg_bitrate_kbps']:.1f} kbps)")

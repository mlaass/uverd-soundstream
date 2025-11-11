# SoundStream/TinyStream Evaluation Tool

Interactive HTML evaluation reports for audio codec models with comprehensive metrics, visualizations, and audio playback.

## Features

- **Audio Quality Metrics**: MSE, MAE, SNR, SI-SNR, spectral convergence, log-spectral distance
- **Perceptual Metrics**: PESQ, STOI (optional)
- **Compression Analysis**: Bitrate, compression ratio, bytes saved
- **Codebook Statistics**: Utilization, perplexity, entropy per layer
- **Interactive Visualizations**:
  - Waveform comparisons
  - Mel spectrograms
  - Linear spectrograms
  - Error/difference plots
  - Plotly interactive charts
- **Audio Playback**: Embedded audio players for original vs reconstructed comparison
- **Smart Sampling**: Automatic diverse sample selection from datasets

## Installation

The evaluation tool requires additional dependencies:

```bash
# Install all required packages
uv add pesq pystoi plotly jinja2 scipy librosa

# Or install individually
uv add pesq        # Perceptual evaluation of speech quality
uv add pystoi      # Short-time objective intelligibility
uv add plotly      # Interactive plots
uv add jinja2      # HTML templating
uv add scipy       # Signal processing
uv add librosa     # Audio analysis
```

## Usage

### Basic Evaluation

```bash
# From project root
uv run python evaluation/evaluate.py \
    --checkpoint checkpoints/soundstream_step_10000.pt \
    --audio_dir datasets/ESC-50-master/audio \
    --num_samples 8 \
    --output evaluation/outputs/report.html
```

### Options

```
--checkpoint PATH     Path to model checkpoint (required)
--audio_dir PATH      Directory with test audio files (required)
--num_samples N       Number of samples to evaluate (default: 8)
--output PATH         Output HTML file path (default: auto-generated)
--device cuda/cpu     Device to use (default: cuda)
--seed INT           Random seed for sample selection (default: 42)
```

### Examples

**Evaluate TinyStream model:**
```bash
uv run python evaluation/evaluate.py \
    --checkpoint checkpoints_tiny/soundstream_20251111_162338_step_50000.pt \
    --audio_dir datasets/ESC-50-master/audio \
    --num_samples 10
```

**Evaluate with specific output path:**
```bash
uv run python evaluation/evaluate.py \
    --checkpoint checkpoints/soundstream_step_100000.pt \
    --audio_dir datasets/ESC-50-master/audio \
    --output reports/step100k_evaluation.html
```

**Evaluate on custom audio:**
```bash
uv run python evaluation/evaluate.py \
    --checkpoint checkpoints/soundstream_step_50000.pt \
    --audio_dir /path/to/test/audio \
    --num_samples 5
```

## Output

The tool generates a self-contained HTML report with:

### Summary Section
- Average metrics across all samples
- Aggregate statistics (SNR, PESQ, STOI, compression ratio)
- Model configuration and checkpoint info

### Per-Sample Sections
Each evaluated sample includes:
1. **Audio Players**: Original and reconstructed audio with playback controls
2. **Metrics Table**: All computed quality and compression metrics
3. **Waveform Comparison**: Time-domain visualization of original, reconstructed, and error
4. **Mel Spectrogram**: Frequency-domain comparison using mel scale
5. **Linear Spectrogram**: Full frequency range comparison
6. **Codebook Analysis**: Heatmap showing utilization, entropy, perplexity per layer
7. **Quality Metrics Chart**: Interactive bar chart of key metrics
8. **Compression Chart**: Pie chart showing compression breakdown

## Metrics Explained

### Audio Quality

- **MSE/MAE**: Mean squared/absolute error (lower is better)
- **SNR**: Signal-to-noise ratio in dB (higher is better)
- **SI-SNR**: Scale-invariant SNR, more robust (higher is better)
- **Spectral Convergence**: Spectrogram difference (lower is better)
- **Log-Spectral Distance**: Perceptual frequency metric in dB (lower is better)

### Perceptual Quality

- **PESQ**: Perceptual Evaluation of Speech Quality (0-4.5, higher is better)
  - Primarily for speech, may not be meaningful for music/environmental sounds
- **STOI**: Short-Time Objective Intelligibility (0-1, higher is better)
  - Measures speech intelligibility

### Compression

- **Bitrate**: Actual bitrate in kbps based on quantizers used
- **Compression Ratio**: Original size / compressed size (higher is better)
- **Bytes Saved**: Absolute savings vs uncompressed audio

### Codebook

- **Utilization**: Percentage of codebook entries actually used
- **Perplexity**: Effective number of codes used (2^entropy)
- **Entropy**: Measure of code distribution diversity in bits

## Smart Sampling

The evaluation tool automatically attempts to select diverse samples:

- **ESC-50**: Selects samples from different sound categories
- **Generic**: Random sampling with seed for reproducibility
- **Fallback**: Ensures coverage across available categories

## Interpreting Results

### Good Quality Indicators
- SNR > 20 dB
- SI-SNR > 15 dB
- PESQ > 3.0 (for speech)
- STOI > 0.8 (for speech)
- Spectral convergence < 0.1
- High codebook utilization (> 50%)

### Common Issues
- **Low SNR + high codebook utilization**: Model learning, needs more training
- **Low codebook utilization**: Codebook too large or model not exploring
- **High spectral convergence**: Poor frequency reconstruction
- **Low perplexity**: Model using only a few codes (possible mode collapse)

## File Structure

```
evaluation/
├── evaluate.py              # Main CLI script
├── metrics.py              # Metric computation functions
├── visualizations.py       # Plotting and visualization
├── report_generator.py     # HTML report generation
├── templates/
│   └── report.html         # Jinja2 HTML template
├── outputs/                # Generated reports (gitignored)
│   └── .gitkeep
└── README.md              # This file
```

## Customization

### Add New Metrics

Edit `evaluation/metrics.py`:

```python
def compute_my_metric(original, reconstructed):
    # Your metric computation
    return score

# Add to compute_all_metrics()
metrics['my_metric'] = compute_my_metric(original, reconstructed)
```

### Add New Visualizations

Edit `evaluation/visualizations.py`:

```python
def create_my_visualization(data):
    # Create matplotlib figure or plotly chart
    return visualization

# Add to create_all_visualizations()
visualizations['my_viz'] = create_my_visualization(data)
```

### Customize HTML Template

Edit `evaluation/templates/report.html` to modify:
- Layout and styling
- Sections to display
- Chart types and formatting

## Troubleshooting

### Missing Dependencies

```bash
# If PESQ fails
uv add pesq

# If STOI fails
uv add pystoi

# If librosa fails
uv add librosa
```

### PESQ/STOI Not Computing

- PESQ requires 8kHz or 16kHz audio (auto-resampled)
- STOI requires minimum audio length
- Both are optional - evaluation continues without them

### Out of Memory

- Reduce `--num_samples`
- Use `--device cpu` for CPU evaluation
- Evaluate shorter audio files

### Audio Loading Errors

- Check audio file formats (.wav, .mp3, .flac, .ogg)
- Verify file permissions
- Check for corrupted files in dataset

## Performance

Typical evaluation times (NVIDIA RTX 3090):
- **SoundStream**: ~5-10 seconds per sample
- **TinyStream**: ~3-5 seconds per sample
- **Report generation**: ~5-15 seconds

For 8 samples: **~1-2 minutes total**

## Examples

See `evaluation/outputs/` for example reports (generated after running evaluation).

To view a report:
```bash
# Open in browser
firefox evaluation/outputs/report_step_10000.html

# Or use Python's built-in server
cd evaluation/outputs
python -m http.server 8000
# Then open http://localhost:8000/report_step_10000.html
```

## Contributing

To add new features:
1. Add metric functions to `metrics.py`
2. Add visualization functions to `visualizations.py`
3. Update `report.html` template to display new content
4. Update `evaluate.py` if needed for CLI options
5. Document in this README

## License

Same as parent project (MIT License)

# Audio Datasets for SoundStream Training

This directory contains audio datasets for training and evaluating the SoundStream neural audio codec.

## Quick Start

Download datasets using the provided script:

```bash
# Download all free datasets (ESC-50, LibriSpeech)
../download_datasets.sh --all

# Download specific datasets
../download_datasets.sh --esc50
../download_datasets.sh --librispeech

# List available datasets
../download_datasets.sh --list
```

## Available Datasets

### ESC-50 (Environmental Sound Classification)
- **Size**: ~600MB
- **Samples**: 2,000 audio files (5 seconds each)
- **Classes**: 50 environmental sound classes
- **Format**: WAV, 44.1kHz
- **License**: Creative Commons
- **Download**: Automatic via script
- **Use case**: General environmental sounds, good for testing codec on diverse audio

**Download:**
```bash
../download_datasets.sh --esc50
```

**Train:**
```bash
python ../train.py --audio_dir datasets/ESC-50-master/audio --batch_size 8
```

---

### FSD50K (Freesound Dataset 50K)
- **Size**: ~80GB total (30GB dev + 50GB eval)
- **Samples**: 51,197 audio clips
- **Classes**: 200 sound event classes
- **Format**: WAV, variable sample rates
- **License**: Creative Commons (various)
- **Download**: Manual from [Zenodo](https://zenodo.org/record/4060432)
- **Use case**: Large-scale diverse audio, excellent for robust training

**Download:**
1. Visit https://zenodo.org/record/4060432
2. Download `FSD50K.dev_audio.zip` and `FSD50K.eval_audio.zip`
3. Place in this directory
4. Run: `../download_datasets.sh --fsd50k` to extract

**Train:**
```bash
python ../train.py --audio_dir datasets/FSD50K/dev_audio --batch_size 8
```

---

### UrbanSound8K
- **Size**: ~6GB
- **Samples**: 8,732 audio files (≤4 seconds each)
- **Classes**: 10 urban sound classes
- **Format**: WAV, variable sample rates
- **License**: Creative Commons
- **Download**: Manual from [website](https://urbansounddataset.weebly.com/urbansound8k.html)
- **Use case**: Urban environmental sounds

**Download:**
1. Visit https://urbansounddataset.weebly.com/urbansound8k.html
2. Fill out download form
3. Download `UrbanSound8K.tar.gz`
4. Place in this directory
5. Run: `../download_datasets.sh --urbansound8k` to extract

**Train:**
```bash
python ../train.py --audio_dir datasets/UrbanSound8K/audio --batch_size 8
```

---

### LibriSpeech (test-clean)
- **Size**: ~350MB
- **Samples**: ~2,600 audio files
- **Format**: FLAC, 16kHz
- **License**: CC BY 4.0
- **Download**: Automatic via script
- **Use case**: Testing codec on speech (clean recordings)

**Download:**
```bash
../download_datasets.sh --librispeech
```

**Train:**
```bash
python ../train.py --audio_dir datasets/LibriSpeech/test-clean --batch_size 8
```

---

### Bird Audio Detection Challenge (DCASE 2018)
- **Size**: ~20GB
- **Samples**: ~15,000 audio files (10 seconds each)
- **Format**: WAV, 44.1kHz
- **License**: Creative Commons
- **Download**: Manual from [DCASE](https://dcase.community/challenge2018/task-bird-audio-detection)
- **Use case**: Real forest monitoring, bird detection in natural environments

**Download:**
1. Visit https://dcase.community/challenge2018/task-bird-audio-detection
2. Download training and validation sets
3. Place archives in this directory
4. Run: `../download_datasets.sh --birdaudio` to extract

**Train:**
```bash
python ../train.py --audio_dir datasets/BirdVox-DCASE-20k --batch_size 8
```

---

### FSC22 (Forest Sound Classification)
- **Size**: ~5GB
- **Samples**: ~7,000 audio clips
- **Classes**: Forest-specific sounds (birds, insects, wind, rain, etc.)
- **Format**: WAV, variable sample rates
- **License**: Creative Commons
- **Download**: Manual from [Zenodo](https://zenodo.org/record/6467836)
- **Use case**: Purpose-built for forest environment classification

**Download:**
1. Visit https://zenodo.org/record/6467836
2. Download FSC22.zip
3. Place in this directory
4. Run: `../download_datasets.sh --fsc22` to extract

**Train:**
```bash
python ../train.py --audio_dir datasets/FSC22 --batch_size 8
```

---

### Xeno-canto Bird Recordings
- **Size**: Variable (you choose)
- **Samples**: 500k+ recordings available worldwide
- **Format**: MP3, variable quality
- **License**: Creative Commons (various)
- **Download**: Via Python API script
- **Use case**: Comprehensive bird sound dataset, customizable by species/region/quality

**Download:**
```bash
# Download 100 high-quality forest bird recordings
python ../download_xenocanto.py --query "forest birds" --quality A --max 100

# Download specific species
python ../download_xenocanto.py --species "Turdus merula" --max 50

# Download by country
python ../download_xenocanto.py --country "Brazil" --type song --max 200

# See all options
python ../download_xenocanto.py --help
```

**Train:**
```bash
python ../train.py --audio_dir datasets/xeno-canto --batch_size 8
```

---

## Dataset Recommendations

### For Natural Sounds (Birds, Forest, etc.)
- **Xeno-canto**: Best for bird sounds, highly customizable
- **FSC22**: Purpose-built for forest environments
- **Bird Audio Detection**: Real-world forest monitoring data
- **ESC-50**: Good starting point, includes nature sounds
- **FSD50K**: Diverse natural sounds

### For General Audio Codec Training
- **FSD50K**: Most comprehensive, diverse audio types
- **ESC-50**: Quick testing and prototyping
- **UrbanSound8K**: Urban environments

### For Speech
- **LibriSpeech**: Clean speech recordings
- **Common Voice**: Larger speech corpus (not included)

## Using Custom Audio

You can also use your own audio files:

1. Create a directory with your audio files:
```
datasets/my_audio/
├── file1.wav
├── file2.mp3
├── file3.flac
└── ...
```

2. Train:
```bash
python ../train.py --audio_dir datasets/my_audio --batch_size 8
```

Supported formats: `.wav`, `.mp3`, `.flac`, `.ogg`

## Dataset Statistics

After downloading, check dataset statistics:

```bash
# Count files in each dataset
find ESC-50-master/audio -name "*.wav" | wc -l
find FSD50K/dev_audio -name "*.wav" | wc -l
find UrbanSound8K/audio -name "*.wav" | wc -l
find LibriSpeech -name "*.flac" | wc -l
```

## Storage Requirements

| Dataset | Compressed | Extracted | Total Files | Best For |
|---------|-----------|-----------|-------------|----------|
| ESC-50 | 600MB | 600MB | 2,000 | Quick testing |
| FSD50K | 80GB | 80GB | 51,197 | Diverse audio |
| UrbanSound8K | 6GB | 6GB | 8,732 | Urban sounds |
| LibriSpeech test | 350MB | 350MB | ~2,600 | Speech |
| Bird Audio Detection | 20GB | 20GB | ~15,000 | Forest monitoring |
| FSC22 | 5GB | 5GB | ~7,000 | Forest classification |
| Xeno-canto | Variable | Variable | Variable | Bird sounds |

**Recommended for Natural Sounds**: 
- Start with **ESC-50** for quick testing
- Use **Xeno-canto** (100-500 files) for bird-specific training
- Use **FSC22** for comprehensive forest environment training
- Use **Bird Audio Detection** for real-world forest monitoring scenarios

## Data Preprocessing

The `dataset.py` module automatically handles:
- ✅ Resampling to 24kHz
- ✅ Mono conversion
- ✅ Chunk extraction (2-3 second segments)
- ✅ Peak normalization
- ✅ Padding/truncation

No manual preprocessing required!

## Troubleshooting

### Download fails
- Check internet connection
- Try manual download for large datasets
- Use `wget -c` or `curl -C -` for resumable downloads

### Extraction fails
- Ensure sufficient disk space
- Check archive integrity: `unzip -t file.zip` or `tar -tzf file.tar.gz`

### Permission denied
- Make script executable: `chmod +x ../download_datasets.sh`
- Check write permissions in datasets directory

## Additional Resources

- **Xeno-canto**: Bird sounds - https://xeno-canto.org/
- **Freesound**: General audio - https://freesound.org/
- **AudioSet**: Large-scale (YouTube) - https://research.google.com/audioset/
- **Common Voice**: Speech - https://commonvoice.mozilla.org/

## License Notes

Each dataset has its own license. Please review before using:
- ESC-50: CC BY-NC 3.0
- FSD50K: Various CC licenses (see dataset documentation)
- UrbanSound8K: CC BY-NC 3.0
- LibriSpeech: CC BY 4.0

Always cite the original datasets in your research!

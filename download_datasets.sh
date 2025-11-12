#!/bin/bash
# Download script for audio datasets
# Supports: ESC-50, FSD50K, UrbanSound8K, LibriSpeech

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default datasets directory
DATASETS_DIR="./datasets"

# Function to check if dataset exists
dataset_exists() {
    local dataset_name=$1
    local check_path=$2

    if [ -d "$check_path" ] && [ "$(ls -A $check_path 2>/dev/null)" ]; then
        echo -e "${GREEN}✓ $dataset_name already downloaded${NC}"
        return 0
    else
        return 1
    fi
}

# Function to download and extract
download_and_extract() {
    local url=$1
    local filename=$2
    local extract_dir=$3

    # Download if not exists
    if [ ! -f "$filename" ]; then
        echo "Downloading $filename..."
        wget -c "$url" -O "$filename" || curl -L -C - "$url" -o "$filename"
    else
        echo "Archive $filename already exists, skipping download"
    fi

    # Extract if not already extracted
    if [ ! -d "$extract_dir" ] || [ ! "$(ls -A $extract_dir 2>/dev/null)" ]; then
        echo "Extracting $filename..."
        case "$filename" in
            *.zip)
                unzip -q "$filename"
                ;;
            *.tar.gz|*.tgz)
                tar -xzf "$filename"
                ;;
            *.tar.bz2)
                tar -xjf "$filename"
                ;;
            *)
                echo "Unknown archive format: $filename"
                return 1
                ;;
        esac
        echo -e "${GREEN}Extracted to $extract_dir${NC}"
    else
        echo "Already extracted: $extract_dir"
    fi
}

# ESC-50: Environmental Sound Classification
download_esc50() {
    echo -e "\n${BLUE}=== ESC-50 Dataset ===${NC}"
    echo "Environmental Sound Classification"
    echo "Size: ~600MB | Samples: 2000 files (5s each) | Classes: 50"

    if dataset_exists "ESC-50" "ESC-50-master/audio"; then
        return 0
    fi

    download_and_extract \
        "https://github.com/karoldvl/ESC-50/archive/master.zip" \
        "ESC-50-master.zip" \
        "ESC-50-master"

    echo -e "${GREEN}✓ ESC-50 ready!${NC}"
    echo "Audio files: ESC-50-master/audio/"
}

# FSD50K: Freesound Dataset 50K
download_fsd50k() {
    echo -e "\n${BLUE}=== FSD50K Dataset ===${NC}"
    echo "Freesound Dataset 50K"
    echo "Size: ~30GB (dev) + ~50GB (eval) | Samples: 51,197 clips | Classes: 200"
    echo ""
    echo -e "${YELLOW}⚠ FSD50K requires manual download from Zenodo${NC}"
    echo "Visit: https://zenodo.org/record/4060432"
    echo ""
    echo "Steps:"
    echo "1. Download FSD50K.dev_audio.zip (~30GB)"
    echo "2. Download FSD50K.eval_audio.zip (~50GB)"
    echo "3. Place them in $(pwd)"
    echo "4. Run this script again"
    echo ""

    # Check if archives exist and extract
    if [ -f "FSD50K.dev_audio.zip" ] || [ -f "FSD50K.eval_audio.zip" ]; then
        echo "Found FSD50K archives, extracting..."

        if [ -f "FSD50K.dev_audio.zip" ] && [ ! -d "FSD50K/dev_audio" ]; then
            echo "Extracting dev_audio..."
            mkdir -p FSD50K
            unzip -q "FSD50K.dev_audio.zip" -d FSD50K/
        fi

        if [ -f "FSD50K.eval_audio.zip" ] && [ ! -d "FSD50K/eval_audio" ]; then
            echo "Extracting eval_audio..."
            mkdir -p FSD50K
            unzip -q "FSD50K.eval_audio.zip" -d FSD50K/
        fi

        echo -e "${GREEN}✓ FSD50K extracted!${NC}"
    else
        echo -e "${YELLOW}FSD50K archives not found. Please download manually.${NC}"
    fi
}

# UrbanSound8K
download_urbansound8k() {
    echo -e "\n${BLUE}=== UrbanSound8K Dataset ===${NC}"
    echo "Urban Sound Classification"
    echo "Size: ~6GB | Samples: 8732 files (<=4s each) | Classes: 10"
    echo ""
    echo -e "${YELLOW}⚠ UrbanSound8K requires manual download${NC}"
    echo "Visit: https://urbansounddataset.weebly.com/urbansound8k.html"
    echo ""
    echo "Steps:"
    echo "1. Fill out the download form"
    echo "2. Download UrbanSound8K.tar.gz"
    echo "3. Place it in $(pwd)"
    echo "4. Run this script again"
    echo ""

    # Check if archive exists and extract
    if [ -f "UrbanSound8K.tar.gz" ]; then
        if [ ! -d "UrbanSound8K" ]; then
            echo "Extracting UrbanSound8K..."
            tar -xzf "UrbanSound8K.tar.gz"
            echo -e "${GREEN}✓ UrbanSound8K extracted!${NC}"
        else
            echo -e "${GREEN}✓ UrbanSound8K already extracted${NC}"
        fi
    else
        echo -e "${YELLOW}UrbanSound8K archive not found. Please download manually.${NC}"
    fi
}

# LibriSpeech test-clean (for speech testing)
download_librispeech() {
    echo -e "\n${BLUE}=== LibriSpeech test-clean ===${NC}"
    echo "Speech corpus (test set only)"
    echo "Size: ~350MB | Samples: ~2600 files"
    echo "Use case: Testing speech reconstruction"

    if dataset_exists "LibriSpeech" "LibriSpeech/test-clean"; then
        return 0
    fi

    download_and_extract \
        "https://www.openslr.org/resources/12/test-clean.tar.gz" \
        "test-clean.tar.gz" \
        "LibriSpeech/test-clean"

    echo -e "${GREEN}✓ LibriSpeech test-clean ready!${NC}"
    echo "Audio files: LibriSpeech/test-clean/"
}

# Bird Audio Detection Challenge (DCASE 2018 Task 3)
download_birdaudio() {
    echo -e "\n${BLUE}=== Bird Audio Detection Challenge ===${NC}"
    echo "Real forest monitoring recordings"
    echo "Size: ~20GB | Samples: ~15,000 10-second clips"
    echo "Use case: Bird detection in real forest environments"
    echo ""
    echo -e "${YELLOW}⚠ Bird Audio Detection requires manual download${NC}"
    echo "Visit: https://dcase.community/challenge2018/task-bird-audio-detection"
    echo ""
    echo "Steps:"
    echo "1. Visit the DCASE 2018 Task 3 page"
    echo "2. Download training and validation sets"
    echo "3. Place archives in $(pwd)"
    echo "4. Run this script again to extract"
    echo ""

    # Check if archives exist and extract
    if [ -f "BirdVox-DCASE-20k.zip" ] || [ -d "BirdVox-DCASE-20k" ]; then
        if [ -f "BirdVox-DCASE-20k.zip" ] && [ ! -d "BirdVox-DCASE-20k" ]; then
            echo "Extracting BirdVox-DCASE-20k..."
            unzip -q "BirdVox-DCASE-20k.zip"
            echo -e "${GREEN}✓ Bird Audio Detection extracted!${NC}"
        else
            echo -e "${GREEN}✓ Bird Audio Detection already extracted${NC}"
        fi
    else
        echo -e "${YELLOW}Bird Audio Detection archive not found. Please download manually.${NC}"
    fi
}

# FSC22: Forest Sound Classification
download_fsc22() {
    echo -e "\n${BLUE}=== FSC22 Dataset ===${NC}"
    echo "Forest Sound Classification"
    echo "Size: ~5GB | Samples: ~7,000 audio clips"
    echo "Classes: Forest-specific sounds (birds, insects, wind, etc.)"
    echo "Use case: Purpose-built for forest environment classification"

    if dataset_exists "FSC22" "FSC22"; then
        return 0
    fi

    echo ""
    echo -e "${YELLOW}⚠ FSC22 requires manual download from Zenodo${NC}"
    echo "Visit: https://zenodo.org/record/6467836"
    echo ""
    echo "Steps:"
    echo "1. Visit https://zenodo.org/record/6467836"
    echo "2. Download FSC22.zip"
    echo "3. Place it in $(pwd)"
    echo "4. Run this script again to extract"
    echo ""

    # Check if archive exists and extract
    if [ -f "FSC22.zip" ]; then
        if [ ! -d "FSC22" ]; then
            echo "Extracting FSC22..."
            unzip -q "FSC22.zip"
            echo -e "${GREEN}✓ FSC22 extracted!${NC}"
        else
            echo -e "${GREEN}✓ FSC22 already extracted${NC}"
        fi
    else
        echo -e "${YELLOW}FSC22 archive not found. Please download manually.${NC}"
    fi
}

# Xeno-canto bird recordings
download_xenocanto() {
    echo -e "\n${BLUE}=== Xeno-canto Bird Recordings ===${NC}"
    echo "500k+ bird recordings worldwide"
    echo "License: Creative Commons (various)"
    echo "Use case: Comprehensive bird sound dataset"
    echo ""
    echo "Xeno-canto requires API access for bulk downloads."
    echo "Use the dedicated Python script for downloading:"
    echo ""
    echo -e "${GREEN}  python download_xenocanto.py --help${NC}"
    echo ""
    echo "Examples:"
    echo "  # Download 100 forest bird recordings"
    echo "  python download_xenocanto.py --query 'forest birds' --max 100"
    echo ""
    echo "  # Download specific species"
    echo "  python download_xenocanto.py --species 'Turdus merula' --max 50"
    echo ""
    echo "  # Download by quality (A = highest)"
    echo "  python download_xenocanto.py --quality A --max 200"
    echo ""
}

# AudioSet (subset via YouTube)
show_audioset_info() {
    echo -e "\n${BLUE}=== AudioSet ===${NC}"
    echo "Large-scale audio event dataset"
    echo ""
    echo -e "${YELLOW}⚠ AudioSet requires downloading from YouTube${NC}"
    echo "Use tools like: youtube-dl or yt-dlp"
    echo "See: https://research.google.com/audioset/download.html"
    echo ""
    echo "Note: Due to complexity, AudioSet is not included in this script."
}

# List available datasets
list_datasets() {
    echo -e "\n${BLUE}=== Available Datasets ===${NC}\n"

    local found=0

    # ESC-50
    if [ -d "ESC-50-master/audio" ]; then
        local count=$(find ESC-50-master/audio -name "*.wav" 2>/dev/null | wc -l)
        echo -e "${GREEN}✓ ESC-50${NC}           $count files    ESC-50-master/audio/"
        found=1
    fi

    # FSD50K
    if [ -d "FSD50K" ]; then
        local dev_count=$(find FSD50K/dev_audio -name "*.wav" 2>/dev/null | wc -l)
        local eval_count=$(find FSD50K/eval_audio -name "*.wav" 2>/dev/null | wc -l)
        local total=$((dev_count + eval_count))
        echo -e "${GREEN}✓ FSD50K${NC}           $total files    FSD50K/"
        found=1
    fi

    # UrbanSound8K
    if [ -d "UrbanSound8K/audio" ]; then
        local count=$(find UrbanSound8K/audio -name "*.wav" 2>/dev/null | wc -l)
        echo -e "${GREEN}✓ UrbanSound8K${NC}     $count files    UrbanSound8K/audio/"
        found=1
    fi

    # LibriSpeech
    if [ -d "LibriSpeech" ]; then
        local count=$(find LibriSpeech -name "*.flac" 2>/dev/null | wc -l)
        echo -e "${GREEN}✓ LibriSpeech${NC}      $count files    LibriSpeech/"
        found=1
    fi

    # Bird Audio Detection
    if [ -d "BirdVox-DCASE-20k" ]; then
        local count=$(find BirdVox-DCASE-20k -name "*.wav" 2>/dev/null | wc -l)
        echo -e "${GREEN}✓ BirdAudio${NC}        $count files    BirdVox-DCASE-20k/"
        found=1
    fi

    # FSC22
    if [ -d "FSC22" ]; then
        local count=$(find FSC22 -name "*.wav" -o -name "*.flac" 2>/dev/null | wc -l)
        echo -e "${GREEN}✓ FSC22${NC}            $count files    FSC22/"
        found=1
    fi

    # Xeno-canto
    if [ -d "xeno-canto" ]; then
        local count=$(find xeno-canto -name "*.mp3" -o -name "*.wav" 2>/dev/null | wc -l)
        echo -e "${GREEN}✓ Xeno-canto${NC}       $count files    xeno-canto/"
        found=1
    fi

    if [ $found -eq 0 ]; then
        echo "No datasets found."
        echo "Run: ./download_datasets.sh --all"
    fi

    echo ""
}

# Show usage
show_usage() {
    cat << EOF
Usage: ./download_datasets.sh [OPTIONS] [DATASETS_DIR]

Options:
    --all           Download all freely available datasets
    --esc50         Download ESC-50 dataset
    --fsd50k        Show FSD50K download instructions
    --urbansound8k  Show UrbanSound8K download instructions
    --librispeech   Download LibriSpeech test-clean
    --birdaudio     Show Bird Audio Detection download instructions
    --fsc22         Show FSC22 download instructions
    --xenocanto     Show Xeno-canto download instructions (use Python script)
    --list          List available datasets
    --help          Show this help message

Examples:
    ./download_datasets.sh --all
    ./download_datasets.sh --esc50 --librispeech
    ./download_datasets.sh --birdaudio --fsc22
    ./download_datasets.sh --list
    ./download_datasets.sh --all ./my_datasets

For Xeno-canto downloads:
    python download_xenocanto.py --query 'forest birds' --max 100

EOF
}

# Parse arguments
if [ $# -eq 0 ]; then
    show_usage
    list_datasets
    exit 0
fi

DOWNLOAD_ALL=0
DOWNLOAD_ESC50=0
DOWNLOAD_FSD50K=0
DOWNLOAD_US8K=0
DOWNLOAD_LIBRI=0
DOWNLOAD_BIRDAUDIO=0
DOWNLOAD_FSC22=0
DOWNLOAD_XENOCANTO=0
LIST_ONLY=0

for arg in "$@"; do
    case $arg in
        --all)
            DOWNLOAD_ALL=1
            ;;
        --esc50)
            DOWNLOAD_ESC50=1
            ;;
        --fsd50k)
            DOWNLOAD_FSD50K=1
            ;;
        --urbansound8k)
            DOWNLOAD_US8K=1
            ;;
        --librispeech)
            DOWNLOAD_LIBRI=1
            ;;
        --birdaudio)
            DOWNLOAD_BIRDAUDIO=1
            ;;
        --fsc22)
            DOWNLOAD_FSC22=1
            ;;
        --xenocanto)
            DOWNLOAD_XENOCANTO=1
            ;;
        --list)
            LIST_ONLY=1
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        -*)
            # Unknown option
            echo "Unknown option: $arg"
            show_usage
            exit 1
            ;;
        *)
            # Assume it's the datasets directory
            DATASETS_DIR="$arg"
            ;;
    esac
done

# Create datasets directory and change to it
mkdir -p "$DATASETS_DIR"
cd "$DATASETS_DIR"

echo -e "${BLUE}=== SoundStream Dataset Downloader ===${NC}"
echo "Datasets directory: $(pwd)"
echo ""

# List and exit if requested
if [ $LIST_ONLY -eq 1 ]; then
    list_datasets
    exit 0
fi

# Download datasets
if [ $DOWNLOAD_ALL -eq 1 ]; then
    download_esc50
    download_librispeech
    download_fsd50k
    download_urbansound8k
    download_birdaudio
    download_fsc22
    download_xenocanto
    show_audioset_info
elif [ $DOWNLOAD_ESC50 -eq 1 ] || [ $DOWNLOAD_FSD50K -eq 1 ] || [ $DOWNLOAD_US8K -eq 1 ] || [ $DOWNLOAD_LIBRI -eq 1 ] || [ $DOWNLOAD_BIRDAUDIO -eq 1 ] || [ $DOWNLOAD_FSC22 -eq 1 ] || [ $DOWNLOAD_XENOCANTO -eq 1 ]; then
    [ $DOWNLOAD_ESC50 -eq 1 ] && download_esc50
    [ $DOWNLOAD_FSD50K -eq 1 ] && download_fsd50k
    [ $DOWNLOAD_US8K -eq 1 ] && download_urbansound8k
    [ $DOWNLOAD_LIBRI -eq 1 ] && download_librispeech
    [ $DOWNLOAD_BIRDAUDIO -eq 1 ] && download_birdaudio
    [ $DOWNLOAD_FSC22 -eq 1 ] && download_fsc22
    [ $DOWNLOAD_XENOCANTO -eq 1 ] && download_xenocanto
fi

# Show summary
echo ""
list_datasets

echo -e "${GREEN}Done!${NC}"
echo ""
echo "To train with a dataset, use:"
echo "  python train.py --audio_dir datasets/ESC-50-master/audio --batch_size 8"

#!/usr/bin/env python3
"""Test if audio files are loading correctly"""
import sys
import torch
import torchaudio
from pathlib import Path

sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)

print("="*60, flush=True)
print("AUDIO FILE LOADING TEST", flush=True)
print("="*60, flush=True)

# Test loading a few ESC-50 files
esc50_dir = Path("./datasets/ESC-50-master/audio")
audio_files = sorted(list(esc50_dir.glob("*.wav")))[:5]

print(f"\nFound {len(list(esc50_dir.glob('*.wav')))} audio files", flush=True)
print(f"\nTesting first 5 files:", flush=True)

for i, audio_file in enumerate(audio_files):
    waveform, sr = torchaudio.load(str(audio_file))

    print(f"\n{i+1}. {audio_file.name}:", flush=True)
    print(f"   Shape: {waveform.shape}", flush=True)
    print(f"   Sample rate: {sr}", flush=True)
    print(f"   Mean: {waveform.mean().item():.6f}", flush=True)
    print(f"   Std: {waveform.std().item():.6f}", flush=True)
    print(f"   Min: {waveform.min().item():.6f}", flush=True)
    print(f"   Max: {waveform.max().item():.6f}", flush=True)

    if waveform.std().item() < 0.001:
        print(f"   ❌ SILENT! Audio has no variance", flush=True)
    else:
        print(f"   ✓ Audio looks valid", flush=True)

print("\n" + "="*60, flush=True)

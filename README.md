# soundpython

soundpython is a Python library for audio file processing that provides a simple, intuitive interface for working with audio data. It supports various audio formats and offers functionality for both mono and stereo audio manipulation.

## Features

- Load and save audio in multiple formats (MP3, WAV, OGG, FLAC)
- Convert between mono and stereo audio
- Manipulate individual audio channels
- Extract portions of audio by time
- Concatenate audio files with optional crossfade
- Create smooth audio transitions with crossfading overlay
- Automatic audio normalization
- Comprehensive metadata handling

## Requirements

- Python 3.10 or higher
- NumPy 2.2.1 or higher
- FFmpeg (must be installed and available in system PATH)

## Installation

Install the package using pip:


```bash
pip install soundpython
```

## Usage

Here are some common usage examples:

```python
from soundpython import Audio

# Load an audio file
audio = Audio.from_file("song.mp3")

# Convert stereo to mono
mono_audio = audio.to_mono()

# Extract left channel from stereo audio
left_channel = audio.get_channel(0)

# Extract portions of audio by time
intro = audio.slice(end_seconds=30.0)  # First 30 seconds
chorus = audio.slice(60.0, 90.0)       # 30-second clip from 1:00 to 1:30
outro = audio.slice(180.0)             # Everything after 3:00

# Simple concatenation of audio segments
combined = intro.concat(chorus)

# Crossfade between two audio segments
# The end of intro will fade out while the start of outro fades in
crossfaded = intro.overlay(outro, fade_duration=2.0)  # 2-second crossfade

# Create silent tracks
stereo_silence = Audio.create_silent(duration_seconds=5.0)  # 5 seconds of stereo silence
mono_silence = Audio.create_silent(duration_seconds=3.0, stereo=False)  # 3 seconds of mono silence

# Save in different formats
audio.save("output.wav")
audio.save("output.mp3")
```

## Audio Metadata

The library provides detailed metadata about audio files through the `AudioMetadata` class:

```python
# Access audio metadata
print(f"Sample rate: {audio.metadata.sample_rate}Hz")
print(f"Channels: {audio.metadata.channels}")
print(f"Duration: {audio.metadata.duration_seconds:.2f}s")
print(f"Bit depth: {audio.metadata.bits_per_sample} bits")
```

## Development

To set up the development environment:

0. Install `uv`
1. Clone the repository
2. Install development dependencies:
```bash
uv sync --dev
```
3. Run tests:
```bash
uv run pytest
```
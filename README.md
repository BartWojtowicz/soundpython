# soundpython

soundpython is a Python library for audio file processing that provides a simple, intuitive interface for working with audio data. It supports various audio formats and offers functionality for both mono and stereo audio manipulation.

## Features

- Load and save audio in multiple formats (MP3, WAV, OGG, FLAC)
- Convert between mono and stereo audio
- Manipulate individual audio channels
- Concatenate audio files
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

# Concatenate two audio files
combined = audio1.concat(audio2)

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
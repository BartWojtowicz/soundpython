import io
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class AudioMetadata:
    """Stores metadata for audio files"""

    sample_rate: int
    channels: int
    sample_width: int  # in bytes
    duration_seconds: float
    frame_count: int

    @property
    def bits_per_sample(self) -> int:
        """Returns the number of bits per sample"""
        return self.sample_width * 8


class AudioLoadError(Exception):
    """Raised when there's an error loading audio"""

    pass


class Audio:
    """
    A class to handle audio data with numpy arrays

    Attributes:
        data (np.ndarray): Audio data as a numpy array, normalized between -1 and 1
        metadata (AudioMetadata): Metadata about the audio file
    """

    def __init__(self, data: np.ndarray, metadata: AudioMetadata):
        """
        Initialize Audio object

        Args:
            data: Audio data as numpy array, normalized between -1 and 1
            metadata: AudioMetadata object containing audio properties
        """
        self.data = data
        self.metadata = metadata

    @staticmethod
    def _get_ffmpeg_info(file_path: Path) -> dict:
        """Get audio metadata using ffprobe"""
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(file_path),
        ]

        try:
            output = subprocess.check_output(cmd)
            info = eval(output)  # Safe since we control the input from ffprobe

            # Find the audio stream
            audio_stream = None
            for stream in info["streams"]:
                if stream["codec_type"] == "audio":
                    audio_stream = stream
                    break

            if audio_stream is None:
                raise AudioLoadError("No audio stream found")

            return {
                "sample_rate": int(audio_stream["sample_rate"]),
                "channels": int(audio_stream["channels"]),
                "duration": float(info["format"]["duration"]),
                "bit_depth": int(audio_stream.get("bits_per_sample", 16)),
            }
        except subprocess.CalledProcessError as e:
            raise AudioLoadError(f"Error getting audio info: {e}")

    # Add this method to the Audio class

    @classmethod
    def create_silent(
        cls, duration_seconds: float, stereo: bool = True, sample_rate: int = 44100, sample_width: int = 2
    ) -> "Audio":
        """
        Create a silent audio track.

        Args:
            duration_seconds: Length of the silent track in seconds
            stereo: If True, create stereo track; if False, create mono track (default: True)
            sample_rate: Sample rate in Hz (default: 44100)
            sample_width: Sample width in bytes (default: 2, which is 16-bit)

        Returns:
            Audio: New Audio instance with silent track

        Raises:
            ValueError: If duration is negative or other parameters are invalid
        """
        if duration_seconds <= 0:
            raise ValueError("Duration must be positive")
        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        if sample_width not in {1, 2, 4}:
            raise ValueError("Sample width must be 1, 2, or 4 bytes")

        # Calculate number of frames
        frame_count = int(duration_seconds * sample_rate)

        # Create silent data array
        channels = 2 if stereo else 1
        shape = (frame_count, channels) if stereo else (frame_count,)
        data = np.zeros(shape, dtype=np.float32)

        # Create metadata
        metadata = AudioMetadata(
            sample_rate=sample_rate,
            channels=channels,
            sample_width=sample_width,
            duration_seconds=duration_seconds,
            frame_count=frame_count,
        )

        return cls(data, metadata)

    @classmethod
    def from_file(cls, file_path: str | Path) -> "Audio":
        """
        Load audio from a file using ffmpeg

        Args:
            file_path: Path to the audio file

        Returns:
            Audio: New Audio instance

        Raises:
            FileNotFoundError: If the file doesn't exist
            AudioLoadError: If there's an error loading the audio
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get audio info
        info = cls._get_ffmpeg_info(file_path)

        # Convert to WAV using ffmpeg
        cmd = [
            "ffmpeg",
            "-i",
            str(file_path),
            "-f",
            "wav",
            "-ar",
            str(info["sample_rate"]),  # sample rate
            "-ac",
            str(info["channels"]),  # channels
            "-bits_per_raw_sample",
            str(info["bit_depth"]),
            "-",  # Output to stdout
        ]

        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            wav_data, stderr = process.communicate()

            if process.returncode != 0:
                raise AudioLoadError(f"FFmpeg error: {stderr.decode()}")

            # Read WAV data
            with io.BytesIO(wav_data) as wav_io:
                with wave.open(wav_io, "rb") as wav_file:
                    # Get WAV metadata
                    sample_width = wav_file.getsampwidth()
                    channels = wav_file.getnchannels()
                    sample_rate = wav_file.getframerate()
                    n_frames = wav_file.getnframes()

                    # Read raw audio data
                    raw_data = wav_file.readframes(n_frames)

                    # Convert bytes to numpy array based on sample width
                    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
                    dtype = dtype_map.get(sample_width)
                    if dtype is None:
                        raise AudioLoadError(f"Unsupported sample width: {sample_width}")

                    data = np.frombuffer(raw_data, dtype=dtype)

                    # Reshape if stereo
                    if channels == 2:
                        data = data.reshape(-1, 2)

                    # Convert to float32
                    data = data.astype(np.float32)

                    # Reshape before normalization if stereo
                    if channels == 2:
                        data = data.reshape(-1, 2)

                    # Normalize to float between -1 and 1
                    max_value = float(np.iinfo(dtype).max)
                    data = data / max_value

                    # Ensure normalization is within bounds due to floating point precision
                    data = np.clip(data, -1.0, 1.0)

                    # Calculate frame count from actual data length
                    # For stereo, len(data) is already correct after reshape
                    frame_count = len(data)

                    metadata = AudioMetadata(
                        sample_rate=sample_rate,
                        channels=channels,
                        sample_width=sample_width,
                        duration_seconds=info["duration"],
                        frame_count=frame_count,
                    )

                    return cls(data, metadata)

        except subprocess.CalledProcessError as e:
            raise AudioLoadError(f"Error running ffmpeg: {e}")

    def to_mono(self) -> "Audio":
        """
        Convert stereo audio to mono by averaging channels

        Returns:
            Audio: New Audio instance with mono audio
        """
        if self.metadata.channels == 1:
            return self

        mono_data = self.data.mean(axis=1)

        new_metadata = AudioMetadata(
            sample_rate=self.metadata.sample_rate,
            channels=1,
            sample_width=self.metadata.sample_width,
            duration_seconds=self.metadata.duration_seconds,
            frame_count=len(mono_data),
        )

        return Audio(mono_data, new_metadata)

    def get_channel(self, channel: int) -> "Audio":
        """
        Extract a single channel from the audio

        Args:
            channel: Channel number (0 for left, 1 for right)

        Returns:
            Audio: New Audio instance with single channel

        Raises:
            ValueError: If channel number is invalid
        """
        if self.metadata.channels == 1:
            return self

        if channel not in [0, 1]:
            raise ValueError("Channel must be 0 (left) or 1 (right)")

        channel_data = self.data[:, channel]

        new_metadata = AudioMetadata(
            sample_rate=self.metadata.sample_rate,
            channels=1,
            sample_width=self.metadata.sample_width,
            duration_seconds=self.metadata.duration_seconds,
            frame_count=len(channel_data),
        )

        return Audio(channel_data, new_metadata)

    def concat(self, other: "Audio") -> "Audio":
        """
        Concatenate another audio segment to this one.
        Audio metadata must match (sample rate, channels, etc.)

        Args:
            other: Another Audio object to concatenate

        Returns:
            Audio: New Audio object with concatenated data

        Raises:
            ValueError: If audio metadata doesn't match
        """
        # Validate matching metadata
        if self.metadata.channels != other.metadata.channels:
            raise ValueError("Channel counts must match")
        if abs(self.metadata.sample_rate - other.metadata.sample_rate) > 0:  # Exact match required
            raise ValueError("Sample rates must match")
        if self.metadata.sample_width != other.metadata.sample_width:
            raise ValueError("Sample widths must match")

        # Concatenate the data
        if self.metadata.channels == 1:
            concatenated_data = np.concatenate([self.data, other.data])
        else:
            concatenated_data = np.vstack([self.data, other.data])

        # Create new metadata
        new_metadata = AudioMetadata(
            sample_rate=self.metadata.sample_rate,
            channels=self.metadata.channels,
            sample_width=self.metadata.sample_width,
            duration_seconds=self.metadata.duration_seconds + other.metadata.duration_seconds,
            frame_count=len(concatenated_data),
        )

        return Audio(concatenated_data, new_metadata)

    def slice(self, start_seconds: float = 0.0, end_seconds: float | None = None) -> "Audio":
        """
        Extract a portion of the audio between start_seconds and end_seconds.

        Args:
            start_seconds: Start time in seconds (default: 0.0)
            end_seconds: End time in seconds (default: None, meaning end of audio)

        Returns:
            Audio: New Audio instance with the extracted portion

        Raises:
            ValueError: If start_seconds or end_seconds are invalid
        """
        # Validate inputs
        if start_seconds < 0:
            raise ValueError("start_seconds must be non-negative")

        if end_seconds is not None:
            if end_seconds < start_seconds:
                raise ValueError("end_seconds must be greater than start_seconds")
            if end_seconds > self.metadata.duration_seconds:
                raise ValueError("end_seconds cannot exceed audio duration")
        else:
            end_seconds = self.metadata.duration_seconds

        # Convert seconds to sample indices
        start_idx = int(start_seconds * self.metadata.sample_rate)
        end_idx = int(end_seconds * self.metadata.sample_rate)

        # Extract the portion of audio data
        sliced_data = self.data[start_idx:end_idx]

        # Calculate new duration
        new_duration = (end_idx - start_idx) / self.metadata.sample_rate

        # Create new metadata
        new_metadata = AudioMetadata(
            sample_rate=self.metadata.sample_rate,
            channels=self.metadata.channels,
            sample_width=self.metadata.sample_width,
            duration_seconds=new_duration,
            frame_count=len(sliced_data) if self.metadata.channels == 1 else len(sliced_data),
        )

        return Audio(sliced_data, new_metadata)

    def overlay(self, other: "Audio", fade_duration: float) -> "Audio":
        """
        Overlay another audio segment onto this one with crossfading.
        The end of the first audio will fade out while the start of the second audio fades in.

        Args:
            other: Another Audio object to overlay
            fade_duration: Duration of the crossfade in seconds

        Returns:
            Audio: New Audio object with overlaid audio and crossfade

        Raises:
            ValueError: If audio metadata doesn't match or fade duration is invalid
        """
        # Validate matching metadata
        if self.metadata.channels != other.metadata.channels:
            raise ValueError("Channel counts must match")
        if abs(self.metadata.sample_rate - other.metadata.sample_rate) > 0:  # Exact match required
            raise ValueError("Sample rates must match")
        if self.metadata.sample_width != other.metadata.sample_width:
            raise ValueError("Sample widths must match")

        # Validate fade duration more strictly
        if fade_duration <= 0:
            raise ValueError("Fade duration must be positive")
        if fade_duration >= min(self.metadata.duration_seconds, other.metadata.duration_seconds):
            raise ValueError("Fade duration cannot exceed the duration of either audio segment")

        # Rest of the implementation remains the same...
        fade_length = int(fade_duration * self.metadata.sample_rate)
        total_length = len(self.data) + len(other.data) - fade_length

        if self.metadata.channels == 1:
            output = np.zeros(total_length, dtype=np.float32)
        else:
            output = np.zeros((total_length, 2), dtype=np.float32)

        fade_start_idx = len(self.data) - fade_length
        output[:fade_start_idx] = self.data[:fade_start_idx]
        output[fade_start_idx + fade_length :] = other.data[fade_length:]

        fade_out = np.linspace(1, 0, fade_length)
        fade_in = np.linspace(0, 1, fade_length)

        if self.metadata.channels == 1:
            output[fade_start_idx : fade_start_idx + fade_length] = (
                self.data[fade_start_idx:] * fade_out + other.data[:fade_length] * fade_in
            )
        else:
            for channel in range(2):
                output[fade_start_idx : fade_start_idx + fade_length, channel] = (
                    self.data[fade_start_idx:, channel] * fade_out + other.data[:fade_length, channel] * fade_in
                )

        new_metadata = AudioMetadata(
            sample_rate=self.metadata.sample_rate,
            channels=self.metadata.channels,
            sample_width=self.metadata.sample_width,
            duration_seconds=(total_length / self.metadata.sample_rate),
            frame_count=total_length,
        )

        return Audio(output, new_metadata)

    def save(self, file_path: str | Path, format: str = None) -> None:
        """
        Save audio to a file using ffmpeg

        Args:
            file_path: Path to save the audio file
            format: Output format (e.g., 'mp3', 'wav'). If None, inferred from extension.
        """
        file_path = Path(file_path)

        # Convert data back to int16
        int_data = (self.data * np.iinfo(np.int16).max).astype(np.int16)

        # Create WAV in memory
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wav_file:
            wav_file.setnchannels(self.metadata.channels)
            wav_file.setsampwidth(self.metadata.sample_width)
            wav_file.setframerate(self.metadata.sample_rate)
            wav_file.writeframes(int_data.tobytes())

        wav_io.seek(0)

        # Check and infer format
        if format is None:
            format = file_path.suffix[1:]  # Remove the dot

        # Validate format
        SUPPORTED_FORMATS = {"mp3", "wav", "ogg", "flac"}
        if format not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {format}. Supported formats are: {', '.join(SUPPORTED_FORMATS)}")

        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file
            "-f",
            "wav",  # Input format
            "-i",
            "-",  # Read from stdin
        ]

        if format:
            cmd.extend(["-f", format])

        cmd.append(str(file_path))

        try:
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            _, stderr = process.communicate(wav_io.getvalue())

            if process.returncode != 0:
                raise AudioLoadError(f"Error saving audio: {stderr.decode()}")

        except subprocess.CalledProcessError as e:
            raise AudioLoadError(f"Error running ffmpeg: {e}")

    def __len__(self) -> int:
        """Returns the number of samples"""
        return self.metadata.frame_count

    def __repr__(self) -> str:
        """String representation of the Audio object"""
        return (
            f"Audio(channels={self.metadata.channels}, "
            f"sample_rate={self.metadata.sample_rate}Hz, "
            f"duration={self.metadata.duration_seconds:.2f}s)"
        )

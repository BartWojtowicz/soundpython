import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Tuple
import subprocess
import io
import struct
import wave

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
    
    def __init__(self, 
                 data: np.ndarray,
                 metadata: AudioMetadata):
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
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            str(file_path)
        ]
        
        try:
            output = subprocess.check_output(cmd)
            info = eval(output)  # Safe since we control the input from ffprobe
            
            # Find the audio stream
            audio_stream = None
            for stream in info['streams']:
                if stream['codec_type'] == 'audio':
                    audio_stream = stream
                    break
            
            if audio_stream is None:
                raise AudioLoadError("No audio stream found")
                
            return {
                'sample_rate': int(audio_stream['sample_rate']),
                'channels': int(audio_stream['channels']),
                'duration': float(info['format']['duration']),
                'bit_depth': int(audio_stream.get('bits_per_sample', 16))
            }
        except subprocess.CalledProcessError as e:
            raise AudioLoadError(f"Error getting audio info: {e}")
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'Audio':
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
            'ffmpeg',
            '-i', str(file_path),
            '-f', 'wav',
            '-ar', str(info['sample_rate']),  # sample rate
            '-ac', str(info['channels']),     # channels
            '-bits_per_raw_sample', str(info['bit_depth']),
            '-'  # Output to stdout
        ]
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            wav_data, stderr = process.communicate()
            
            if process.returncode != 0:
                raise AudioLoadError(f"FFmpeg error: {stderr.decode()}")
            
            # Read WAV data
            with io.BytesIO(wav_data) as wav_io:
                with wave.open(wav_io, 'rb') as wav_file:
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
                        duration_seconds=info['duration'],
                        frame_count=frame_count
                    )
                    
                    return cls(data, metadata)
                    
        except subprocess.CalledProcessError as e:
            raise AudioLoadError(f"Error running ffmpeg: {e}")
    
    def to_mono(self) -> 'Audio':
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
            frame_count=len(mono_data)
        )
        
        return Audio(mono_data, new_metadata)
    
    def get_channel(self, channel: int) -> 'Audio':
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
            frame_count=len(channel_data)
        )
        
        return Audio(channel_data, new_metadata)
    
    def concat(self, other: 'Audio') -> 'Audio':
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
        if self.metadata.sample_rate != other.metadata.sample_rate:
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
            frame_count=len(concatenated_data)
        )
        
        return Audio(concatenated_data, new_metadata)        # Add format-specific options
    
    def save(self, file_path: Union[str, Path], format: str = None) -> None:
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
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(self.metadata.channels)
            wav_file.setsampwidth(self.metadata.sample_width)
            wav_file.setframerate(self.metadata.sample_rate)
            wav_file.writeframes(int_data.tobytes())
        
        wav_io.seek(0)
        
        # Check and infer format
        if format is None:
            format = file_path.suffix[1:]  # Remove the dot
        
        # Validate format
        SUPPORTED_FORMATS = {'mp3', 'wav', 'ogg', 'flac'}
        if format not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {format}. Supported formats are: {', '.join(SUPPORTED_FORMATS)}")
            
        # Build ffmpeg command
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-f', 'wav',  # Input format
            '-i', '-',    # Read from stdin
        ]
        
        if format:
            cmd.extend(['-f', format])
            
        cmd.append(str(file_path))
        
        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
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
        return (f"Audio(channels={self.metadata.channels}, "
                f"sample_rate={self.metadata.sample_rate}Hz, "
                f"duration={self.metadata.duration_seconds:.2f}s)")
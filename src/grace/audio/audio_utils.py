"""
Grace AI System - Audio Utilities Module

This module provides common audio utility functions with improved error
handling and resource management.
"""

import logging
import threading
import numpy as np
import collections
from typing import Dict, List, Optional, Tuple, Union, Set
from contextlib import contextmanager

# Try to import optional audio dependencies
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

try:
    import webrtcvad
    WEBRTCVAD_AVAILABLE = True
except ImportError:
    WEBRTCVAD_AVAILABLE = False


@contextmanager
def audio_device_context():
    """Context manager for audio device operations to ensure proper cleanup."""
    current_devices = set()
    try:
        yield current_devices
    finally:
        # Clean up any remaining devices
        for device in current_devices:
            try:
                if hasattr(device, 'close'):
                    device.close()
            except Exception:
                pass


def get_device_list() -> List[Dict]:
    """
    Get a list of available audio devices with improved error handling.
    
    Returns:
        List of audio device information
    """
    devices = []
    
    if not SOUNDDEVICE_AVAILABLE:
        return devices
        
    try:
        device_list = sd.query_devices()
        for i, device in enumerate(device_list):
            device_type = "unknown"
            if device.get('max_input_channels', 0) > 0:
                device_type = "input"
            elif device.get('max_output_channels', 0) > 0:
                device_type = "output"
                
            devices.append({
                'index': i,
                'name': device.get('name', 'Unknown'),
                'type': device_type,
                'channels': max(device.get('max_input_channels', 0), 
                               device.get('max_output_channels', 0)),
                'default': (device.get('name') == sd.query_devices(kind='input').get('name'))
            })
    except Exception as e:
        logging.getLogger('grace.audio.utils').error(f"Error getting audio devices: {e}")
        
    return devices


def convert_audio_format(audio_data: np.ndarray, source_format: str, target_format: str) -> np.ndarray:
    """
    Convert audio data between different formats with proper error handling.
    
    Args:
        audio_data: Audio data as numpy array
        source_format: Source format (e.g., 'float32', 'int16')
        target_format: Target format (e.g., 'float32', 'int16')
        
    Returns:
        Converted audio data
    """
    if source_format == target_format:
        return audio_data
        
    try:
        # Convert float32 (range -1 to 1) to int16 (range -32768 to 32767)
        if source_format == 'float32' and target_format == 'int16':
            return (audio_data * 32767).astype(np.int16)
            
        # Convert int16 (range -32768 to 32767) to float32 (range -1 to 1)
        elif source_format == 'int16' and target_format == 'float32':
            return (audio_data / 32767).astype(np.float32)
            
        # Handle other conversions
        elif source_format == 'int8' and target_format == 'float32':
            return (audio_data / 127).astype(np.float32)
            
        elif source_format == 'float32' and target_format == 'int8':
            return (audio_data * 127).astype(np.int8)
            
        elif source_format == 'int16' and target_format == 'int8':
            return (audio_data / 256).astype(np.int8)
            
        elif source_format == 'int8' and target_format == 'int16':
            return (audio_data * 256).astype(np.int16)
            
        # Unsupported conversion
        else:
            logging.getLogger('grace.audio.utils').warning(
                f"Unsupported audio format conversion: {source_format} to {target_format}")
            return audio_data
    except Exception as e:
        logging.getLogger('grace.audio.utils').error(
            f"Error converting audio format from {source_format} to {target_format}: {e}")
        # Return original data if conversion fails
        return audio_data


def detect_silence(audio_data: np.ndarray, threshold: float = 0.01, format: str = 'float32') -> bool:
    """
    Detect if audio contains silence with proper error handling.
    
    Args:
        audio_data: Audio data as numpy array
        threshold: Silence threshold
        format: Audio data format
        
    Returns:
        True if audio is silent, False otherwise
    """
    try:
        # Handle empty input
        if audio_data is None or len(audio_data) == 0:
            return True
            
        # Convert to float32 if needed for consistent processing
        if format != 'float32':
            audio_data = convert_audio_format(audio_data, format, 'float32')
            
        # Calculate RMS amplitude
        rms = np.sqrt(np.mean(np.square(audio_data)))
        
        # Compare with threshold
        return rms < threshold
    except Exception as e:
        logging.getLogger('grace.audio.utils').error(f"Error detecting silence: {e}")
        # Default to not silent if detection fails
        return False


def normalize_audio(audio_data: np.ndarray, target_level: float = 0.5, format: str = 'float32') -> np.ndarray:
    """
    Normalize audio to a target level with improved error handling.
    
    Args:
        audio_data: Audio data as numpy array
        target_level: Target normalization level (0-1 for float32)
        format: Audio data format
        
    Returns:
        Normalized audio data
    """
    try:
        # Handle empty input
        if audio_data is None or len(audio_data) == 0:
            return audio_data
            
        # Convert to float32 if needed for consistent processing
        if format != 'float32':
            audio_data = convert_audio_format(audio_data, format, 'float32')
            
        # Calculate current peak level
        peak = np.max(np.abs(audio_data))
        
        # Avoid division by zero
        if peak > 0:
            # Normalize
            normalized = audio_data * (target_level / peak)
            return normalized
        else:
            return audio_data
    except Exception as e:
        logging.getLogger('grace.audio.utils').error(f"Error normalizing audio: {e}")
        # Return original data if normalization fails
        return audio_data


def trim_silence(audio_data: np.ndarray, threshold: float = 0.01, 
               min_silence_duration: float = 0.1, sample_rate: int = 16000,
               format: str = 'float32') -> np.ndarray:
    """
    Trim silence from the beginning and end of audio with improved error handling.
    
    Args:
        audio_data: Audio data as numpy array
        threshold: Silence threshold
        min_silence_duration: Minimum silence duration to trim (seconds)
        sample_rate: Audio sample rate
        format: Audio data format
        
    Returns:
        Trimmed audio data
    """
    try:
        # Handle empty input
        if audio_data is None or len(audio_data) == 0:
            return audio_data
            
        # Convert to float32 if needed for consistent processing
        if format != 'float32':
            audio_data = convert_audio_format(audio_data, format, 'float32')
            
        # Calculate energy (absolute value of samples)
        energy = np.abs(audio_data)
        
        # Minimum number of samples to consider as silence
        min_silence_samples = int(min_silence_duration * sample_rate)
        
        # Ensure min_silence_samples is at least 1
        min_silence_samples = max(1, min_silence_samples)
        
        # Ensure we don't exceed audio length
        if min_silence_samples >= len(audio_data):
            return audio_data
        
        # Find start index (first non-silent sample)
        start_idx = 0
        for i in range(len(audio_data) - min_silence_samples):
            if np.mean(energy[i:i+min_silence_samples]) > threshold:
                start_idx = i
                break
                
        # Find end index (last non-silent sample)
        end_idx = len(audio_data)
        for i in range(len(audio_data) - min_silence_samples, 0, -1):
            if np.mean(energy[i:i+min_silence_samples]) > threshold:
                end_idx = i + min_silence_samples
                break
                
        # Ensure we have valid indices
        if start_idx >= end_idx:
            # If trimming would remove everything, return original
            return audio_data
        
        # Return trimmed audio
        return audio_data[start_idx:end_idx]
    except Exception as e:
        logging.getLogger('grace.audio.utils').error(f"Error trimming silence: {e}")
        # Return original data if trimming fails
        return audio_data


class VADProcessor:
    """Process audio with Voice Activity Detection and robust error handling."""
    
    def __init__(self, aggressiveness=3, sample_rate=16000, frame_duration_ms=30):
        """Initialize VAD with parameters."""
        self.logger = logging.getLogger('grace.audio.vad')
        
        if not WEBRTCVAD_AVAILABLE:
            self.logger.error("webrtcvad not available. Install with: pip install webrtcvad")
            self.vad = None
            return
            
        try:
            self.vad = webrtcvad.Vad(aggressiveness)
            self.sample_rate = sample_rate
            self.frame_duration_ms = frame_duration_ms
            
            # Validate parameters
            if sample_rate not in (8000, 16000, 32000, 48000):
                raise ValueError(f"Sample rate must be 8000, 16000, 32000, or 48000 Hz")
            if frame_duration_ms not in (10, 20, 30):
                raise ValueError(f"Frame duration must be 10, 20, or 30 ms")
            
            # Calculate frame size in bytes
            self.frame_size = int(sample_rate * frame_duration_ms / 1000) * 2  # 2 bytes per sample
            
            self.logger.info(f"VAD initialized with aggressiveness {aggressiveness}")
        except Exception as e:
            self.logger.error(f"Failed to initialize VAD: {e}")
            self.vad = None
    
    def process_audio_file(self, audio_path):
        """Process an audio file and extract voiced segments with error handling."""
        if not self.vad:
            self.logger.error("VAD not initialized")
            return []
            
        try:
            # Read audio file
            import wave
            with wave.open(audio_path, 'rb') as wf:
                # Validate audio format
                if wf.getnchannels() != 1:
                    raise ValueError("Audio must be mono")
                if wf.getsampwidth() != 2:
                    raise ValueError("Audio must be 16-bit")
                if wf.getframerate() != self.sample_rate:
                    raise ValueError(f"Audio must have {self.sample_rate}Hz sample rate")
                
                # Read all audio data
                pcm_data = wf.readframes(wf.getnframes())
            
            # Process frames
            frames = self._frame_generator(pcm_data)
            voiced_segments = self._vad_collector(frames)
            
            return voiced_segments
        except Exception as e:
            self.logger.error(f"VAD processing failed: {e}")
            return []
    
    def _frame_generator(self, audio_data):
        """Generate audio frames from PCM data."""
        offset = 0
        while offset + self.frame_size <= len(audio_data):
            yield audio_data[offset:offset + self.frame_size]
            offset += self.frame_size
    
    def _vad_collector(self, frames, padding_duration_ms=300):
        """Collect voiced segments from a stream of audio frames."""
        if not self.vad:
            return []
            
        num_padding_frames = int(padding_duration_ms / self.frame_duration_ms)
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False
        voiced_segments = []
        
        for frame in frames:
            try:
                is_speech = self.vad.is_speech(frame, self.sample_rate)
                
                if not triggered:
                    ring_buffer.append((frame, is_speech))
                    num_voiced = len([f for f, speech in ring_buffer if speech])
                    
                    # Start collecting if enough voiced frames in buffer
                    if num_voiced > 0.9 * ring_buffer.maxlen:
                        triggered = True
                        voiced_segments.append([])
                        for f, s in ring_buffer:
                            voiced_segments[-1].append(f)
                        ring_buffer.clear()
                else:
                    # Continue collecting if still in voice segment
                    voiced_segments[-1].append(frame)
                    ring_buffer.append((frame, is_speech))
                    num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                    
                    # Stop collecting if enough unvoiced frames
                    if num_unvoiced > 0.9 * ring_buffer.maxlen:
                        triggered = False
                        ring_buffer.clear()
            except Exception as e:
                self.logger.debug(f"Error processing VAD frame: {e}")
                continue
        
        # Convert segments to byte strings
        return [''.join(segment) for segment in voiced_segments]
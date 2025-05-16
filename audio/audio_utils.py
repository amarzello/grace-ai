"""
Grace AI System - Audio Utilities Module

This module provides common audio utility functions for the Grace AI system.
"""

import logging
import numpy as np
from typing import Dict, List, Optional

# Try to import optional audio dependencies
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False


def get_device_list() -> List[Dict]:
    """
    Get a list of available audio devices.
    
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
    Convert audio data between different formats.
    
    Args:
        audio_data: Audio data as numpy array
        source_format: Source format (e.g., 'float32', 'int16')
        target_format: Target format (e.g., 'float32', 'int16')
        
    Returns:
        Converted audio data
    """
    if source_format == target_format:
        return audio_data
        
    # Convert float32 (range -1 to 1) to int16 (range -32768 to 32767)
    if source_format == 'float32' and target_format == 'int16':
        return (audio_data * 32767).astype(np.int16)
        
    # Convert int16 (range -32768 to 32767) to float32 (range -1 to 1)
    elif source_format == 'int16' and target_format == 'float32':
        return (audio_data / 32767).astype(np.float32)
        
    # Handle other conversions if needed
    else:
        logging.getLogger('grace.audio.utils').warning(
            f"Unsupported audio format conversion: {source_format} to {target_format}")
        return audio_data


def detect_silence(audio_data: np.ndarray, threshold: float = 0.01, format: str = 'float32') -> bool:
    """
    Detect if audio contains silence.
    
    Args:
        audio_data: Audio data as numpy array
        threshold: Silence threshold
        format: Audio data format
        
    Returns:
        True if audio is silent, False otherwise
    """
    # Convert to float32 if needed for consistent processing
    if format != 'float32':
        audio_data = convert_audio_format(audio_data, format, 'float32')
        
    # Calculate RMS amplitude
    rms = np.sqrt(np.mean(np.square(audio_data)))
    
    # Compare with threshold
    return rms < threshold


def normalize_audio(audio_data: np.ndarray, target_level: float = 0.5, format: str = 'float32') -> np.ndarray:
    """
    Normalize audio to a target level.
    
    Args:
        audio_data: Audio data as numpy array
        target_level: Target normalization level (0-1 for float32)
        format: Audio data format
        
    Returns:
        Normalized audio data
    """
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


def trim_silence(audio_data: np.ndarray, threshold: float = 0.01, 
               min_silence_duration: float = 0.1, sample_rate: int = 16000,
               format: str = 'float32') -> np.ndarray:
    """
    Trim silence from the beginning and end of audio.
    
    Args:
        audio_data: Audio data as numpy array
        threshold: Silence threshold
        min_silence_duration: Minimum silence duration to trim (seconds)
        sample_rate: Audio sample rate
        format: Audio data format
        
    Returns:
        Trimmed audio data
    """
    # Convert to float32 if needed for consistent processing
    if format != 'float32':
        audio_data = convert_audio_format(audio_data, format, 'float32')
        
    # Calculate energy
    energy = np.abs(audio_data)
    
    # Minimum number of samples to consider as silence
    min_silence_samples = int(min_silence_duration * sample_rate)
    
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
            
    # Return trimmed audio
    return audio_data[start_idx:end_idx]

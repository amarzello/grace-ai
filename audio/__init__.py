"""
Grace AI System - Audio Package

This package contains audio-related modules for the Grace AI assistant.
"""

from .audio_system import AudioSystem
from .audio_input import AudioInput
from .audio_output import AudioOutput
from .speech_recognition import SpeechRecognizer
from .audio_utils import (
    get_device_list, convert_audio_format, detect_silence,
    normalize_audio, trim_silence
)

__all__ = [
    'AudioSystem',
    'AudioInput',
    'AudioOutput',
    'SpeechRecognizer',
    'get_device_list',
    'convert_audio_format',
    'detect_silence',
    'normalize_audio',
    'trim_silence'
]

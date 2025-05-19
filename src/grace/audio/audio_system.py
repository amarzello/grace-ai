"""
Grace AI System - Audio System Module

This module integrates the audio components with improved error handling
and resource management.
"""

import logging
import time
import asyncio
import gc
import numpy as np
from typing import Dict, Optional, Tuple, Union, List
from contextlib import asynccontextmanager

# Import audio components
from .audio_input import AudioInput
from .audio_output import AudioOutput
from .speech_recognition import SpeechRecognizer
from .audio_utils import trim_silence, normalize_audio


class AudioSystem:
    """
    Integrated audio system for Grace AI with proper resource management.
    
    This class combines audio input, output, and speech recognition
    to provide a complete audio interface for the Grace AI system.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the audio system with the provided configuration.
        
        Args:
            config: Audio system configuration
        """
        self.logger = logging.getLogger('grace.audio')
        self.audio_config = config.get('audio', {})
        
        # Flag to track if the system is enabled
        self.enabled = not self.audio_config.get('disable_audio', False)
        
        if not self.enabled:
            self.logger.info("Audio system is disabled")
            # Initialize empty components to avoid NoneType errors
            self.input = None
            self.output = None
            self.recognizer = None
            self.whisper_model = None
            return
        
        try:
            # Initialize components
            self.logger.info("Initializing audio components")
            
            # Prioritize component initialization to minimize startup time
            # Initialize output first, as it doesn't depend on other components
            self.output = AudioOutput(config)
            
            # Initialize input next, as it's needed for recording
            self.input = AudioInput(config)
            
            # Initialize recognizer last, as it's the most resource-intensive
            self.recognizer = SpeechRecognizer(config)
            
            # Store whisper model reference for status checks
            self.whisper_model = self.recognizer.whisper_model if self.recognizer else None
            
            self.logger.info("Audio system initialized")
        except Exception as e:
            self.logger.error(f"Error initializing audio system: {e}")
            # In case of initialization error, disable the system
            self.enabled = False
            self.input = None
            self.output = None
            self.recognizer = None
            self.whisper_model = None
    
    def start_listening(self) -> bool:
        """
        Start listening for audio input.
        
        Returns:
            Success status
        """
        if not self.enabled or not self.input:
            self.logger.warning("Cannot start listening: Audio system is disabled")
            return False
            
        return self.input.start_listening()
    
    def stop_listening(self):
        """Stop listening for audio input."""
        if self.enabled and self.input:
            self.input.stop_listening()
    
    def listen_and_transcribe(self) -> Tuple[str, Optional[np.ndarray]]:
        """
        Listen for speech and transcribe it.
        
        Returns:
            Tuple of (transcribed_text, audio_data)
        """
        if not self.enabled or not self.input or not self.recognizer:
            self.logger.warning("Cannot listen and transcribe: Audio system is disabled")
            return "", None
            
        # Listen for command
        audio_data = self.input.listen_for_command()
        if audio_data is None or len(audio_data) == 0:
            return "", None
            
        # Process audio
        if self.audio_config.get('trim_silence', True):
            audio_data = trim_silence(audio_data)
            
        if self.audio_config.get('normalize_audio', True):
            audio_data = normalize_audio(audio_data)
            
        # Transcribe audio
        text = self.recognizer.transcribe(audio_data)
        return text, audio_data
    
    async def listen_and_transcribe_async(self) -> Tuple[str, Optional[np.ndarray]]:
        """
        Listen for speech and transcribe it asynchronously.
        
        Returns:
            Tuple of (transcribed_text, audio_data)
        """
        if not self.enabled or not self.input or not self.recognizer:
            self.logger.warning("Cannot listen and transcribe async: Audio system is disabled")
            return "", None
            
        loop = asyncio.get_running_loop()
        
        try:
            # Listen for command in a separate thread to avoid blocking
            audio_data = await loop.run_in_executor(None, self.input.listen_for_command)
            
            if audio_data is None or len(audio_data) == 0:
                return "", None
                
            # Process audio
            if self.audio_config.get('trim_silence', True):
                audio_data = await loop.run_in_executor(None, trim_silence, audio_data)
                
            if self.audio_config.get('normalize_audio', True):
                audio_data = await loop.run_in_executor(None, normalize_audio, audio_data)
                
            # Transcribe audio asynchronously
            text = await self.recognizer.transcribe_async(audio_data)
            return text, audio_data
        except Exception as e:
            self.logger.error(f"Error in listen_and_transcribe_async: {e}")
            return "", None
    
    def speak(self, text: str) -> bool:
        """
        Convert text to speech and play it.
        
        Args:
            text: Text to speak
            
        Returns:
            Success status
        """
        if not self.enabled or not self.output or not text:
            return False
            
        return self.output.speak(text)
    
    async def speak_async(self, text: str) -> bool:
        """
        Convert text to speech and play it asynchronously.
        
        Args:
            text: Text to speak
            
        Returns:
            Success status
        """
        if not self.enabled or not self.output or not text:
            return False
            
        return await self.output.speak_async(text)
    
    def transcribe_audio(self, audio_data: Union[str, np.ndarray]) -> str:
        """
        Transcribe audio data.
        
        Args:
            audio_data: Audio data (path or numpy array)
            
        Returns:
            Transcribed text
        """
        if not self.enabled or not self.recognizer:
            self.logger.warning("Cannot transcribe: Audio system is disabled")
            return ""
            
        return self.recognizer.transcribe(audio_data)
    
    async def transcribe_audio_async(self, audio_data: Union[str, np.ndarray]) -> str:
        """
        Transcribe audio data asynchronously.
        
        Args:
            audio_data: Audio data (path or numpy array)
            
        Returns:
            Transcribed text
        """
        if not self.enabled or not self.recognizer:
            self.logger.warning("Cannot transcribe async: Audio system is disabled")
            return ""
            
        return await self.recognizer.transcribe_async(audio_data)
    
    def get_audio_devices(self) -> list:
        """
        Get a list of available audio devices.
        
        Returns:
            List of audio device information
        """
        from .audio_utils import get_device_list
        return get_device_list()
    
    @asynccontextmanager
    async def _timed_operation(self, operation_name: str, timeout: float = 5.0):
        """Context manager for timing operations with timeout."""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.logger.debug(f"{operation_name} completed in {elapsed:.2f} seconds")
    
    def get_status(self) -> Dict:
        """
        Get detailed status of the audio system.
        
        Returns:
            Dictionary with status information
        """
        status = {
            "enabled": self.enabled
        }
        
        if not self.enabled:
            status["error"] = "Audio system is disabled"
            return status
        
        # Check if components are initialized
        if not self.input or not self.output or not self.recognizer:
            status["error"] = "Audio components not fully initialized"
            status["input_ready"] = self.input is not None
            status["output_ready"] = self.output is not None
            status["recognition_ready"] = self.recognizer is not None
            status["overall_ready"] = False
            return status
        
        # Get status from components
        try:
            status["input"] = self.input.get_status()
        except Exception as e:
            status["input_error"] = str(e)
            status["input"] = {"error": str(e)}
            
        try:
            status["output"] = self.output.get_status()
        except Exception as e:
            status["output_error"] = str(e)
            status["output"] = {"error": str(e)}
            
        try:
            status["recognizer"] = self.recognizer.get_status()
        except Exception as e:
            status["recognizer_error"] = str(e)
            status["recognizer"] = {"error": str(e)}
        
        # Add overall state
        status["input_ready"] = status.get("input", {}).get("microphone_available", False) and not status.get("input", {}).get("last_mic_error")
        status["output_ready"] = not status.get("output", {}).get("last_tts_error")
        status["recognition_ready"] = status.get("recognizer", {}).get("whisper_model_loaded", False)
        
        status["overall_ready"] = all([
            status["enabled"],
            status["input_ready"],
            status["output_ready"],
            status["recognition_ready"]
        ])
        
        # Add memory information if torch is available
        try:
            import torch
            if torch.cuda.is_available():
                status["gpu_memory_allocated_gb"] = round(torch.cuda.memory_allocated() / (1024 ** 3), 2)
                status["gpu_memory_reserved_gb"] = round(torch.cuda.memory_reserved() / (1024 ** 3), 2)
                status["gpu_memory_max_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
        except Exception:
            pass
        
        return status
    
    def stop(self):
        """Stop all audio components and clean up resources."""
        self.logger.info("Shutting down audio system")
        
        # Stop components in reverse order to ensure proper cleanup
        
        # Stop input first to stop audio capture
        if self.input:
            try:
                self.input.stop()
            except Exception as e:
                self.logger.error(f"Error stopping audio input: {e}")
        
        # Stop output next
        if self.output:
            try:
                self.output.stop()
            except Exception as e:
                self.logger.error(f"Error stopping audio output: {e}")
        
        # Stop recognizer last as it might be using GPU resources
        if self.recognizer:
            try:
                self.recognizer.stop()
            except Exception as e:
                self.logger.error(f"Error stopping speech recognizer: {e}")
        
        # Force garbage collection to free resources
        gc.collect()
        
        # Clear CUDA cache if torch is available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
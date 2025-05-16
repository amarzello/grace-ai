"""
Grace AI System - Speech Recognition Module

This module implements speech-to-text functionality using faster-whisper.
"""

import logging
import threading
import time
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List

# Try to import optional dependencies
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    WhisperModel = None  # For type checking

# Import from Grace modules
from grace.utils import MODELS_PATH
from .audio_utils import convert_audio_format


class SpeechRecognizer:
    """
    Speech recognition system using faster-whisper.
    
    Features:
    - Optimized speech-to-text with faster-whisper
    - Support for different model sizes
    - GPU acceleration when available
    - Proper error handling and timeout management
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the speech recognizer with the provided configuration.
        
        Args:
            config: Speech recognition configuration
        """
        self.logger = logging.getLogger('grace.audio.speech')
        self.whisper_config = config.get('whisper', {})
        
        # Initialize components
        self.whisper_model = None
        
        # Track failures for better error reporting
        self.last_stt_error = None
        
        # Create models directory if it doesn't exist
        MODELS_PATH.mkdir(parents=True, exist_ok=True)
        
        # Initialize the speech recognition model
        self._init_whisper()
    
    def _init_whisper(self):
        """Initialize Whisper speech-to-text model with proper error handling."""
        if not WHISPER_AVAILABLE:
            self.logger.error("faster-whisper not available. Install with: pip install faster-whisper")
            return
            
        try:
            self.logger.info("Loading Whisper model...")
            
            # Check for model existence before loading
            model_size = self.whisper_config.get('model_size', 'large-v2')
            model_path = self.whisper_config.get('model_path')
            device = self.whisper_config.get('device', 'cuda')
            
            # If model_path is specified, use that, otherwise use model_size
            if model_path and Path(model_path).exists():
                self.logger.info(f"Loading Whisper model from specific path: {model_path}")
                self.whisper_model = WhisperModel(
                    model_path,
                    device=device,
                    compute_type=self.whisper_config.get('compute_type', 'float16'),
                    download_root=str(MODELS_PATH)
                )
            else:
                # Try to determine if CUDA is available and fall back if needed
                if device == "cuda":
                    try:
                        import torch
                        if not torch.cuda.is_available():
                            self.logger.warning("CUDA not available, falling back to CPU for Whisper")
                            device = "cpu"
                            self.whisper_config['device'] = "cpu" 
                            # If using CPU, use int8 for better performance
                            self.whisper_config['compute_type'] = "int8"
                    except ImportError:
                        self.logger.warning("PyTorch not available to check CUDA, using CPU for Whisper")
                        device = "cpu"
                        self.whisper_config['device'] = "cpu"
                        self.whisper_config['compute_type'] = "int8"
                
                self.logger.info(f"Loading Whisper model: {model_size} on {device}")
                self.whisper_model = WhisperModel(
                    model_size,
                    device=device,
                    compute_type=self.whisper_config.get('compute_type', 'float16'),
                    download_root=str(MODELS_PATH)
                )
            
            self.logger.info("Whisper model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {e}")
            self.last_stt_error = str(e)
    
    def transcribe(self, audio_data: Union[str, np.ndarray]) -> str:
        """
        Transcribe audio using faster-whisper.
        
        Args:
            audio_data: Audio data (path or numpy array)
            
        Returns:
            Transcribed text
        """
        if not self.whisper_model:
            self.logger.error("Whisper model not loaded")
            self.last_stt_error = "Whisper model not loaded"
            return ""
            
        try:
            # Handle different input types
            if isinstance(audio_data, str):
                # Audio file path
                audio_path = audio_data
                if not Path(audio_path).exists():
                    self.logger.error(f"Audio file not found: {audio_path}")
                    self.last_stt_error = f"Audio file not found: {audio_path}"
                    return ""
            elif isinstance(audio_data, np.ndarray):
                # Numpy array
                # Normalize audio if it's in int16 format
                if audio_data.dtype == np.int16:
                    audio_data = convert_audio_format(audio_data, 'int16', 'float32')
                audio_path = audio_data
            else:
                self.logger.error(f"Unsupported audio data type: {type(audio_data)}")
                self.last_stt_error = f"Unsupported audio data type: {type(audio_data)}"
                return ""
            
            # Add visual indicator that transcription is happening
            import sys
            try:
                if sys.stdout.isatty():  # Check if running in a terminal
                    sys.stdout.write("\rTranscribing...")
                    sys.stdout.flush()
            except Exception:
                pass
                
            # Transcribe with a timeout implemented via a separate thread
            transcription_completed = threading.Event()
            transcription_result = [None]
            
            def transcribe_thread():
                try:
                    seg_gen, info = self.whisper_model.transcribe(
                        audio_path,
                        beam_size=self.whisper_config.get('beam_size', 5),
                        language=self.whisper_config.get('language', 'en'),
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500)
                    )
                    
                    # Join all segments
                    segments = list(seg_gen)
                    text = " ".join([segment.text for segment in segments])
                    transcription_result[0] = text.strip()
                except Exception as e:
                    self.logger.error(f"Transcription error in thread: {e}")
                    self.last_stt_error = str(e)
                finally:
                    transcription_completed.set()
            
            # Start transcription in a separate thread with a timeout
            thread = threading.Thread(target=transcribe_thread)
            thread.daemon = True
            thread.start()
            
            # Wait for completion with timeout
            timeout = self.whisper_config.get('timeout', 30)  # Default 30-second timeout
            transcription_completed.wait(timeout)
            
            # Clear the indicator
            try:
                if sys.stdout.isatty():
                    sys.stdout.write("\r" + " " * 20 + "\r")
                    sys.stdout.flush()
            except Exception:
                pass
            
            if not transcription_completed.is_set():
                self.logger.error(f"Transcription timed out after {timeout} seconds")
                self.last_stt_error = f"Transcription timed out after {timeout} seconds"
                return ""
                
            # Return the result
            return transcription_result[0] or ""
            
        except Exception as e:
            self.logger.error(f"Transcription error: {e}")
            self.last_stt_error = str(e)
            
            # Clear any indicator
            try:
                if sys.stdout.isatty():
                    sys.stdout.write("\r" + " " * 20 + "\r")
                    sys.stdout.flush()
            except Exception:
                pass
                
            return ""
    
    async def transcribe_async(self, audio_data: Union[str, np.ndarray]) -> str:
        """
        Transcribe audio asynchronously.
        
        Args:
            audio_data: Audio data (path or numpy array)
            
        Returns:
            Transcribed text
        """
        # Run transcription in the thread pool
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.transcribe, audio_data)

    def get_status(self) -> Dict:
        """
        Get detailed status of the speech recognition system.
        
        Returns:
            Dictionary with status information
        """
        status = {
            "whisper_available": WHISPER_AVAILABLE,
            "whisper_model_loaded": self.whisper_model is not None,
            "last_stt_error": self.last_stt_error,
            "model_path": self.whisper_config.get('model_path'),
            "model_size": self.whisper_config.get('model_size'),
            "device": self.whisper_config.get('device'),
            "compute_type": self.whisper_config.get('compute_type')
        }
        
        return status
    
    def stop(self):
        """Clean up resources."""
        self.logger.info("Shutting down speech recognition system")
        # No explicit cleanup needed for WhisperModel

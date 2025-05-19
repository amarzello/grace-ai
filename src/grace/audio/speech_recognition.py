"""
Grace AI System - Speech Recognition Module

This module implements speech-to-text functionality using faster-whisper
with improved resource management, error handling, and GPU memory cleanup.
"""

import logging
import threading
import time
import gc
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List

# Try to import torch for GPU memory management
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
# Try to import optional dependencies
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    WhisperModel = None  # For type checking

# Import from Grace modules
from grace.utils.common import MODELS_PATH
from .audio_utils import convert_audio_format


class SpeechRecognizer:
    """
    Speech recognition system using faster-whisper with improved resource management.
    
    Features:
    - Optimized speech-to-text with faster-whisper
    - Support for different model sizes
    - GPU acceleration when available
    - Proper error handling and timeout management
    - Memory cleanup and resource management
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
        self.model_lock = threading.RLock()  # Thread safety for model access
        
        # Create models directory if it doesn't exist
        MODELS_PATH.mkdir(parents=True, exist_ok=True)
        
        # Initialize the speech recognition model
        self._init_whisper()
    
    def _init_whisper(self):
        """Initialize Whisper speech-to-text model with proper error handling and fallbacks."""
        if not WHISPER_AVAILABLE:
            self.logger.error("faster-whisper not available. Install with: pip install faster-whisper")
            return
            
        with self.model_lock:
            try:
                self.logger.info("Loading Whisper model...")
                
                # Check for model existence before loading
                model_size = self.whisper_config.get('model_size', 'large-v2')
                model_path = self.whisper_config.get('model_path')
                device = self.whisper_config.get('device', 'cuda')
                
                # Determine if CUDA is available and fall back if needed
                if device == "cuda":
                    try:
                        if TORCH_AVAILABLE and not torch.cuda.is_available():
                            self.logger.warning("CUDA not available, falling back to CPU for Whisper")
                            device = "cpu"
                            self.whisper_config['device'] = "cpu" 
                            # If using CPU, use int8 for better performance
                            self.whisper_config['compute_type'] = "int8"
                    except Exception:
                        self.logger.warning("Error checking CUDA availability, using CPU for Whisper")
                        device = "cpu"
                        self.whisper_config['device'] = "cpu"
                        self.whisper_config['compute_type'] = "int8"
                
                # Load model with proper error handling
                try:
                    # If model_path is specified, use that, otherwise use model_size
                    if model_path and Path(model_path).exists():
                        self.logger.info(f"Loading Whisper model from specific path: {model_path}")
                        self.whisper_model = WhisperModel(
                            model_path,
                            device=device,
                            compute_type=self.whisper_config.get('compute_type', 'float16'),
                            download_root=str(MODELS_PATH),
                            cpu_threads=self.whisper_config.get('cpu_threads', 4),
                            num_workers=self.whisper_config.get('num_workers', 1)
                        )
                    else:
                        self.logger.info(f"Loading Whisper model: {model_size} on {device}")
                        self.whisper_model = WhisperModel(
                            model_size,
                            device=device,
                            compute_type=self.whisper_config.get('compute_type', 'float16'),
                            download_root=str(MODELS_PATH),
                            cpu_threads=self.whisper_config.get('cpu_threads', 4),
                            num_workers=self.whisper_config.get('num_workers', 1)
                        )
                        
                    self.logger.info("Whisper model loaded successfully")
                    
                except RuntimeError as e:
                    # Handle out of memory errors
                    if "CUDA out of memory" in str(e):
                        self.logger.warning(f"CUDA out of memory error: {e}, trying a smaller model")
                        # Try a smaller model size as fallback
                        smaller_sizes = ["medium", "small", "base", "tiny"]
                        for smaller_size in smaller_sizes:
                            try:
                                self.logger.info(f"Attempting to load smaller model: {smaller_size}")
                                self.whisper_model = WhisperModel(
                                    smaller_size,
                                    device=device,
                                    compute_type="int8",  # Use int8 for smaller memory footprint
                                    download_root=str(MODELS_PATH),
                                    cpu_threads=self.whisper_config.get('cpu_threads', 4),
                                    num_workers=self.whisper_config.get('num_workers', 1)
                                )
                                self.logger.info(f"Successfully loaded {smaller_size} model as fallback")
                                break
                            except Exception as inner_e:
                                self.logger.warning(f"Failed to load {smaller_size} model: {inner_e}")
                        
                        # If all smaller models failed, try CPU
                        if self.whisper_model is None:
                            self.logger.warning("All GPU models failed, falling back to CPU with small model")
                            try:
                                self.whisper_model = WhisperModel(
                                    "small",
                                    device="cpu",
                                    compute_type="int8",
                                    download_root=str(MODELS_PATH),
                                    cpu_threads=self.whisper_config.get('cpu_threads', 4)
                                )
                                self.logger.info("Successfully loaded small model on CPU as fallback")
                            except Exception as cpu_e:
                                self.logger.error(f"Failed to load fallback CPU model: {cpu_e}")
                    else:
                        # Other runtime errors
                        self.logger.error(f"Runtime error loading Whisper model: {e}")
                        raise
                        
            except Exception as e:
                self.logger.error(f"Failed to load Whisper model: {e}")
                self.last_stt_error = str(e)
    
    def transcribe(self, audio_data: Union[str, np.ndarray]) -> str:
        """
        Transcribe audio using faster-whisper with proper resource management.
        
        Args:
            audio_data: Audio data (path or numpy array)
            
        Returns:
            Transcribed text
        """
        if not self.whisper_model:
            self.logger.error("Whisper model not loaded")
            self.last_stt_error = "Whisper model not loaded"
            return ""
            
        with self.model_lock:
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
                    
                # Transcribe with a timeout implemented via a thread with proper cleanup
                transcription_completed = threading.Event()
                transcription_result = [None]
                transcription_error = [None]
                
                def transcribe_thread():
                    try:
                        # Use VAD filtering for better results
                        seg_gen, info = self.whisper_model.transcribe(
                            audio_path,
                            beam_size=self.whisper_config.get('beam_size', 5),
                            language=self.whisper_config.get('language', 'en'),
                            vad_filter=True,
                            vad_parameters=dict(min_silence_duration_ms=500)
                        )
                        
                        # Force evaluation to capture errors early
                        segments = list(seg_gen)
                        text = " ".join([segment.text for segment in segments])
                        transcription_result[0] = text.strip()
                    except Exception as e:
                        self.logger.error(f"Transcription error in thread: {e}")
                        self.last_stt_error = str(e)
                        transcription_error[0] = e
                    finally:
                        transcription_completed.set()
                
                # Start transcription in a separate thread with a timeout
                thread = threading.Thread(target=transcribe_thread)
                thread.daemon = True
                thread.start()
                
                # Wait for completion with timeout
                timeout = self.whisper_config.get('timeout', 30)  # Default 30-second timeout
                completed = transcription_completed.wait(timeout)
                
                # Clear the indicator
                try:
                    if sys.stdout.isatty():
                        sys.stdout.write("\r" + " " * 20 + "\r")
                        sys.stdout.flush()
                except Exception:
                    pass
                
                if not completed:
                    self.logger.error(f"Transcription timed out after {timeout} seconds")
                    self.last_stt_error = f"Transcription timed out after {timeout} seconds"
                    return ""
                
                # Handle transcription errors
                if transcription_error[0] is not None:
                    # If it's an OOM error, try to free memory and suggest restart
                    error_str = str(transcription_error[0])
                    if "out of memory" in error_str.lower():
                        self.logger.error("CUDA out of memory error during transcription")
                        # Force garbage collection and CUDA cache clearing
                        gc.collect()
                        if TORCH_AVAILABLE and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        return "Speech recognition failed due to insufficient GPU memory. Please try again."
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
        loop = asyncio.get_running_loop()
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
        
        # Add GPU memory info if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # GB
                status["gpu_memory_allocated_gb"] = round(allocated, 2)
                status["gpu_memory_reserved_gb"] = round(reserved, 2)
            except Exception:
                pass
        
        return status
    
    def stop(self):
        """Clean up resources."""
        self.logger.info("Shutting down speech recognition system")
        
        with self.model_lock:
            # Properly clean up model to free GPU memory
            if self.whisper_model:
                try:
                    del self.whisper_model
                    self.whisper_model = None
                    
                    # Force garbage collection
                    gc.collect()
                    
                    # Clear CUDA cache if available
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                    self.logger.info("Speech recognition resources freed successfully")
                except Exception as e:
                    self.logger.error(f"Error cleaning up whisper model: {e}")
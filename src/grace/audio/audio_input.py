"""
Grace AI System - Audio Input Module

This module handles microphone input and recording functionality with improved
resource management and Voice Activity Detection (VAD).
"""

import logging
import time
import queue
import threading
import socket
import os
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List
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

# Import from Grace modules
from grace.utils.common import MODELS_PATH
from .audio_utils import get_device_list


class AudioInput:
    """
    Audio input system with voice activity detection and recording capabilities.
    
    Features:
    - Direct microphone input with configurable parameters
    - Voice Activity Detection (VAD) for better speech recognition
    - Thread-safe audio recording and processing
    - Proper resource management and cleanup
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the audio input system with the provided configuration.
        
        Args:
            config: Audio input configuration
        """
        self.logger = logging.getLogger('grace.audio.input')
        self.audio_config = config.get('audio', {})
        
        # Initialize components
        self.vad = None
        self.audio_stream = None
        self.audio_queue = queue.Queue(maxsize=100)  # Bounded queue to prevent memory issues
        self.recording = False
        self.recording_lock = threading.RLock()  # Use RLock to prevent deadlocks
        
        # Track failures for better error reporting
        self.last_mic_error = None
        self.last_startup_time = 0
        
        # Keep track of stream properties
        self.stream_sample_rate = self.audio_config.get('sample_rate', 16000)
        self.stream_channels = self.audio_config.get('channels', 1)
        
        # Register cleanup for process exit
        import atexit
        atexit.register(self.stop)
        
        # Initialize VAD if available
        self._init_vad()
    
    def _init_vad(self):
        """Initialize Voice Activity Detection with proper error handling."""
        if not WEBRTCVAD_AVAILABLE:
            self.logger.warning("webrtcvad not available. Install with: pip install webrtcvad")
            return
            
        try:
            # Create VAD with configurable aggressiveness (0-3)
            aggressiveness = self.audio_config.get('vad_aggressiveness', 3)
            self.vad = webrtcvad.Vad(aggressiveness)
            self.logger.info(f"VAD initialized with aggressiveness {aggressiveness}")
        except Exception as e:
            self.logger.error(f"Failed to initialize VAD: {e}")
    
    def start_listening(self) -> bool:
        """
        Start listening for audio input with Voice Activity Detection and improved resource management.
        
        Returns:
            Success status
        """
        if not SOUNDDEVICE_AVAILABLE:
            self.logger.error("sounddevice not available. Install with: pip install sounddevice")
            self.last_mic_error = "sounddevice module not available"
            return False
            
        if not self.audio_config.get('use_microphone', True):
            self.logger.info("Microphone input disabled in config")
            return False
            
        # Use lock to prevent concurrent start/stop operations
        with self.recording_lock:
            if self.recording:
                self.logger.debug("Already listening")
                return True
            
            # Add cooldown to prevent rapid start/stop cycles
            current_time = time.time()
            if current_time - self.last_startup_time < 1.0:  # 1 second cooldown
                self.logger.debug("Start listening request too soon after last startup")
                # Still return True if we're already recording
                return self.recording
                
            try:
                # Clear any existing audio in the queue
                while not self.audio_queue.empty():
                    try:
                        self.audio_queue.get_nowait()
                    except Exception:
                        break
                    
                # Set up audio parameters
                sample_rate = self.audio_config.get('sample_rate', 16000)
                channels = self.audio_config.get('channels', 1)
                self.stream_sample_rate = sample_rate
                self.stream_channels = channels
                
                # Create callback with proper VAD integration
                def audio_callback(indata, frames, time_info, status):
                    """Callback for sounddevice stream to process incoming audio."""
                    if status:
                        self.logger.warning(f"Audio status: {status}")
                    
                    # Convert to the format expected by VAD (16-bit PCM)
                    audio_data = (indata * 32767).astype(np.int16)
                    
                    # If we have VAD, check if this is speech
                    if self.vad and WEBRTCVAD_AVAILABLE:
                        # VAD works on specific frame sizes (10, 20, or 30ms)
                        frame_duration_ms = 30
                        frame_size = int(sample_rate * frame_duration_ms / 1000) * 2  # 2 bytes per sample
                        
                        # Validate frame size for WebRTC VAD
                        valid_frame_sizes = [320, 480, 640, 960, 1280, 1920]  # Valid sizes for different sample rates
                        if frame_size not in valid_frame_sizes:
                            # Adjust to nearest valid size
                            frame_size = min(valid_frame_sizes, key=lambda x: abs(x - frame_size))
                        
                        # Process in VAD-compatible chunks
                        for i in range(0, len(audio_data), frame_size):
                            if i + frame_size <= len(audio_data):
                                chunk = audio_data[i:i+frame_size].copy()
                                frame = chunk.tobytes()
                                try:
                                    if self.vad.is_speech(frame, sample_rate):
                                        # Only add if it's speech
                                        try:
                                            self.audio_queue.put_nowait(chunk)
                                        except queue.Full:
                                            # Remove oldest item if queue is full
                                            try:
                                                self.audio_queue.get_nowait()
                                                self.audio_queue.put_nowait(chunk)
                                            except Exception:
                                                pass
                                except Exception as e:
                                    self.logger.debug(f"VAD error: {e}")
                            # Handle the remaining incomplete chunk if any
                            elif i < len(audio_data):
                                # Pad the remaining data to frame_size
                                padding = np.zeros(frame_size - (len(audio_data) - i), dtype=audio_data.dtype)
                                padded_frame = np.concatenate([audio_data[i:], padding])
                                frame = padded_frame.tobytes()
                                try:
                                    if self.vad.is_speech(frame, sample_rate):
                                        # Only add the actual data, not the padding
                                        try:
                                            self.audio_queue.put_nowait(audio_data[i:].copy())
                                        except queue.Full:
                                            # Remove oldest item if queue is full
                                            try:
                                                self.audio_queue.get_nowait()
                                                self.audio_queue.put_nowait(audio_data[i:].copy())
                                            except Exception:
                                                pass
                                except Exception as e:
                                    self.logger.debug(f"VAD error on last chunk: {e}")
                    else:
                        # Without VAD, add all audio
                        try:
                            self.audio_queue.put_nowait(audio_data.copy())
                        except queue.Full:
                            # Remove oldest item if queue is full
                            try:
                                self.audio_queue.get_nowait()
                                self.audio_queue.put_nowait(audio_data.copy())
                            except Exception:
                                pass
                
                # Clean up any existing stream first
                self._cleanup_stream()
                
                # Check if we can get a device list before starting the stream
                try:
                    devices = get_device_list()
                    self.logger.debug(f"Available audio devices: {len(devices)}")
                    
                    # Try to find a working input device
                    input_device = self.audio_config.get('input_device')
                    
                    # If no specific device is configured, try to find a default input
                    if input_device is None:
                        try:
                            default_device = sd.query_devices(kind='input')
                            if default_device:
                                input_device = default_device['name']
                                self.logger.info(f"Using default input device: {input_device}")
                        except Exception as e:
                            self.logger.warning(f"Error finding default input device: {e}")
                    
                    # Start the audio stream with the selected device and blocksize
                    try:
                        # Calculate appropriate blocksize for VAD
                        frame_duration_ms = 30  # 30ms frames for VAD
                        blocksize = int(sample_rate * frame_duration_ms / 1000)
                        
                        self.audio_stream = sd.InputStream(
                            device=input_device,
                            samplerate=sample_rate,
                            channels=channels,
                            dtype='float32',
                            callback=audio_callback,
                            blocksize=blocksize,
                            latency='low'  # Lower latency for better responsiveness
                        )
                    except Exception as e:
                        self.logger.warning(f"Error creating audio stream with device {input_device}: {e}")
                        # Fall back to default device
                        self.audio_stream = sd.InputStream(
                            samplerate=sample_rate,
                            channels=channels,
                            dtype='float32',
                            callback=audio_callback,
                            blocksize=blocksize,
                            latency='low'
                        )
                    
                except Exception as e:
                    self.logger.warning(f"Error querying audio devices: {e}")
                    # Fall back to default device with minimal configuration
                    frame_duration_ms = 30  # 30ms frames for VAD
                    blocksize = int(sample_rate * frame_duration_ms / 1000)
                    
                    self.audio_stream = sd.InputStream(
                        samplerate=sample_rate,
                        channels=channels,
                        dtype='float32',
                        callback=audio_callback,
                        blocksize=blocksize
                    )
                
                # Start the audio stream with proper error handling
                self.audio_stream.start()
                self.recording = True
                self.last_startup_time = time.time()
                    
                self.logger.info("Started listening for audio input")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to start audio input: {e}")
                self.last_mic_error = str(e)
                
                # Clean up any partially initialized stream
                self._cleanup_stream()
                
                self.recording = False
                return False
    
    def _cleanup_stream(self):
        """Clean up audio stream with proper resource management."""
        if self.audio_stream:
            try:
                if self.audio_stream.active:
                    self.audio_stream.stop()
                self.audio_stream.close()
            except Exception as e:
                self.logger.warning(f"Error closing audio stream: {e}")
            finally:
                self.audio_stream = None
    
    def stop_listening(self):
        """Stop listening for audio input with proper resource cleanup."""
        with self.recording_lock:
            self.recording = False
            self._cleanup_stream()
            
            # Clear the audio queue
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except Exception:
                    break
                    
            self.logger.info("Stopped listening for audio input")
    
    def listen_for_command(self, timeout: float = 5.0, silence_timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Listen for a voice command until silence is detected.
        
        Args:
            timeout: Maximum listening time
            silence_timeout: Silence duration to end recording
            
        Returns:
            Recorded audio as numpy array or None if no speech detected
        """
        if not self.recording:
            success = self.start_listening()
            if not success:
                return None
        
        # Use a list to collect audio chunks
        audio_chunks = []
        last_audio_time = time.time()
        start_time = time.time()
        
        # Display a listening indicator if in interactive mode
        is_interactive = False
        try:
            is_interactive = sys.stdout.isatty()  # Check if running in a terminal
            if is_interactive:
                sys.stdout.write("\rListening...")
                sys.stdout.flush()
        except Exception:
            pass
        
        try:
            # Listen until timeout or silence
            while time.time() - start_time < timeout:
                try:
                    # Get audio chunk with timeout
                    chunk = self.audio_queue.get(timeout=0.1)
                    audio_chunks.append(chunk)
                    last_audio_time = time.time()
                    
                    # Update the indicator
                    if is_interactive:
                        try:
                            sys.stdout.write("\rListening" + "." * (len(audio_chunks) % 4 + 1) + "   ")
                            sys.stdout.flush()
                        except Exception:
                            pass
                        
                except queue.Empty:
                    # Check if we've had enough silence to stop
                    if time.time() - last_audio_time > silence_timeout and audio_chunks:
                        break
                    
                    # Update the waiting indicator
                    if is_interactive:
                        try:
                            sys.stdout.write("\rListening" + "." * (int(time.time() * 2) % 4) + "   ")
                            sys.stdout.flush()
                        except Exception:
                            pass
            
            # Clear the listening indicator
            if is_interactive:
                try:
                    sys.stdout.write("\r" + " " * 20 + "\r")
                    sys.stdout.flush()
                except Exception:
                    pass
            
            # If we collected audio, combine the chunks
            if audio_chunks:
                try:
                    audio_data = np.concatenate(audio_chunks)
                    return audio_data
                except Exception as e:
                    self.logger.error(f"Error concatenating audio chunks: {e}")
                    return None
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error listening for command: {e}")
            
            # Clear the listening indicator
            if is_interactive:
                try:
                    sys.stdout.write("\r" + " " * 20 + "\r")
                    sys.stdout.flush()
                except Exception:
                    pass
                    
            return None
    
    def get_status(self) -> Dict:
        """
        Get detailed status of the audio input system.
        
        Returns:
            Dictionary with status information
        """
        status = {
            "microphone_available": SOUNDDEVICE_AVAILABLE,
            "vad_available": WEBRTCVAD_AVAILABLE,
            "recording": self.recording,
            "last_mic_error": self.last_mic_error,
            "sample_rate": self.stream_sample_rate,
            "channels": self.stream_channels
        }
        
        # Get audio devices if available
        if SOUNDDEVICE_AVAILABLE:
            try:
                status["audio_devices"] = get_device_list()
                status["default_input"] = self.audio_config.get('input_device', 'default')
            except Exception as e:
                status["audio_devices_error"] = str(e)
                
        return status
            
    def stop(self):
        """Stop audio input and clean up resources."""
        self.logger.info("Shutting down audio input system")
        self.stop_listening()
        
        # Force garbage collection to free resources
        try:
            import gc
            gc.collect()
        except Exception:
            pass
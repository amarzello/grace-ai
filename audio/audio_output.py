"""
Grace AI System - Audio Output Module

This module implements text-to-speech functionality for the Grace AI system.
"""

import logging
import os
import tempfile
import time
import threading
import subprocess
import shutil
import asyncio
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List

# Import from Grace modules
from grace.utils.paths import MODELS_PATH


class AudioOutput:
    """
    Audio output system for text-to-speech functionality.
    
    Features:
    - Text-to-speech using piper
    - Multiple fallback mechanisms
    - Support for various audio players
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the audio output system with the provided configuration.
        
        Args:
            config: Audio output configuration
        """
        self.logger = logging.getLogger('grace.audio.output')
        self.piper_config = config.get('piper', {})
        self.audio_config = config.get('audio', {})
        
        # Initialize components
        self.piper_process = None
        
        # Track failures for better error reporting
        self.last_tts_error = None
        self.model_found = False
        
        # Lock for thread safety
        self.piper_lock = threading.RLock()
        
        # Create models directory if it doesn't exist
        MODELS_PATH.mkdir(parents=True, exist_ok=True)
        
        # Verify if piper model exists
        if self.get_piper_model_path():
            self.model_found = True
    
    def get_piper_model_path(self):
        """
        Get the path to the Piper TTS model, checking multiple possible locations.
        
        Returns:
            Path to the Piper model file or None if not found
        """
        # Try the configured path first
        primary_path = self.piper_config.get('model_path', str(MODELS_PATH / 'cori-high.onnx'))
        primary_path = Path(primary_path)
        
        # Backward compatibility: Handle string paths
        if isinstance(primary_path, str):
            primary_path = Path(primary_path)
        
        if primary_path.exists():
            return str(primary_path)
            
        # Check alternative model names in the models directory
        model_candidates = [
            MODELS_PATH / 'cori-high.onnx',
            MODELS_PATH / 'en_US-cori-high.onnx',
            MODELS_PATH / 'en_US-amy-medium.onnx',
            MODELS_PATH / 'en_US-lessac-medium.onnx',
            Path('/usr/share/piper/voices/en_US-cori-high.onnx'),
            Path('/usr/share/piper/voices/en_US-amy-medium.onnx'),
            Path('/usr/share/piper/voices/en_US-lessac-medium.onnx'),
            Path('/usr/local/share/piper/voices/en_US-cori-high.onnx')
        ]
        
        for model_path in model_candidates:
            if model_path.exists():
                # Update the config with the found model path for future use
                self.piper_config['model_path'] = str(model_path)
                return str(model_path)
        
        # Search for any .onnx files in the models directory
        try:
            onnx_files = list(MODELS_PATH.glob("*.onnx"))
            if onnx_files:
                self.piper_config['model_path'] = str(onnx_files[0])
                return str(onnx_files[0])
        except Exception as e:
            self.logger.debug(f"Error searching for .onnx files: {e}")
                
        # No model found
        return None
    
    def start_piper(self):
        """
        Start piper TTS process with proper resource management.
        
        Returns:
            Success status
        """
        with self.piper_lock:
            # Check if already running
            if self.piper_process and self.piper_process.poll() is None:
                return True
                
            # Clean up any previous process
            self._cleanup_piper()
                
            # Get model path with fallback options
            model_path = self.get_piper_model_path()
            
            # Check if model exists
            if not model_path:
                self.logger.error(f"Piper model not found at {self.piper_config.get('model_path', 'unknown')} or in {MODELS_PATH}")
                self.last_tts_error = "Piper model not found"
                return False
                
            # Check if piper executable exists
            piper_executable = shutil.which("piper")
            if not piper_executable:
                self.logger.error("Piper executable not found in PATH")
                self.last_tts_error = "Piper executable not found"
                return False
            
            cmd = [
                piper_executable,
                "--model", model_path,
                "--output_raw",
                "--length_scale", "1",
                "--sentence_silence", "0.2"
            ]
            
            try:
                self.piper_process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=0,  # Unbuffered for binary mode
                    universal_newlines=False  # Binary mode
                )
                self.logger.info(f"Piper TTS started with model {model_path}")
                return True
            except FileNotFoundError:
                self.logger.error("Piper executable not found. Please install piper.")
                self.last_tts_error = "Piper executable not found"
                return False
            except Exception as e:
                self.logger.error(f"Failed to start Piper: {e}")
                self.last_tts_error = str(e)
                return False
            
    def _cleanup_piper(self):
        """Cleanup piper process to avoid resource leaks."""
        with self.piper_lock:
            if self.piper_process:
                try:
                    if self.piper_process.poll() is None:
                        self.piper_process.terminate()
                        try:
                            self.piper_process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            self.piper_process.kill()
                            self.piper_process.wait()
                except Exception as e:
                    self.logger.debug(f"Error cleaning up piper process: {e}")
                finally:
                    self.piper_process = None
                
    def speak_fallback(self, text: str) -> bool:
        """
        Fallback method to use system text-to-speech if piper fails.
        
        Args:
            text: Text to speak
            
        Returns:
            Success status
        """
        try:
            # Check if we have installed espeak or festival
            espeak_exists = shutil.which("espeak")
            festival_exists = shutil.which("festival")
            
            if not (espeak_exists or festival_exists):
                self.logger.warning("No TTS fallbacks found (espeak or festival)")
                self.last_tts_error = "No TTS fallbacks available"
                return False
            
            # Try to use system TTS via espeak or festival
            fallback_options = []
            
            if espeak_exists:
                fallback_options.append(["espeak"])
                
            if festival_exists:
                fallback_options.append(["bash", "-c", f"cat {{tmp_path}} | festival --tts"])
                
            for tts_cmd in fallback_options:
                try:
                    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
                        tmp_path = f.name
                        f.write(text)
                    
                    # Handle special case for festival shell commands
                    if "festival" in tts_cmd[0]:
                        cmd = ["bash", "-c", f"cat {tmp_path} | festival --tts"]
                    else:
                        cmd = tts_cmd + [tmp_path]
                    
                    result = subprocess.run(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=len(text) * 0.1 + 5  # Dynamic timeout based on text length
                    )
                    
                    # Clean up temp file
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
                    
                    if result.returncode == 0:
                        self.logger.info(f"Used fallback TTS: {tts_cmd[0]}")
                        return True
                    
                except (subprocess.SubprocessError, FileNotFoundError):
                    continue
                    
            # If all fallbacks failed
            self.logger.warning("All TTS fallbacks failed")
            self.last_tts_error = "All TTS fallbacks failed"
            return False
            
        except Exception as e:
            self.logger.error(f"TTS fallback error: {e}")
            self.last_tts_error = str(e)
            return False
        
    def speak(self, text: str) -> bool:
        """
        Convert text to speech using piper and play it.
        
        Args:
            text: Text to speak
            
        Returns:
            Success status
        """
        if not text:
            return False
            
        # Simple check for mute mode
        if self.audio_config.get('mute', False):
            self.logger.info("TTS is muted")
            return True
            
        # Start piper if not running
        if not self.start_piper():
            self.logger.warning("Failed to start piper, trying fallback TTS")
            return self.speak_fallback(text)
            
        try:
            # Make sure we can access piper_process safely
            with self.piper_lock:
                if not self.piper_process or self.piper_process.poll() is not None:
                    self.logger.warning("Piper process not available, restarting")
                    if not self.start_piper():
                        return self.speak_fallback(text)
                
                # Send text to piper with proper encoding and termination
                self.piper_process.stdin.write((text + "\n").encode('utf-8'))
                self.piper_process.stdin.flush()
            
            # Read raw audio output with timeout handling
            audio_data = bytearray()
            start_time = time.time()
            
            # Use a separate thread to read data to avoid blocking
            read_complete = threading.Event()
            
            def read_output():
                nonlocal audio_data
                try:
                    with self.piper_lock:
                        if not self.piper_process or self.piper_process.poll() is not None:
                            read_complete.set()
                            return
                            
                        while time.time() - start_time < 30:  # 30 second absolute timeout
                            chunk = self.piper_process.stdout.read(4096)
                            if not chunk:
                                break
                            audio_data.extend(chunk)
                            # Add reasonable size limit to avoid memory issues
                            if len(audio_data) > 10 * 1024 * 1024:  # 10MB limit
                                break
                except Exception as e:
                    self.logger.debug(f"Error reading audio data: {e}")
                finally:
                    read_complete.set()
            
            # Start reading thread
            read_thread = threading.Thread(target=read_output)
            read_thread.daemon = True
            read_thread.start()
            
            # Wait for thread with timeout
            read_complete.wait(timeout=10)  # 10 second timeout for reading
            
            if audio_data:
                # Use a temp file to avoid issues with large audio data
                with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as tmp:
                    tmp_path = tmp.name
                    tmp.write(audio_data)
                
                # Check for alternative audio players
                player_commands = []
                
                # Linux audio players
                if shutil.which("aplay"):
                    player_commands.append([
                        "aplay", 
                        "-r", str(self.piper_config.get('sample_rate', 22050)), 
                        "-f", "S16_LE", 
                        "-c", "1", 
                        tmp_path
                    ])
                
                if shutil.which("play"):
                    player_commands.append([
                        "play",
                        "-r", str(self.piper_config.get('sample_rate', 22050)),
                        "-b", "16",
                        "-c", "1",
                        "-e", "signed",
                        "-t", "raw",
                        tmp_path
                    ])
                    
                if shutil.which("paplay"):
                    player_commands.append([
                        "paplay",
                        "--rate", str(self.piper_config.get('sample_rate', 22050)),
                        "--format", "s16le",
                        "--channels", "1",
                        tmp_path
                    ])
                    
                # Try each player in order
                success = False
                for play_cmd in player_commands:
                    try:
                        self.logger.debug(f"Trying audio player: {play_cmd[0]}")
                        subprocess.run(play_cmd, check=True, 
                                    stdout=subprocess.DEVNULL, 
                                    stderr=subprocess.DEVNULL,
                                    timeout=len(text) * 0.1 + 3)  # Dynamic timeout based on text length
                        success = True
                        break
                    except Exception as e:
                        self.logger.debug(f"Error with player {play_cmd[0]}: {e}")
                        continue
                
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                    
                if success:
                    return True
                
                self.logger.warning("All audio players failed")
                
            else:
                self.logger.warning("No audio data received from piper")
            
            # If piper fails, try the fallback
            self.logger.warning("Piper TTS failed, trying fallback")
            return self.speak_fallback(text)
            
        except Exception as e:
            self.logger.error(f"TTS error: {e}")
            self.last_tts_error = str(e)
            # Try fallback if piper fails
            return self.speak_fallback(text)
    
    async def speak_async(self, text: str) -> bool:
        """
        Convert text to speech asynchronously.
        
        Args:
            text: Text to speak
            
        Returns:
            Success status
        """
        # Run TTS in the thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.speak, text)
        
    def get_status(self) -> Dict:
        """
        Get detailed status of the audio output system.
        
        Returns:
            Dictionary with status information
        """
        status = {
            "piper_process_running": False,
            "last_tts_error": self.last_tts_error,
            "mute": self.audio_config.get('mute', False),
            "model_found": self.model_found
        }
        
        # Check if piper process is running
        if self.piper_process:
            try:
                status["piper_process_running"] = self.piper_process.poll() is None
            except Exception:
                pass
            
        # Get piper model info if available
        piper_model = self.get_piper_model_path()
        if piper_model:
            status["piper_model"] = piper_model
            status["model_found"] = True
        else:
            status["piper_model"] = "Not found"
            status["model_found"] = False
            
        # Check available audio players
        audio_players = []
        if shutil.which("aplay"):
            audio_players.append("aplay")
        if shutil.which("play"):
            audio_players.append("play")
        if shutil.which("paplay"):
            audio_players.append("paplay")
        
        status["available_players"] = audio_players
        
        # Check fallback TTS options
        fallbacks = []
        if shutil.which("espeak"):
            fallbacks.append("espeak")
        if shutil.which("festival"):
            fallbacks.append("festival")
            
        status["available_fallbacks"] = fallbacks
            
        return status
            
    def stop(self):
        """Stop audio services and clean up resources."""
        self.logger.info("Shutting down audio output system")
        
        # Clean up piper
        self._cleanup_piper()

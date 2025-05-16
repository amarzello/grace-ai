#!/usr/bin/env python3
# tests/test_audio.py

import os
import sys
import asyncio
import tempfile
import numpy as np
import subprocess
from pathlib import Path

# Import base test class
from test_base import BaseTest, run_async_tests

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

class AudioTest(BaseTest):
    """Tests for the Grace audio system."""
    
    def __init__(self, verbose=False):
        super().__init__("audio", verbose)
        
        # Create a test config
        self.test_config = {
            'audio': {
                'use_microphone': False,  # Don't use actual microphone for tests
                'mute': True,             # Don't play actual audio
                'sample_rate': 16000,
                'channels': 1
            },
            'whisper': {
                'model_size': 'tiny',     # Use smallest model for fast tests
                'device': 'cpu',
                'compute_type': 'int8'
            },
            'piper': {
                'model_path': None        # Will use default or fallback
            },
            'debug': verbose
        }
        
        # Create a test audio file
        self.test_audio_file = self.create_test_audio_file()
    
    def create_test_audio_file(self):
        """Create a test audio file with a sine wave."""
        try:
            # Create a temporary file
            fd, path = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
            
            # Create a sine wave
            sample_rate = 16000
            duration = 1.0  # seconds
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            tone = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
            
            # Scale to 16-bit range
            audio = (tone * 32767).astype(np.int16)
            
            # Save to WAV file using subprocess to avoid scipy dependency
            try:
                import wave
                with wave.open(path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio.tobytes())
            except ImportError:
                # Fallback to creating a simpler raw file
                with open(path, 'wb') as f:
                    f.write(audio.tobytes())
            
            return path
        except Exception as e:
            self.logger.error(f"Failed to create test audio file: {e}")
            return None
    
    def test_audio_utils(self):
        """Test audio utility functions."""
        from grace.audio.audio_utils import normalize_audio, trim_silence, detect_silence
        
        # Create a simple audio sample with silence at beginning and end
        sample_rate = 16000
        duration = 1.0  # seconds
        silence_duration = 0.2  # seconds
        silence_samples = int(silence_duration * sample_rate)
        
        # Create a sine wave with silence at beginning and end
        t = np.linspace(0, duration - 2*silence_duration, int(sample_rate * (duration - 2*silence_duration)), False)
        tone = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
        
        # Add silence at beginning and end
        audio_with_silence = np.concatenate([
            np.zeros(silence_samples),
            tone,
            np.zeros(silence_samples)
        ])
        
        # Convert to float32 in range [-1, 1]
        audio_float = audio_with_silence.astype(np.float32) / 32767
        
        # Test silence detection
        is_silent = detect_silence(np.zeros(1000, dtype=np.float32))
        assert is_silent, "detect_silence should return True for silent audio"
        
        is_silent = detect_silence(audio_float)
        assert not is_silent, "detect_silence should return False for audio with content"
        
        # Test normalization
        normalized = normalize_audio(audio_float)
        assert np.max(np.abs(normalized)) <= 1.0, "Normalized audio should be within range [-1, 1]"
        assert np.max(np.abs(normalized)) >= 0.4, "Normalized audio should be amplified"
        
        # Test trim_silence 
        # Note: This test specifically targets the bug we identified in trim_silence
        trimmed = trim_silence(audio_float, threshold=0.01)
        
        # Check that the beginning and end silence were reduced
        assert len(trimmed) < len(audio_float), "Trimmed audio should be shorter than original"
        
        # If the trim_silence function works properly, the resulting audio should start and end
        # with non-silent samples
        assert np.abs(trimmed[0]) > 0.01, "Trimmed audio should start with non-silent sample"
        assert np.abs(trimmed[-1]) > 0.01, "Trimmed audio should end with non-silent sample"
    
    def test_audio_input_initialization(self):
        """Test audio input initialization without using real microphone."""
        from grace.audio.audio_input import AudioInput
        
        audio_input = AudioInput(self.test_config)
        assert audio_input is not None, "AudioInput should initialize"
        
        # Get status
        status = audio_input.get_status()
        assert isinstance(status, dict), "get_status should return a dictionary"
        
        # No resources to clean up specifically
    
    def test_audio_output_initialization(self):
        """Test audio output initialization."""
        from grace.audio.audio_output import AudioOutput
        
        audio_output = AudioOutput(self.test_config)
        assert audio_output is not None, "AudioOutput should initialize"
        
        # Get status
        status = audio_output.get_status()
        assert isinstance(status, dict), "get_status should return a dictionary"
        
        # Check for model_found field
        assert "model_found" in status, "Status should include model_found field"
        
        # Since we're not using a real model or requiring one to be present,
        # just check that the function returns either True or False
        assert status["model_found"] in [True, False], "model_found should be True or False"
        
        # Test cleanup
        audio_output.stop()
    
    def test_speech_recognition_initialization(self):
        """Test speech recognition initialization."""
        from grace.audio.speech_recognition import SpeechRecognizer
        
        # Skip this test if faster-whisper is not available
        try:
            import faster_whisper
        except ImportError:
            self.logger.warning("Skipping test_speech_recognition_initialization: faster-whisper not available")
            return
        
        speech_recognizer = SpeechRecognizer(self.test_config)
        assert speech_recognizer is not None, "SpeechRecognizer should initialize"
        
        # Get status
        status = speech_recognizer.get_status()
        assert isinstance(status, dict), "get_status should return a dictionary"
        
        # Check for whisper_available field
        assert "whisper_available" in status, "Status should include whisper_available field"
        
        # Test cleanup
        speech_recognizer.stop()
    
    def test_audio_system_initialization(self):
        """Test the entire audio system initialization."""
        from grace.audio.audio_system import AudioSystem
        
        audio_system = AudioSystem(self.test_config)
        assert audio_system is not None, "AudioSystem should initialize"
        
        # Check that the components are initialized
        assert audio_system.input is not None, "Input component should be initialized"
        assert audio_system.output is not None, "Output component should be initialized"
        assert audio_system.recognizer is not None, "Recognizer component should be initialized"
        
        # Get status
        status = audio_system.get_status()
        assert isinstance(status, dict), "get_status should return a dictionary"
        
        # Test cleanup
        audio_system.stop()
    
    def test_speak_with_fallback(self):
        """Test speak functionality with fallback mechanisms."""
        from grace.audio.audio_output import AudioOutput
        
        # Create config with mute enabled
        mute_config = self.test_config.copy()
        mute_config['audio'] = self.test_config['audio'].copy()
        mute_config['audio']['mute'] = True
        
        audio_output = AudioOutput(mute_config)
        
        # Test speak with mute enabled
        result = audio_output.speak("This is a test.")
        assert result is True, "speak should return True when mute is enabled"
        
        # Create config with mute disabled but using fallback
        unmute_config = self.test_config.copy()
        unmute_config['audio'] = self.test_config['audio'].copy()
        unmute_config['audio']['mute'] = False
        
        audio_output = AudioOutput(unmute_config)
        
        # Check if we have any TTS executable available
        has_tts = False
        for cmd in ["piper", "espeak", "festival"]:
            if any(os.access(os.path.join(path, cmd), os.X_OK) 
                  for path in os.environ["PATH"].split(os.pathsep)):
                has_tts = True
                break
        
        if has_tts:
            # Try speak with real TTS but capture output
            # This tests if the speak_fallback mechanism works
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = tempfile.TemporaryFile()
            sys.stderr = tempfile.TemporaryFile()
            
            try:
                result = audio_output.speak("This is a fallback test.")
                # Depending on environment, speak may succeed or fail with fallback
                assert result is not None, "speak should return a boolean result"
            finally:
                sys.stdout.close()
                sys.stderr.close()
                sys.stdout = old_stdout
                sys.stderr = old_stderr
        
        # Test cleanup
        audio_output.stop()
    
    async def test_async_speak(self):
        """Test asynchronous speak functionality."""
        from grace.audio.audio_output import AudioOutput
        
        audio_output = AudioOutput(self.test_config)
        
        # Test async speak
        result = await audio_output.speak_async("This is an async test.")
        assert result is not None, "speak_async should return a result"
        
        # Test cleanup
        audio_output.stop()
    
    def tearDown(self):
        """Clean up test resources."""
        if self.test_audio_file and os.path.exists(self.test_audio_file):
            try:
                os.unlink(self.test_audio_file)
            except Exception as e:
                self.logger.warning(f"Failed to delete test audio file: {e}")
    
    def run_all_tests(self):
        """Run all audio system tests."""
        try:
            self.test_audio_utils()
            self.test_audio_input_initialization()
            self.test_audio_output_initialization()
            self.test_speech_recognition_initialization()
            self.test_audio_system_initialization()
            self.test_speak_with_fallback()
            
            # Run async tests
            run_async_tests(self.test_async_speak())
            
            return self.print_results()
        finally:
            self.tearDown()

def run_tests(verbose=False):
    """Run audio system tests."""
    test = AudioTest(verbose=verbose)
    return test.run_all_tests()

if __name__ == "__main__":
    run_tests(verbose=True)
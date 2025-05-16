#!/usr/bin/env python3
"""
Test script for piper-tts
This script will download a voice model if needed, convert text to speech,
and save the output as a WAV file.
"""

import os
import sys
import tempfile
import subprocess
from pathlib import Path
import urllib.request
import json

def check_piper_installation():
    """Check if piper-tts is installed."""
    try:
        import piper
        print("✓ piper-tts is successfully installed!")
        return True
    except ImportError:
        print("✗ piper-tts is not installed correctly.")
        return False

def download_voice_model(model_name="cori-high", download_dir=None):
    """Download a voice model if not already present."""
    if download_dir is None:
        download_dir = os.path.expanduser("~/.local/share/piper_tts")
    
    os.makedirs(download_dir, exist_ok=True)
    
    base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
    model_file = f"{model_name}.onnx"
    config_file = f"{model_name}.onnx.json"
    model_path = os.path.join(download_dir, model_file)
    config_path = os.path.join(download_dir, config_file)
    
    # Download model and config if they don't exist
    if not os.path.exists(model_path):
        print(f"Downloading voice model: {model_name}")
        urllib.request.urlretrieve(f"{base_url}/{model_file}", model_path)
        print(f"Model saved to: {model_path}")
    
    if not os.path.exists(config_path):
        print(f"Downloading model config")
        urllib.request.urlretrieve(f"{base_url}/{config_file}", config_path)
        print(f"Config saved to: {config_path}")
        
    return model_path, config_path

def test_piper_tts():
    """Test piper-tts by synthesizing speech."""
    if not check_piper_installation():
        print("Please install piper-tts first.")
        return False
    
    # Download a voice model
    try:
        model_path, config_path = download_voice_model()
    except Exception as e:
        print(f"Failed to download voice model: {e}")
        return False
    
    # Test text to be synthesized
    test_text = "If I wanted you dead, you'd already be a dead fuckin cunt. . Gettin me twisted, so I insist on a roll plate.  We out of square, despite my words being wove straight.  Tryin to explain, the pain- the looms create.  We got to fix, the oh four six we fabricate.  Yeah.  My tongue's so loose that it's way too rackable, need to say my phrase in a way that's tactical.  Workin weekends- seekin a sabbatical.  Alone with my phone cuzz I'm so damn distractable.  Narrow the scope of what I utter like a side-cutter.  Awkwardly- cock-blockin hypocrisy- non-stop sloppily- adopting to monotony.  Run out the clock to top- the bureaucracy.  We playin chess-- but they winnin at monopoly.  --Alarm blares- despair somethin broke.  Wear and tear stopped the sley mid-stroke.  Take great care when you date to invoke, the need for a spare that aint there- its no joke.  I'm, rockin the mic to see if the wire is round I'm, mockin the loom as I groove to the sound I'm, stockin the room with skill by the pound I'm, not in the mood, for fuckin around."
#    test_text = "{{haha}} {ha} [haha] [[ha]] (ha) ((haha))  {{{{{hahaha}}}}}"
    output_file = os.path.join(tempfile.gettempdir(), "piper_test_output.wav")
    
    try:
        # Method 1: Using Python API
        try:
            from piper.pipeline import PipelineTTS
            
            # Create a TTS pipeline
            print("Testing with Python API...")
            tts = PipelineTTS(model_path, config_path)
            
            # Generate speech
            with open(output_file, "wb") as f:
                tts.synthesize(test_text, f)
            
            print(f"✓ Successfully synthesized speech using Python API")
            print(f"  Output saved to: {output_file}")
            
        except Exception as e:
            print(f"Python API test failed: {e}")
            print("Falling back to command-line method...")
            
            # Method 2: Using command-line interface
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
                tmp.write(test_text)
                tmp_path = tmp.name
            
            cmd = ["piper", "--model", model_path, "--output_file", output_file]
            with open(tmp_path, 'r') as f:
                process = subprocess.run(cmd, stdin=f, capture_output=True, text=True)
            
            os.unlink(tmp_path)  # Clean up temp file
            
            if process.returncode != 0:
                print(f"Error: {process.stderr}")
                return False
            
            print(f"✓ Successfully synthesized speech using command-line")
            print(f"  Output saved to: {output_file}")
        
        # Try to play the audio if possible
        try:
            if sys.platform == "darwin":  # macOS
                subprocess.run(["afplay", output_file])
            elif sys.platform == "linux":
                subprocess.run(["aplay", output_file])
            elif sys.platform == "win32":
                os.startfile(output_file)
        except Exception:
            print("Note: Could not automatically play the audio file.")
        
        return True
        
    except Exception as e:
        print(f"Failed to test piper-tts: {e}")
        return False

if __name__ == "__main__":
    print("=== Piper TTS Test ===")
    if test_piper_tts():
        print("\n✓ Piper TTS is working correctly!")
    else:
        print("\n✗ Piper TTS test failed.")

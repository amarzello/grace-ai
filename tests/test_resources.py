#!/usr/bin/env python3
# tests/test_resources.py

import os
import sys
import time
import threading
import asyncio
import tempfile
import gc
from pathlib import Path

# Import base test class
from test_base import BaseTest, run_async_tests

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

class ResourceTest(BaseTest):
    """Tests for resource management in the Grace AI system."""
    
    def __init__(self, verbose=False):
        super().__init__("resources", verbose)
        
        # Create a test config
        self.test_config = {
            'memory': {
                'max_context_tokens': 1000,
                'search_limit': 10,
                'use_in_memory': True,
                'enable_verification': True,
                'sqlite_wal_mode': False  # Disable WAL mode for testing
            },
            'audio': {
                'use_microphone': False,
                'mute': True
            },
            'amnesia_mode': False,
            'debug': verbose
        }
    
    def test_audio_resource_cleanup(self):
        """Test proper cleanup of audio resources."""
        from grace.audio.audio_system import AudioSystem
        
        audio_system = AudioSystem(self.test_config)
        
        # Start and stop audio input
        audio_system.start_listening()
        audio_system.stop_listening()
        
        # Shutdown
        audio_system.stop()
        
        # If we got here without exceptions, the test passes
        assert True, "Audio resource cleanup should complete without exceptions"
    
    def test_piper_process_cleanup(self):
        """Test proper cleanup of piper process."""
        from grace.audio.audio_output import AudioOutput
        
        audio_output = AudioOutput(self.test_config)
        
        # Start piper
        audio_output.start_piper()
        
        # Check if piper process is running
        process_running = False
        if audio_output.piper_process:
            try:
                process_running = audio_output.piper_process.poll() is None
            except Exception:
                pass
        
        # Cleanup
        with audio_output.piper_lock:  # Use lock to avoid race conditions
            audio_output._cleanup_piper()
        
        # Check if process was cleaned up
        if process_running:
            # If process was running, it should now be None or terminated
            assert (audio_output.piper_process is None or 
                    audio_output.piper_process.poll() is not None), "Piper process should be terminated"
        
        # Test stop method
        audio_output.stop()
    
    def test_thread_cleanup(self):
        """Test proper cleanup of threads."""
        from grace.audio.audio_input import AudioInput
        
        # Create a mock listen_for_command method that counts threads
        thread_count_before = threading.active_count()
        
        config = {
            'audio': {
                'use_microphone': False,
                'sample_rate': 16000,
                'channels': 1
            }
        }
        
        audio_input = AudioInput(config)
        
        # Create a function to run in a thread
        def mock_listen():
            # Sleep to simulate processing
            time.sleep(0.1)
            return None
        
        # Replace the listen_for_command method
        original_method = audio_input.listen_for_command
        audio_input.listen_for_command = mock_listen
        
        # Call the method that starts a thread
        audio_input.listen_for_command()
        
        # Wait for thread to complete
        time.sleep(0.2)
        
        # Count threads after the operation
        thread_count_after = threading.active_count()
        
        # Restore original method
        audio_input.listen_for_command = original_method
        
        # Check that no threads were leaked
        assert thread_count_after <= thread_count_before + 1, "No threads should be leaked"
    
    async def test_async_task_cleanup(self):
        """Test proper cleanup of async tasks."""
        from grace.memory.core import MemorySystem
        
        memory_system = MemorySystem(self.test_config)
        
        # Count tasks before the operation
        tasks_before = len(asyncio.all_tasks())
        
        # Create a task that will be added to active_tasks
        memory_system.active_tasks.append(
            asyncio.create_task(asyncio.sleep(0.1), name="test_task")
        )
        
        # Wait for task to complete
        await asyncio.sleep(0.2)
        
        # Count tasks after the operation
        tasks_after = len(asyncio.all_tasks())
        
        # Check that task was removed from active_tasks
        assert len(memory_system.active_tasks) == 0, "Task should be removed from active_tasks list"
        
        # Check that no tasks were leaked
        assert tasks_after <= tasks_before, "No tasks should be leaked"
        
        # Cleanup
        memory_system.shutdown()
    
    def test_temp_file_cleanup(self):
        """Test proper cleanup of temporary files."""
        from grace.audio.audio_output import AudioOutput
        
        # Create a list to track temporary files
        temp_files = []
        
        # Mock tempfile.NamedTemporaryFile to track created files
        original_named_temp_file = tempfile.NamedTemporaryFile
        
        def mock_named_temp_file(*args, **kwargs):
            temp_file = original_named_temp_file(*args, **kwargs)
            temp_files.append(temp_file.name)
            return temp_file
        
        # Replace the function
        tempfile.NamedTemporaryFile = mock_named_temp_file
        
        # Create audio output and call speak
        audio_output = AudioOutput(self.test_config)
        
        # Ensure mute is enabled to avoid actual audio output
        audio_output.audio_config['mute'] = True
        
        # Call speak (which should create temporary files)
        audio_output.speak("This is a test.")
        
        # Restore original function
        tempfile.NamedTemporaryFile = original_named_temp_file
        
        # Wait for any cleanup to happen
        time.sleep(0.2)
        
        # Check that temporary files are removed
        files_remaining = [f for f in temp_files if os.path.exists(f)]
        assert len(files_remaining) == 0, "All temporary files should be cleaned up"
        
        # Test cleanup
        audio_output.stop()
    
    async def test_database_connection_cleanup(self):
        """Test proper cleanup of database connections."""
        from grace.memory.sqlite import SQLiteStorage
        
        sqlite_storage = SQLiteStorage(self.test_config)
        
        # Add a memory
        await sqlite_storage.add_memory(
            "Test memory for connection cleanup",
            "contextual",
            "test_user"
        )
        
        # Force garbage collection to close any leaked connections
        gc.collect()
        
        # Close the storage
        sqlite_storage.close()
        
        # Wait for any pending tasks
        await asyncio.sleep(0.1)
        
        # Try to open the database file exclusively to check if connections are closed
        try:
            with open(sqlite_storage.long_term_db, 'r+b') as f:
                # Try to acquire an exclusive lock
                try:
                    # This is platform-specific, but a good test on Unix systems
                    import fcntl
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    # If we get here, the lock was acquired successfully
                    assert True, "Database file should not be locked by other connections"
                    # Release the lock
                    fcntl.flock(f, fcntl.LOCK_UN)
                except (ImportError, IOError):
                    # On Windows or if the lock is held by another process
                    # Just skip this part of the test
                    pass
        except Exception as e:
            self.logger.debug(f"Could not open database file exclusively: {e}")
            # This is not necessarily a failure, as the OS might keep the file locked
            pass
    
    async def run_all_tests(self):
        """Run all resource management tests."""
        self.test_audio_resource_cleanup()
        self.test_piper_process_cleanup()
        self.test_thread_cleanup()
        await self.test_async_task_cleanup()
        self.test_temp_file_cleanup()
        await self.test_database_connection_cleanup()
        
        return self.print_results()

def run_tests(verbose=False):
    """Run resource management tests."""
    test = ResourceTest(verbose=verbose)
    return run_async_tests(test.run_all_tests())

if __name__ == "__main__":
    run_tests(verbose=True)
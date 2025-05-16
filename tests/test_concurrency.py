#!/usr/bin/env python3
# tests/test_concurrency.py

import os
import sys
import time
import threading
import asyncio
from pathlib import Path
import tempfile

# Import base test class
from test_base import BaseTest, run_async_tests

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

class ConcurrencyTest(BaseTest):
    """Tests for concurrency issues in the Grace AI system."""
    
    def __init__(self, verbose=False):
        super().__init__("concurrency", verbose)
        
        # Create a test config
        self.test_config = {
            'memory': {
                'max_context_tokens': 1000,
                'search_limit': 10,
                'use_in_memory': True,
                'enable_verification': True,
                'sqlite_wal_mode': False  # Disable WAL mode for testing
            },
            'amnesia_mode': False,
            'debug': verbose
        }
    
    async def test_concurrent_memory_operations(self):
        """Test concurrent memory operations."""
        from grace.memory.core import MemorySystem
        from grace.memory.types import MemoryType
        
        memory_system = MemorySystem(self.test_config)
        
        # Create tasks for concurrent operations
        tasks = []
        for i in range(10):
            task = asyncio.create_task(
                memory_system.add_memory(
                    f"Concurrent memory {i}",
                    MemoryType.CONTEXTUAL,
                    "test_user"
                )
            )
            tasks.append(task)
        
        # Run concurrent operations
        memory_ids = await asyncio.gather(*tasks)
        
        # Check that all memories were added
        assert len(memory_ids) == 10, "All memories should be added"
        assert all(id is not None for id in memory_ids), "All memory IDs should be valid"
        
        # Test concurrent searching
        search_tasks = []
        for i in range(5):
            search_task = asyncio.create_task(
                memory_system.search_memories(f"memory {i}")
            )
            search_tasks.append(search_task)
        
        # Run concurrent searches
        search_results = await asyncio.gather(*search_tasks)
        
        # Check that all searches completed
        assert len(search_results) == 5, "All searches should complete"
        assert all(result is not None for result in search_results), "All search results should be valid"
        
        # Cleanup
        memory_system.shutdown()
    
    def test_thread_safety_audio_input(self):
        """Test thread safety in audio input."""
        from grace.audio.audio_input import AudioInput
        
        # Create a test config that doesn't use actual microphone
        config = {
            'audio': {
                'use_microphone': False,
                'sample_rate': 16000,
                'channels': 1
            }
        }
        
        audio_input = AudioInput(config)
        
        # Create threads that call start_listening concurrently
        def start_listening_thread():
            for _ in range(5):
                audio_input.start_listening()
                time.sleep(0.01)
                audio_input.stop_listening()
        
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=start_listening_thread)
            thread.daemon = True
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # If we got here without exceptions, it's a success
        # But let's also check the final state
        # Try to start listening one more time
        result = audio_input.start_listening()
        assert result is not None, "start_listening should return a result"
        
        # Stop listening
        audio_input.stop_listening()
    
    async def test_concurrent_file_access(self):
        """Test concurrent file access in critical memory manager."""
        from grace.memory.critical import CriticalMemoryManager
        
        # Create a temporary directory for the test
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set the memory path to the temporary directory
            old_memory_db_path = Path(os.environ.get("GRACE_MEMORY_DB_PATH", ""))
            os.environ["GRACE_MEMORY_DB_PATH"] = temp_dir
            
            try:
                # Create a critical memory manager
                critical_manager = CriticalMemoryManager(self.test_config)
                
                # Create tasks for concurrent operations
                tasks = []
                for i in range(10):
                    task = asyncio.create_task(
                        critical_manager.add_memory(
                            f"Concurrent critical memory {i}",
                            {"test": True},
                            i
                        )
                    )
                    tasks.append(task)
                
                # Run concurrent operations
                memory_ids = await asyncio.gather(*tasks)
                
                # Check that all memories were added
                assert len(memory_ids) == 10, "All memories should be added"
                assert all(id is not None for id in memory_ids), "All memory IDs should be valid"
                
                # Test concurrent searching
                search_tasks = []
                for i in range(5):
                    search_task = asyncio.create_task(
                        critical_manager.search_memories(f"memory {i}")
                    )
                    search_tasks.append(search_task)
                
                # Run concurrent searches
                search_results = await asyncio.gather(*search_tasks)
                
                # Check that all searches completed
                assert len(search_results) == 5, "All searches should complete"
                assert all(result is not None for result in search_results), "All search results should be valid"
                
                # Cleanup
                critical_manager.close()
            finally:
                # Restore original memory path
                if old_memory_db_path:
                    os.environ["GRACE_MEMORY_DB_PATH"] = str(old_memory_db_path)
                else:
                    del os.environ["GRACE_MEMORY_DB_PATH"]
                
                # Ensure any tasks are cleaned up
                tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
                for task in tasks:
                    if task.get_name().startswith('save_critical_memory'):
                        task.cancel()
                
                await asyncio.gather(*tasks, return_exceptions=True)
    
    async def test_async_resource_cleanup(self):
        """Test proper cleanup of async resources."""
        from grace.memory.core import MemorySystem
        
        memory_system = MemorySystem(self.test_config)
        
        # Create some memory tasks
        tasks = []
        for i in range(5):
            task = asyncio.create_task(
                memory_system._check_and_run_maintenance(),
                name=f"maintenance_task_{i}"
            )
            tasks.append(task)
        
        # Wait for tasks to complete
        await asyncio.gather(*tasks)
        
        # Check active tasks
        assert len(memory_system.active_tasks) == 0, "All tasks should be removed from active_tasks list"
        
        # Shutdown memory system
        memory_system.shutdown()
        
        # Wait for any pending tasks
        await asyncio.sleep(0.1)
        
        # Check for active tasks in the event loop
        active_tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        memory_tasks = [t for t in active_tasks if 'memory' in t.get_name()]
        
        assert len(memory_tasks) == 0, "No memory tasks should be left in the event loop"
    
    async def run_all_tests(self):
        """Run all concurrency tests."""
        await self.test_concurrent_memory_operations()
        self.test_thread_safety_audio_input()
        await self.test_concurrent_file_access()
        await self.test_async_resource_cleanup()
        
        return self.print_results()

def run_tests(verbose=False):
    """Run concurrency tests."""
    test = ConcurrencyTest(verbose=verbose)
    return run_async_tests(test.run_all_tests())

if __name__ == "__main__":
    run_tests(verbose=True)
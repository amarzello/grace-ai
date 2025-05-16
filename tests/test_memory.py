#!/usr/bin/env python3
# tests/test_memory.py

import os
import sys
import asyncio
import json
from pathlib import Path

# Import base test class
from test_base import BaseTest, run_async_tests

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from grace.memory.types import MemoryType
from grace.utils.common import ConversationEntry

class MemoryTest(BaseTest):
    """Tests for the Grace memory system."""
    
    def __init__(self, verbose=False):
        super().__init__("memory", verbose)
        
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
    
    async def test_memory_system_initialization(self):
        """Test that the memory system initializes correctly."""
        from grace.memory.core import MemorySystem
        
        # Test initialization
        memory_system = MemorySystem(self.test_config)
        assert memory_system is not None, "Memory system should initialize"
        assert memory_system.sqlite is not None, "SQLite storage should initialize"
        assert memory_system.vector is not None, "Vector storage should initialize"
        assert memory_system.critical is not None, "Critical memory should initialize"
        
        # Cleanup
        memory_system.shutdown()

    async def test_add_memory(self):
        """Test adding a memory."""
        from grace.memory.core import MemorySystem
        
        memory_system = MemorySystem(self.test_config)
        
        # Add a test memory
        test_content = "This is a test memory"
        memory_id = await memory_system.add_memory(
            test_content,
            MemoryType.CONTEXTUAL,
            "test_user"
        )
        
        assert memory_id is not None, "Memory ID should not be None"
        
        # Cleanup
        memory_system.shutdown()
    
    async def test_search_memories(self):
        """Test searching for memories."""
        from grace.memory.core import MemorySystem
        
        memory_system = MemorySystem(self.test_config)
        
        # Add a test memory
        test_content = "This is a unique test memory for search"
        await memory_system.add_memory(
            test_content,
            MemoryType.CONTEXTUAL,
            "test_user"
        )
        
        # Search for the memory
        results = await memory_system.search_memories("unique test memory")
        
        assert results is not None, "Search results should not be None"
        assert "contextual" in results, "Should find contextual memories"
        assert len(results["contextual"]) > 0, "Should find at least one memory"
        
        # Check that the content matches
        found = False
        for memory in results["contextual"]:
            if isinstance(memory, dict) and "content" in memory:
                if test_content in memory["content"]:
                    found = True
                    break
        
        assert found, f"Should find the test memory: {test_content}"
        
        # Cleanup
        memory_system.shutdown()
    
    async def test_log_conversation(self):
        """Test logging a conversation."""
        from grace.memory.core import MemorySystem
        
        memory_system = MemorySystem(self.test_config)
        
        # Create a test conversation entry
        entry = ConversationEntry(
            timestamp="2025-01-01T12:00:00",
            user_input="Test input",
            model_response="Test response"
        )
        
        # Log the conversation
        await memory_system.log_conversation(entry)
        
        # Search for the conversation
        results = await memory_system.search_memories("Test input")
        
        assert results is not None, "Search results should not be None"
        assert "conversations" in results, "Should find conversation memories"
        
        # Cleanup
        memory_system.shutdown()
    
    async def test_amnesia_mode(self):
        """Test that amnesia mode prevents storing memories."""
        from grace.memory.core import MemorySystem
        
        # Create config with amnesia mode enabled
        amnesia_config = self.test_config.copy()
        amnesia_config['amnesia_mode'] = True
        
        memory_system = MemorySystem(amnesia_config)
        
        # Add a test memory
        test_content = "This memory should not be stored"
        memory_id = await memory_system.add_memory(
            test_content,
            MemoryType.CONTEXTUAL,
            "test_user"
        )
        
        assert memory_id is None, "Memory ID should be None in amnesia mode"
        
        # Log a conversation
        entry = ConversationEntry(
            timestamp="2025-01-01T12:00:00",
            user_input="Test input in amnesia mode",
            model_response="Test response"
        )
        
        # Log the conversation
        await memory_system.log_conversation(entry)
        
        # Search for the memories
        results = await memory_system.search_memories("amnesia mode")
        
        assert results is not None, "Search results should not be None"
        assert "contextual" in results, "Should have contextual category"
        assert len(results["contextual"]) == 0, "Should not find any memories in amnesia mode"
        
        # Cleanup
        memory_system.shutdown()
    
    async def test_concurrent_memory_operations(self):
        """Test concurrent memory operations."""
        from grace.memory.core import MemorySystem
        
        memory_system = MemorySystem(self.test_config)
        
        # Create tasks for concurrent operations
        async def add_memories():
            tasks = []
            for i in range(5):
                task = asyncio.create_task(
                    memory_system.add_memory(
                        f"Concurrent memory {i}",
                        MemoryType.CONTEXTUAL,
                        "test_user"
                    )
                )
                tasks.append(task)
            return await asyncio.gather(*tasks)
        
        # Run concurrent operations
        memory_ids = await add_memories()
        
        assert len(memory_ids) == 5, "Should add 5 memories"
        assert all(id is not None for id in memory_ids), "All memory IDs should be valid"
        
        # Search for the memories
        results = await memory_system.search_memories("Concurrent memory")
        
        assert results is not None, "Search results should not be None"
        assert "contextual" in results, "Should have contextual category"
        assert len(results["contextual"]) > 0, "Should find the concurrent memories"
        
        # Cleanup
        memory_system.shutdown()
        
    async def test_memory_shutdown(self):
        """Test proper memory system shutdown."""
        from grace.memory.core import MemorySystem
        
        memory_system = MemorySystem(self.test_config)
        
        # Add a test memory
        await memory_system.add_memory(
            "Memory before shutdown",
            MemoryType.CONTEXTUAL,
            "test_user"
        )
        
        # Shutdown the memory system
        memory_system.shutdown()
        
        # Create a new memory system
        new_memory_system = MemorySystem(self.test_config)
        
        # Search for the memory
        results = await new_memory_system.search_memories("before shutdown")
        
        assert results is not None, "Search results should not be None"
        assert "contextual" in results, "Should have contextual category"
        assert len(results["contextual"]) > 0, "Should find the memory after shutdown and restart"
        
        # Cleanup
        new_memory_system.shutdown()
    
    async def run_all_tests(self):
        """Run all memory system tests."""
        await self.test_memory_system_initialization()
        await self.test_add_memory()
        await self.test_search_memories()
        await self.test_log_conversation()
        await self.test_amnesia_mode()
        await self.test_concurrent_memory_operations()
        await self.test_memory_shutdown()
        
        return self.print_results()

def run_tests(verbose=False):
    """Run memory system tests."""
    test = MemoryTest(verbose=verbose)
    return run_async_tests(test.run_all_tests())

if __name__ == "__main__":
    run_tests(verbose=True)
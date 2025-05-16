#!/usr/bin/env python3
# tests/test_integration.py

import os
import sys
import time
import asyncio
from pathlib import Path

# Import base test class
from test_base import BaseTest, run_async_tests

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

class IntegrationTest(BaseTest):
    """Integration tests for the Grace AI system."""
    
    def __init__(self, verbose=False):
        super().__init__("integration", verbose)
        
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
            'ovos': {
                'disable_ovos': True  # Disable OVOS for integration tests
            },
            'system': {
                'input_mode': 'text'
            },
            'amnesia_mode': False,
            'debug': verbose
        }
    
    async def test_system_initialization(self):
        """Test that the entire system initializes correctly."""
        # This test only imports the main module without a real model
        
        # Skip if llama-cpp-python is not available
        try:
            import llama_cpp
        except ImportError:
            self.logger.warning("Skipping test_system_initialization: llama_cpp not available")
            return
        
        from grace.utils.common import setup_logging

        # Initialize logging
        logger = setup_logging(debug=True)
        
        # Import the orchestrator
        from grace.orchestrator import SystemOrchestrator
        
        # Import memory system
        from grace.memory.core import MemorySystem
        memory_system = MemorySystem(self.test_config)
        
        # Import language model with a mock
        from grace.language_model import LlamaWrapper
        
        # Create a minimal config for the language model
        language_model_config = self.test_config.copy()
        language_model_config['llama'] = {
            'model_path': '/path/to/nonexistent/model.gguf',
            'n_ctx': 2048,
            'n_gpu_layers': 0,
            'temperature': 0.7
        }
        
        # Initialize the language model (will not actually load a model)
        language_model = LlamaWrapper(language_model_config)
        
        # Initialize the orchestrator
        orchestrator = SystemOrchestrator(
            self.test_config,
            memory_system,
            language_model,
            None,  # No audio system
            None   # No OVOS interface
        )
        
        # Start the orchestrator
        await orchestrator.start()
        
        # Check that the orchestrator is running
        assert orchestrator.running, "Orchestrator should be running"
        
        # Stop the orchestrator
        await orchestrator.stop()
        
        # Check that the orchestrator is stopped
        assert not orchestrator.running, "Orchestrator should be stopped"
        
        # Cleanup
        memory_system.shutdown()
    
    async def test_memory_and_language_model_integration(self):
        """Test integration between memory system and language model."""
        # Skip if llama-cpp-python is not available
        try:
            import llama_cpp
        except ImportError:
            self.logger.warning("Skipping test_memory_and_language_model_integration: llama_cpp not available")
            return
        
        from grace.memory.core import MemorySystem
        from grace.memory.types import MemoryType
        from grace.language_model import LlamaWrapper
        
        # Create memory system
        memory_system = MemorySystem(self.test_config)
        
        # Create a language model with response mocking
        class MockLlamaWrapper(LlamaWrapper):
            def __init__(self, config):
                # Skip actual model loading
                self.logger = self.logger = logging.getLogger('grace.mock_llama')
                self.config = config
                self.llama_config = config.get('llama', {})
                self.model = None
                self.json_grammar = None
                self.generation_count = 0
                self.total_tokens_generated = 0
                self.last_generation_time = 0
                self.stop_sequences = []
            
            def generate(self, prompt, max_tokens=None, temperature=None, use_grammar=True,
                        top_p=None, top_k=None, presence_penalty=None, min_p=None):
                # Check if the prompt contains memory context
                memory_found = "Contextual Memories" in prompt
                
                # Return a mock response
                if memory_found:
                    return """<think>
I see the user is asking about something related to our previous conversation about testing.
The memory system has provided contextual information that will help me answer.
</think>
{
  "response": "I remember we were discussing testing. Let me use that information to respond."
}"""
                else:
                    return """<think>
I don't have any specific memories about this topic.
</think>
{
  "response": "I don't have specific information about that topic in my memories."
}"""
        
        # Create a minimal config for the language model
        language_model_config = self.test_config.copy()
        language_model_config['llama'] = {
            'model_path': '/path/to/nonexistent/model.gguf',
            'n_ctx': 2048,
            'n_gpu_layers': 0,
            'temperature': 0.7
        }
        
        # Initialize the mock language model
        language_model = MockLlamaWrapper(language_model_config)
        
        # Add a test memory
        await memory_system.add_memory(
            "I'm testing the integration between the memory system and language model.",
            MemoryType.CONTEXTUAL,
            "test_user"
        )
        
        # Create a function to generate a response with memory context
        async def generate_response(query):
            # Search for relevant memories
            memories = await memory_system.search_memories(query)
            
            # Build a prompt with memory context
            prompt = f"USER: {query}\n\nMEMORY:\n"
            
            if memories.get("contextual"):
                prompt += "=== Contextual Memories ===\n"
                for mem in memories["contextual"][:3]:  # Limit to 3 memories
                    if isinstance(mem, dict):
                        content = mem.get('memory', mem.get('content', ''))
                        if content:
                            prompt += f"- {content}\n"
            
            # Generate a response
            response = language_model.generate(prompt)
            
            # Parse the response
            thinking, json_response = language_model.parse_response(response)
            
            return json_response.get("response", "")
        
        # Test with a query that should match the memory
        response1 = await generate_response("Tell me about testing")
        
        # Test with a query that should not match the memory
        response2 = await generate_response("Tell me about something unrelated")
        
        # Check that the responses are different
        assert "remember" in response1, "Response should reference memory for related query"
        assert "don't have specific information" in response2, "Response should indicate no memory for unrelated query"
        
        # Cleanup
        memory_system.shutdown()
    
    async def run_all_tests(self):
        """Run all integration tests."""
        # Import needed modules for test mocking
        import logging
        global logging
        
        await self.test_system_initialization()
        await self.test_memory_and_language_model_integration()
        
        return self.print_results()

def run_tests(verbose=False):
    """Run integration tests."""
    test = IntegrationTest(verbose=verbose)
    return run_async_tests(test.run_all_tests())

if __name__ == "__main__":
    run_tests(verbose=True)
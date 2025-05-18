#!/usr/bin/env python3
"""
Grace AI System - Main Entry Point

This script provides the main entry point for the Grace AI system.
"""

import os
import sys
import asyncio
import argparse
import signal
import logging
from pathlib import Path

# Set up directories for imports
script_dir = Path(__file__).resolve().parent
if script_dir not in sys.path:
    sys.path.insert(0, str(script_dir))

# Import Grace modules
from grace.utils.common import setup_logging, load_config
from grace.memory.core import MemorySystem
from grace.llm.language_model import LlamaWrapper
from grace.audio.audio_system import AudioSystem
from grace.ovos import OVOSInterface
from grace.orchestrator import SystemOrchestrator  # New central orchestrator


async def main():
    """Main entry point for the Grace AI system."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Grace - Locally Run Personal AI Assistant"
    )
    
    parser.add_argument("--config", type=str, help="Path to custom config file")
    parser.add_argument("--amnesia", action="store_true", help="Enable amnesia mode (no memory storage)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--test-memory", action="store_true", help="Test memory system functionality")
    parser.add_argument("--wipe-memory", action="store_true", help="Wipe all memory storage")
    parser.add_argument("--voice-mode", action="store_true", help="Start in voice interaction mode")
    parser.add_argument("--text-mode", action="store_true", help="Start in text interaction mode")
    parser.add_argument("--no-ovos", action="store_true", help="Disable OVOS integration")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio system")
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(debug=args.debug)
    logger.info("Starting Grace AI System")
    
    # Create configuration with overrides from command line
    config_overrides = {
        'debug': args.debug,
        'amnesia_mode': args.amnesia
    }
    
    if args.config:
        config_overrides['config'] = args.config
    
    if args.no_ovos:
        if 'ovos' not in config_overrides:
            config_overrides['ovos'] = {}
        config_overrides['ovos']['disable_ovos'] = True
    
    if args.no_audio:
        if 'audio' not in config_overrides:
            config_overrides['audio'] = {}
        config_overrides['audio']['disable_audio'] = True
    
    config = load_config(args.config, config_overrides)
    
    # Special option: wipe memory if requested
    if args.wipe_memory:
        if input("Are you sure you want to wipe all memory? (yes/no): ").lower() == 'yes':
            from shutil import rmtree
            from grace.utils.common import MEMORY_DB_PATH
            
            # Create backup before wiping
            import datetime
            backup_dir = MEMORY_DB_PATH.parent / f"memory_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                if MEMORY_DB_PATH.exists():
                    import shutil
                    shutil.copytree(MEMORY_DB_PATH, backup_dir)
                    logger.info(f"Created memory backup at {backup_dir}")
                    
                    # Remove memory directory
                    rmtree(MEMORY_DB_PATH)
                    MEMORY_DB_PATH.mkdir(parents=True, exist_ok=True)
                    logger.info("Memory storage has been wiped")
                    print("Memory has been wiped. A backup was created.")
                else:
                    logger.info("No memory directory found to wipe")
                    print("No memory directory found.")
            except Exception as e:
                logger.error(f"Error during memory wipe: {e}")
                print(f"Error during memory wipe: {e}")
            
            # Exit after wiping
            return
    
    # Initialize components
    memory_system = None
    language_model = None
    audio_system = None
    ovos_interface = None
    orchestrator = None
    
    try:
        # Initialize the memory system
        logger.info("Initializing memory system...")
        memory_system = MemorySystem(config)
        
        # Run memory system tests if requested
        if args.test_memory:
            await test_memory_system(memory_system)
            # Clean shutdown
            await shutdown_gracefully(memory_system, None, None, None, None)
            return
        
        # Initialize the language model
        logger.info("Initializing language model...")
        language_model = LlamaWrapper(config)
        
        # Initialize audio system if not disabled
        if not config.get('audio', {}).get('disable_audio', False):
            logger.info("Initializing audio system...")
            audio_system = AudioSystem(config)
        else:
            logger.info("Audio system disabled by configuration")
        
        # Initialize OVOS interface if not disabled
        if not config.get('ovos', {}).get('disable_ovos', False):
            logger.info("Initializing OVOS interface...")
            ovos_interface = OVOSInterface(config)
        else:
            logger.info("OVOS integration disabled by configuration")
        
        # Initialize the orchestrator
        logger.info("Initializing system orchestrator...")
        orchestrator = SystemOrchestrator(
            config=config,
            memory_system=memory_system,
            language_model=language_model,
            audio_system=audio_system,
            ovos_interface=ovos_interface
        )
        
        # Register signal handlers for graceful shutdown
        register_signal_handlers(memory_system, language_model, audio_system, ovos_interface, orchestrator)
        
        # Start the system
        logger.info("Starting Grace AI System")
        await orchestrator.start()
        
        # Determine interaction mode
        if args.voice_mode:
            interaction_mode = "voice"
        elif args.text_mode:
            interaction_mode = "text"
        else:
            interaction_mode = config.get("system", {}).get("input_mode", "text")
        
        # Start the appropriate interaction loop
        if interaction_mode == "voice" and audio_system:
            logger.info("Starting voice interaction mode")
            await orchestrator.voice_interaction_loop()
        else:
            logger.info("Starting text interaction mode")
            await orchestrator.text_interaction_loop()
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Error during Grace AI operation: {e}", exc_info=True)
    finally:
        # Clean shutdown
        await shutdown_gracefully(memory_system, language_model, audio_system, ovos_interface, orchestrator)
        logger.info("Grace AI System shutdown complete")


async def test_memory_system(memory_system):
    """Run tests on the memory system."""
    from grace.memory.types import MemoryType
    
    print("\n==== Testing Memory System ====\n")
    
    # Test adding a memory
    print("Testing memory addition...")
    memory_id = await memory_system.add_memory(
        "This is a test memory for Grace AI",
        MemoryType.CRITICAL,
        "test_user",
        {"test": True, "topic": "memory_test"}
    )
    print(f"Memory added with ID: {memory_id}")
    
    # Test searching for memories
    print("\nTesting memory search...")
    results = await memory_system.search_memories("test memory")
    
    print(f"Found {len(results.get('critical', []))} critical memories")
    for memory in results.get('critical', []):
        print(f"- [{memory.get('relevance', 0):.2f}] {memory.get('content')}")
    
    # Get memory statistics
    print("\nGetting memory statistics...")
    stats = await memory_system.get_memory_stats()
    
    print("Memory Statistics:")
    print(f"- Total Memories: {stats.get('total_memories', 0)}")
    print(f"- SQLite: {stats.get('sqlite', {}).get('long_term_count', 0)} long term memories")
    print(f"- Critical: {stats.get('critical', {}).get('total_memories', 0)} critical memories")
    
    print("\n==== Memory System Test Complete ====\n")


def register_signal_handlers(memory_system, language_model, audio_system, ovos_interface, orchestrator):
    """Register signal handlers for graceful shutdown."""
    loop = asyncio.get_running_loop()
    
    def signal_handler():
        """Handle termination signals."""
        asyncio.create_task(shutdown_gracefully(
            memory_system, language_model, audio_system, ovos_interface, orchestrator
        ))
    
    # Register for SIGINT (Ctrl+C) and SIGTERM
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)


async def shutdown_gracefully(memory_system, language_model, audio_system, ovos_interface, orchestrator):
    """Gracefully shut down all system components."""
    logger = logging.getLogger('grace')
    logger.info("Shutting down Grace AI System...")
    
    # First stop the orchestrator
    if orchestrator:
        logger.info("Stopping orchestrator...")
        await orchestrator.stop()
    
    # Shutdown components in reverse order of initialization
    if ovos_interface:
        logger.info("Shutting down OVOS interface...")
        ovos_interface.shutdown()
    
    if audio_system:
        logger.info("Shutting down audio system...")
        audio_system.stop()
    
    if language_model:
        logger.info("Shutting down language model...")
        if hasattr(language_model, 'shutdown'):
            language_model.shutdown()
    
    if memory_system:
        logger.info("Shutting down memory system...")
        memory_system.shutdown()
    
    logger.info("Shutdown complete")


if __name__ == "__main__":
    # Handle Windows asyncio issues
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(main())
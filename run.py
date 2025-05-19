#!/usr/bin/env python3
"""
Grace AI System - Main Entry Point

This script provides the main entry point for the Grace AI system with improved
resource management, error handling, and async patterns.
"""

import os
import sys
import asyncio
import argparse
import signal
import logging
import traceback
import gc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

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
from grace.utils.error_handling import GraceAIError, ResourceError


async def main():
    """Main entry point for the Grace AI system with improved error handling."""
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
    parser.add_argument("--profile-memory", action="store_true", help="Enable memory profiling")
    
    args = parser.parse_args()
    
    # Set up logging with proper error handling
    try:
        logger = setup_logging(debug=args.debug)
        logger.info("Starting Grace AI System")
    except Exception as e:
        print(f"ERROR: Failed to set up logging: {e}")
        return 1
    
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
    
    # Load configuration with proper error handling
    try:
        config = load_config(args.config, config_overrides)
    except Exception as e:
        logger.critical(f"Failed to load configuration: {e}")
        print(f"CRITICAL: Failed to load configuration: {e}")
        return 1
    
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
            return 0
    
    # Memory profiling if requested
    if args.profile_memory:
        try:
            import tracemalloc
            tracemalloc.start()
            logger.info("Memory profiling enabled")
        except ImportError:
            logger.warning("tracemalloc module not available, memory profiling disabled")
    
    # Initialize components with proper resource management
    memory_system = None
    language_model = None
    audio_system = None
    ovos_interface = None
    orchestrator = None
    
    try:
        # Initialize the memory system
        logger.info("Initializing memory system...")
        try:
            memory_system = MemorySystem(config)
        except Exception as e:
            logger.critical(f"Failed to initialize memory system: {e}")
            print(f"CRITICAL: Failed to initialize memory system: {e}")
            return 1
        
        # Run memory system tests if requested
        if args.test_memory:
            test_result = await test_memory_system(memory_system)
            # Clean shutdown
            await shutdown_gracefully(memory_system, None, None, None, None)
            return 0 if test_result else 1
        
        # Initialize the language model with fallback options
        logger.info("Initializing language model...")
        try:
            language_model = LlamaWrapper(config)
            if not language_model.model:
                raise ResourceError("Failed to load language model")
        except GraceAIError as e:
            logger.critical(f"Failed to initialize language model: {e}")
            print(f"CRITICAL: Failed to initialize language model: {e}")
            await shutdown_gracefully(memory_system, None, None, None, None)
            return 1
        except Exception as e:
            logger.critical(f"Unexpected error initializing language model: {e}")
            print(f"CRITICAL: Unexpected error initializing language model: {e}")
            await shutdown_gracefully(memory_system, None, None, None, None)
            return 1
        
        # Initialize audio system if not disabled
        if not config.get('audio', {}).get('disable_audio', False):
            logger.info("Initializing audio system...")
            try:
                audio_system = AudioSystem(config)
            except Exception as e:
                logger.error(f"Failed to initialize audio system, continuing without audio: {e}")
                print(f"WARNING: Audio system initialization failed: {e}")
        else:
            logger.info("Audio system disabled by configuration")
        
        # Initialize OVOS interface if not disabled
        if not config.get('ovos', {}).get('disable_ovos', False):
            logger.info("Initializing OVOS interface...")
            try:
                ovos_interface = OVOSInterface(config)
            except Exception as e:
                logger.error(f"Failed to initialize OVOS interface, continuing without OVOS: {e}")
                print(f"WARNING: OVOS integration failed: {e}")
        else:
            logger.info("OVOS integration disabled by configuration")
        
        # Initialize the orchestrator
        logger.info("Initializing system orchestrator...")
        try:
            orchestrator = SystemOrchestrator(
                config=config,
                memory_system=memory_system,
                language_model=language_model,
                audio_system=audio_system,
                ovos_interface=ovos_interface
            )
        except Exception as e:
            logger.critical(f"Failed to initialize system orchestrator: {e}")
            print(f"CRITICAL: Failed to initialize system orchestrator: {e}")
            await shutdown_gracefully(memory_system, language_model, audio_system, ovos_interface, None)
            return 1
        
        # Register signal handlers for graceful shutdown
        register_signal_handlers(memory_system, language_model, audio_system, ovos_interface, orchestrator)
        
        # Start the system with proper error handling
        logger.info("Starting Grace AI System")
        try:
            await orchestrator.start()
        except Exception as e:
            logger.critical(f"Failed to start orchestrator: {e}")
            print(f"CRITICAL: Failed to start orchestrator: {e}")
            await shutdown_gracefully(memory_system, language_model, audio_system, ovos_interface, orchestrator)
            return 1
        
        # Determine interaction mode
        if args.voice_mode:
            interaction_mode = "voice"
        elif args.text_mode:
            interaction_mode = "text"
        else:
            interaction_mode = config.get("system", {}).get("input_mode", "text")
        
        # Start the appropriate interaction loop with proper error handling
        try:
            if interaction_mode == "voice" and audio_system:
                logger.info("Starting voice interaction mode")
                await orchestrator.voice_interaction_loop()
            else:
                logger.info("Starting text interaction mode")
                await orchestrator.text_interaction_loop()
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
        except Exception as e:
            logger.error(f"Error during interaction loop: {e}", exc_info=True)
            print(f"ERROR: Interaction loop failed: {e}")
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Error during Grace AI operation: {e}", exc_info=True)
        print(f"ERROR: Unexpected error: {e}")
    finally:
        # Print memory usage report if profiling was enabled
        if args.profile_memory:
            try:
                import tracemalloc
                current, peak = tracemalloc.get_traced_memory()
                logger.info(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
                logger.info(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")
                
                # Get top memory allocations
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')
                logger.info("Top 10 memory allocations:")
                for stat in top_stats[:10]:
                    logger.info(f"{stat}")
                
                tracemalloc.stop()
            except Exception as e:
                logger.error(f"Error generating memory profile: {e}")
        
        # Clean shutdown
        await shutdown_gracefully(memory_system, language_model, audio_system, ovos_interface, orchestrator)
        logger.info("Grace AI System shutdown complete")
        
    return 0


async def test_memory_system(memory_system):
    """Run tests on the memory system with improved error handling."""
    from grace.memory.types import MemoryType
    
    print("\n==== Testing Memory System ====\n")
    
    try:
        # Test adding a memory
        print("Testing memory addition...")
        memory_id = await memory_system.add_memory(
            "This is a test memory for Grace AI",
            MemoryType.CRITICAL,
            "test_user",
            {"test": True, "topic": "memory_test"}
        )
        
        if not memory_id:
            print("ERROR: Failed to add memory")
            return False
            
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
        
        # Test memory deletion
        print("\nTesting memory deletion...")
        deletion_success = await memory_system.delete_memory(memory_id)
        print(f"Memory deletion {'successful' if deletion_success else 'failed'}")
        
        # Verify deletion
        results_after_delete = await memory_system.search_memories("test memory")
        deleted_count = len(results.get('critical', [])) - len(results_after_delete.get('critical', []))
        print(f"Verified {deleted_count} memories were removed")
        
        print("\n==== Memory System Test Complete ====\n")
        return True
    except Exception as e:
        print(f"ERROR during memory test: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def register_signal_handlers(memory_system, language_model, audio_system, ovos_interface, orchestrator):
    """Register signal handlers for graceful shutdown with improved error handling."""
    loop = asyncio.get_running_loop()
    
    def signal_handler():
        """Handle termination signals with proper error handling."""
        logger = logging.getLogger('grace')
        logger.info("Received termination signal, initiating graceful shutdown")
        
        # Create shutdown task
        shutdown_task = asyncio.create_task(
            shutdown_gracefully(
                memory_system, language_model, audio_system, ovos_interface, orchestrator
            ),
            name="grace_shutdown"
        )
        
        # Add timeout to shutdown
        def shutdown_timeout():
            if not shutdown_task.done():
                logger.warning("Shutdown taking too long, forcing exit")
                # Force Python to exit if shutdown takes too long
                import os
                os._exit(1)
                
        # Set shutdown timeout of 30 seconds
        loop.call_later(30, shutdown_timeout)
    
    # Register for SIGINT (Ctrl+C) and SIGTERM
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)


async def shutdown_gracefully(memory_system, language_model, audio_system, ovos_interface, orchestrator):
    """Gracefully shut down all system components with comprehensive error handling and resource cleanup."""
    logger = logging.getLogger('grace')
    logger.info("Shutting down Grace AI System...")
    
    try:
        # First stop the orchestrator if it exists
        if orchestrator:
            logger.info("Stopping orchestrator...")
            try:
                await orchestrator.stop()
            except Exception as e:
                logger.error(f"Error stopping orchestrator: {e}")
        
        # Shutdown components in reverse order of initialization
        
        # Stop OVOS interface
        if ovos_interface:
            logger.info("Shutting down OVOS interface...")
            try:
                ovos_interface.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down OVOS interface: {e}")
        
        # Stop audio system
        if audio_system:
            logger.info("Shutting down audio system...")
            try:
                audio_system.stop()
            except Exception as e:
                logger.error(f"Error shutting down audio system: {e}")
        
        # Stop language model
        if language_model:
            logger.info("Shutting down language model...")
            try:
                if hasattr(language_model, 'shutdown'):
                    language_model.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down language model: {e}")
        
        # Stop memory system
        if memory_system:
            logger.info("Shutting down memory system...")
            try:
                memory_system.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down memory system: {e}")
        
        # Force garbage collection to free resources
        gc.collect()
        
        # Clear CUDA cache if torch is available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("CUDA cache cleared")
        except ImportError:
            pass
        
        logger.info("Shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    # Handle Windows asyncio issues
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nShutdown requested by keyboard interrupt")
        sys.exit(0)
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
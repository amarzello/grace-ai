"""
Grace AI System - Memory Core Module

This module implements the core memory system integrator for Grace.
"""

import logging
from typing import Dict, Any, List, Optional
import asyncio
import time
from datetime import datetime

from .sqlite import SQLiteStorage
from .vector import VectorStorage
from .critical import CriticalMemoryManager
from .types import MemoryType


class MemorySystem:
    """
    Memory system with vector database and SQLite persistence.
    
    Features:
    - Vector-based contextual memory via mem0
    - Long-term persistent storage with SQLite
    - Support for reference materials
    - Privacy-respecting amnesia mode
    - Memory optimization and maintenance
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the memory system with the provided configuration.
        
        Args:
            config: Memory system configuration
        """
        self.logger = logging.getLogger('grace.memory')
        self.config = config.get('memory', {})
        self.amnesia_mode = config.get('amnesia_mode', False)
        self.debug_mode = config.get('debug', False)
        
        # Memory statistics
        self.memory_stats = {
            "total_memories": 0,
            "last_maintenance": datetime.now(),
            "insertion_count": 0,
            "search_count": 0,
            "last_error": None,
            "last_error_time": None
        }
        
        # Initialize storage components
        self.sqlite = SQLiteStorage(config)
        self.vector = VectorStorage(config)
        self.critical = CriticalMemoryManager(config)
        
        # Track active tasks
        self.active_tasks = []
        
        # Set up periodic maintenance
        self._schedule_maintenance()
        
        if self.amnesia_mode:
            self.logger.warning("Amnesia mode active - no new memories will be stored")
        else:
            # Load memory statistics
            self._load_memory_stats()
            
        self.logger.info("Memory system initialized")
        
    def _schedule_maintenance(self):
        """Schedule periodic memory maintenance tasks."""
        # We'll use a simple approach with timestamp checks
        from datetime import datetime, timedelta
        self.maintenance_interval = timedelta(hours=self.config.get('maintenance_interval_hours', 24))
        self.memory_stats["last_maintenance"] = datetime.now()
        
        # Initialize maintenance state
        self.maintenance_needed = False
        self.maintenance_running = False
        
    def _load_memory_stats(self):
        """Load memory statistics."""
        stats = self.sqlite.get_memory_stats()
        if stats:
            # Update memory stats with loaded values, keeping defaults for missing keys
            for key, value in stats.items():
                if value is not None:
                    self.memory_stats[key] = value
            
    async def add_memory(self, content: str, memory_type: MemoryType, 
                        user_id: str = "grace_user", metadata: Dict = None) -> Optional[str]:
        """
        Add a new memory to the system.
        
        Args:
            content: Memory content
            memory_type: Type of memory
            user_id: User identifier
            metadata: Additional metadata
            
        Returns:
            String identifier of the created memory
        """
        if self.amnesia_mode:
            self.logger.debug("Amnesia mode: not storing new memory")
            return None
            
        # Skip empty content
        if not content or not content.strip():
            return None
            
        # Store based on memory type
        memory_id = None
        
        try:
            # First, store in appropriate vector storage
            vector_id = await self.vector.add_memory(content, memory_type, metadata)
            
            # Next, store in SQLite for persistence
            sql_id = await self.sqlite.add_memory(content, memory_type, user_id, metadata, vector_id)
            
            # If SQLite storage failed, skip critical memory storage too
            if sql_id is None:
                self.logger.error("Failed to add memory to SQLite storage")
                return None
                
            # If it's a critical memory, also store in the critical memory manager
            if memory_type == MemoryType.CRITICAL:
                critical_id = await self.critical.add_memory(content, metadata, sql_id)
                
            # Update memory stats
            self.memory_stats["total_memories"] = self.memory_stats.get("total_memories", 0) + 1
            self.memory_stats["insertion_count"] = self.memory_stats.get("insertion_count", 0) + 1
            
            # Save stats periodically
            if self.memory_stats["insertion_count"] % 10 == 0:
                # Create a task and track it
                task = asyncio.create_task(
                    self.sqlite.save_memory_stats(self.memory_stats),
                    name=f"save_stats_{time.time()}"
                )
                self.active_tasks.append(task)
                task.add_done_callback(lambda t: self.active_tasks.remove(t) if t in self.active_tasks else None)
                
            return sql_id
            
        except Exception as e:
            self.logger.error(f"Error adding memory: {e}")
            self.memory_stats["last_error"] = f"Memory addition error: {e}"
            self.memory_stats["last_error_time"] = datetime.now().isoformat()
            return None
            
    async def search_memories(self, query: str, user_id: str = "grace_user", 
                            limit: int = None) -> Dict[str, List[Dict]]:
        """
        Search memories across all storage types.
        
        Args:
            query: Search query
            user_id: User identifier
            limit: Maximum results to return
            
        Returns:
            Dictionary of memory results by category
        """
        # Check for empty query
        if not query or not query.strip():
            return {
                "contextual": [],
                "critical": [],
                "reference": [],
                "conversations": []
            }
            
        # Update search count
        self.memory_stats["search_count"] = self.memory_stats.get("search_count", 0) + 1
        
        limit = limit or self.config.get('search_limit', 50)
        
        results = {
            "contextual": [],
            "critical": [],
            "reference": [],
            "conversations": []
        }
        
        try:
            # Run searches in parallel for better performance
            vector_task = asyncio.create_task(self.vector.search_memories(query, limit))
            critical_task = asyncio.create_task(self.critical.search_memories(query, limit))
            sql_task = asyncio.create_task(self.sqlite.search_memories(query, limit))
            
            # Wait for all searches to complete
            vector_results, critical_results, sql_results = await asyncio.gather(
                vector_task, critical_task, sql_task
            )
            
            # Process results from each source
            if vector_results and "results" in vector_results:
                results["contextual"] = vector_results["results"]
            
            if critical_results:
                results["critical"] = critical_results
            
            # Add results from SQLite - prioritize more specific categories
            if sql_results:
                for memory_type, memories in sql_results.items():
                    if memory_type in results:
                        results[memory_type] = memories
                
            # Check if maintenance should run (async, non-blocking)
            self._schedule_async_maintenance()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Memory search error: {e}")
            dt_now = datetime.now()
            self.memory_stats["last_error"] = f"Memory search error: {e}"
            self.memory_stats["last_error_time"] = dt_now.isoformat()
            return results
    
    def _schedule_async_maintenance(self):
        """Schedule maintenance as an async task if needed."""
        # Check if maintenance is due
        now = datetime.now()
        last_maintenance = self.memory_stats.get("last_maintenance", now)
        
        # Handle string ISO format conversion if needed
        if isinstance(last_maintenance, str):
            try:
                last_maintenance = datetime.fromisoformat(last_maintenance)
            except ValueError:
                last_maintenance = now
                self.memory_stats["last_maintenance"] = now
        
        if now - last_maintenance >= self.maintenance_interval:
            if not self.maintenance_running:
                # Create a maintenance task and track it
                task = asyncio.create_task(
                    self._check_and_run_maintenance(), 
                    name=f"maintenance_{time.time()}"
                )
                self.active_tasks.append(task)
                task.add_done_callback(lambda t: self.active_tasks.remove(t) if t in self.active_tasks else None)
            
    async def _check_and_run_maintenance(self):
        """Check if maintenance is needed and run it if necessary."""
        if self.amnesia_mode:
            return
            
        # Check if maintenance is due
        now = datetime.now()
        last_maintenance = self.memory_stats.get("last_maintenance", now)
        
        # Handle string ISO format conversion if needed
        if isinstance(last_maintenance, str):
            try:
                last_maintenance = datetime.fromisoformat(last_maintenance)
            except ValueError:
                last_maintenance = now
        
        if now - last_maintenance >= self.maintenance_interval:
            if not self.maintenance_running:
                self.maintenance_running = True
                try:
                    # Run maintenance tasks
                    await self._run_maintenance()
                except Exception as e:
                    self.logger.error(f"Maintenance error: {e}")
                finally:
                    self.maintenance_running = False
                    self.memory_stats["last_maintenance"] = datetime.now()
                    
    async def _run_maintenance(self):
        """Run memory maintenance tasks asynchronously."""
        self.logger.info("Running memory maintenance")
        
        tasks = [
            self.sqlite.archive_old_memories(),
            self.sqlite.optimize_database(),
            self.vector.optimize_indexes()
        ]
        
        # Execute maintenance tasks
        await asyncio.gather(*tasks)
        
        # Update memory stats
        self.memory_stats["last_maintenance"] = datetime.now()
        await self.sqlite.save_memory_stats(self.memory_stats)
        
        self.logger.info("Memory maintenance completed")
        
    async def log_conversation(self, entry):
        """
        Log conversation to storage.
        
        Args:
            entry: Conversation entry to log
        """
        if self.amnesia_mode:
            self.logger.debug("Amnesia mode: not logging conversation")
            return
            
        await self.sqlite.log_conversation(entry)
        
    async def load_reference_materials(self, directory):
        """
        Load reference materials into memory.
        
        Args:
            directory: Directory containing reference materials
            
        Returns:
            Number of files loaded
        """
        if self.amnesia_mode:
            self.logger.info("Amnesia mode: not loading reference materials")
            return 0
            
        from pathlib import Path
        import os
        
        directory_path = Path(directory)
        if not directory_path.exists() or not directory_path.is_dir():
            self.logger.error(f"Reference directory does not exist: {directory}")
            return 0
            
        # File types to process
        text_extensions = {'.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.yaml', '.yml', '.csv'}
        
        files_loaded = 0
        
        # Walk through directory and process files
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = Path(root) / file
                
                # Check if it's a supported file type
                if file_path.suffix.lower() in text_extensions:
                    try:
                        # Read file content
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        # Create metadata
                        metadata = {
                            'file_path': str(file_path),
                            'file_name': file,
                            'file_type': file_path.suffix.lower(),
                            'file_size': file_path.stat().st_size,
                            'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                        }
                        
                        # Add to memory as reference material
                        memory_id = await self.add_memory(
                            content=content,
                            memory_type=MemoryType.REFERENCE,
                            user_id="system",
                            metadata=metadata
                        )
                        
                        if memory_id:
                            files_loaded += 1
                    except Exception as e:
                        self.logger.error(f"Error loading reference file {file_path}: {e}")
        
        self.logger.info(f"Loaded {files_loaded} reference files from {directory}")
        return files_loaded
    
    async def get_memory_stats(self):
        """
        Get memory system statistics.
        
        Returns:
            Dictionary of memory statistics
        """
        stats = self.memory_stats.copy()
        
        # Run storage stats functions in parallel
        sqlite_task = asyncio.create_task(self.sqlite.get_stats())
        vector_task = asyncio.create_task(self.vector.get_stats())
        critical_task = asyncio.create_task(self.critical.get_stats())
        
        # Wait for all stats to complete
        sqlite_stats, vector_stats, critical_stats = await asyncio.gather(
            sqlite_task, vector_task, critical_task
        )
        
        stats.update({
            "sqlite": sqlite_stats,
            "vector": vector_stats,
            "critical": critical_stats
        })
        
        return stats
    
    async def delete_memory(self, memory_id: str, memory_type: MemoryType = None) -> bool:
        """
        Delete a memory from all storage components.
        
        Args:
            memory_id: ID of the memory to delete
            memory_type: Optional memory type for more targeted deletion
            
        Returns:
            Success status
        """
        if self.amnesia_mode:
            self.logger.debug("Amnesia mode: delete operation not available")
            return False
        
        success = True
        # Delete from SQLite and get additional IDs
        vector_id, critical_id = await self.sqlite.delete_memory(memory_id)
        
        # Delete from vector storage if we have the ID
        if vector_id:
            vector_success = await self.vector.delete_memory(vector_id)
            if not vector_success:
                success = False
                self.logger.warning(f"Failed to delete memory from vector storage: {vector_id}")
        
        # Delete from critical if needed
        if critical_id or (memory_type == MemoryType.CRITICAL):
            critical_success = await self.critical.delete_memory(critical_id or memory_id)
            if not critical_success:
                success = False
                self.logger.warning(f"Failed to delete memory from critical storage: {critical_id or memory_id}")
        
        return success
    
    def shutdown(self):
        """Clean shutdown of memory systems."""
        self.logger.info("Shutting down memory system")
        
        # Wait for any active tasks to complete
        if self.active_tasks:
            self.logger.info(f"Waiting for {len(self.active_tasks)} memory tasks to complete")
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, create a task to wait for completion
                try:
                    # Create a gather future for all tasks
                    gather_future = asyncio.gather(*self.active_tasks, return_exceptions=True)
                    # Set a timeout to avoid hanging
                    loop.call_later(2, gather_future.cancel)
                except Exception as e:
                    self.logger.warning(f"Error waiting for memory tasks: {e}")
        
        # Close all components
        self.sqlite.close()
        self.vector.close()
        self.critical.close()
        
        self.logger.info("Memory system shutdown complete")
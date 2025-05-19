"""
Grace AI System - Memory Core Module

This module implements the hybrid memory system integrator for Grace with improved
resource management, error handling, and Neo4j/Qdrant/SQLite integration.
"""

import logging
import asyncio
import time
import gc
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor

from .sqlite import SQLiteStorage
from .vector import VectorStorage
from .critical import CriticalMemoryManager
from .types import MemoryType
from grace.utils.common import calculate_relevance


class MemorySystem:
    """
    Hybrid memory system combining vector, graph, and relational databases.
    
    Features:
    - Vector-based semantic search via Qdrant
    - Graph relationships via Neo4j
    - Long-term persistent storage with SQLite
    - Critical memory management
    - Privacy-respecting amnesia mode
    - Automatic memory optimization and maintenance
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
        
        # Memory statistics tracking
        self.memory_stats = {
            "total_memories": 0,
            "last_maintenance": datetime.now(),
            "insertion_count": 0,
            "search_count": 0,
            "last_error": None,
            "last_error_time": None
        }
        
        # Thread pool for background operations
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="memory-pool")
        
        # Initialize storage components with proper error handling
        try:
            self.sqlite = SQLiteStorage(config)
            self.vector = VectorStorage(config)
            self.critical = CriticalMemoryManager(config)
            
            # Load memory statistics
            if not self.amnesia_mode:
                self._load_memory_stats()
        except Exception as e:
            self.logger.critical(f"Failed to initialize memory system: {e}")
            raise
        
        # Track active tasks
        self.active_tasks = []
        self.tasks_lock = threading.RLock()
        
        # Set up periodic maintenance
        self._schedule_maintenance()
        
        if self.amnesia_mode:
            self.logger.warning("Amnesia mode active - no new memories will be stored")
            
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
        """Load memory statistics from SQLite storage."""
        try:
            stats = self.sqlite.get_memory_stats()
            if stats:
                # Update memory stats with loaded values, keeping defaults for missing keys
                for key, value in stats.items():
                    if value is not None:
                        self.memory_stats[key] = value
        except Exception as e:
            self.logger.error(f"Failed to load memory stats: {e}")
            
    async def add_memory(self, content: str, memory_type: MemoryType, 
                        user_id: str = "grace_user", metadata: Dict = None) -> Optional[str]:
        """
        Add a new memory to the hybrid system with proper integration.
        
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
            # 1. First, store in vector database for semantic search
            vector_id = await self.vector.add_memory(content, memory_type, user_id, metadata)
            
            # 2. Store in SQLite for persistence and relational queries
            sql_id = await self.sqlite.add_memory(content, memory_type, user_id, metadata, vector_id)
            
            # If SQLite storage failed, skip critical memory storage too
            if sql_id is None:
                self.logger.error("Failed to add memory to SQLite storage")
                return None
                
            # 3. If it's a critical memory, also store in the critical memory manager
            if memory_type == MemoryType.CRITICAL:
                critical_id = await self.critical.add_memory(content, metadata, sql_id)
                self.logger.debug(f"Added critical memory with ID: {critical_id}")
            
            # Update memory stats
            self.memory_stats["total_memories"] = self.memory_stats.get("total_memories", 0) + 1
            self.memory_stats["insertion_count"] = self.memory_stats.get("insertion_count", 0) + 1
            
            # Save stats periodically
            if self.memory_stats["insertion_count"] % 10 == 0:
                # Create a task and track it
                with self.tasks_lock:
                    task = asyncio.create_task(
                        self.sqlite.save_memory_stats(self.memory_stats),
                        name=f"save_stats_{time.time()}"
                    )
                    self.active_tasks.append(task)
                    task.add_done_callback(lambda t: self._task_done_callback(t))
                
            return sql_id
            
        except Exception as e:
            self.logger.error(f"Error adding memory: {e}")
            self.memory_stats["last_error"] = f"Memory addition error: {e}"
            self.memory_stats["last_error_time"] = datetime.now().isoformat()
            return None
    
    def _task_done_callback(self, task):
        """Callback to remove completed tasks from tracking."""
        with self.tasks_lock:
            if task in self.active_tasks:
                self.active_tasks.remove(task)
            
    async def search_memories(self, query: str, user_id: str = "grace_user", 
                            limit: int = None) -> Dict[str, List[Dict]]:
        """
        Search memories across all storage types with improved integration.
        
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
                with self.tasks_lock:
                    task = asyncio.create_task(
                        self._check_and_run_maintenance(), 
                        name=f"maintenance_{time.time()}"
                    )
                    self.active_tasks.append(task)
                    task.add_done_callback(lambda t: self._task_done_callback(t))
            
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
        """Run memory maintenance tasks asynchronously for all storage systems."""
        self.logger.info("Running memory maintenance")
        
        tasks = [
            self.sqlite.archive_old_memories(),
            self.sqlite.optimize_database(),
            self.vector.optimize_indexes()
        ]
        
        # Execute maintenance tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check for errors
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Maintenance task {i} failed: {result}")
        
        # Update memory stats
        self.memory_stats["last_maintenance"] = datetime.now()
        await self.sqlite.save_memory_stats(self.memory_stats)
        
        self.logger.info("Memory maintenance completed")
        
    async def log_conversation(self, entry):
        """
        Log conversation to storage with proper error handling.
        
        Args:
            entry: Conversation entry to log
        """
        if self.amnesia_mode:
            self.logger.debug("Amnesia mode: not logging conversation")
            return
        
        try:    
            await self.sqlite.log_conversation(entry)
        except Exception as e:
            self.logger.error(f"Error logging conversation: {e}")
        
    async def load_reference_materials(self, directory):
        """
        Load reference materials into memory with improved error handling.
        
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
            
        # File types to process with extended support
        text_extensions = {'.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.yaml', '.yml', 
                          '.csv', '.conf', '.ini', '.toml', '.xml', '.c', '.cpp', '.h', '.java', 
                          '.rs', '.go', '.ts', '.sh', '.bat', '.ps1'}
        
        files_loaded = 0
        processed_files = 0
        skipped_files = 0
        errors = 0
        
        # Walk through directory and process files
        for root, _, files in os.walk(directory_path):
            # Skip hidden folders
            if '/.' in root or '\\.' in root:
                continue
                
            for file in files:
                # Skip hidden files
                if file.startswith('.'):
                    continue
                    
                file_path = Path(root) / file
                processed_files += 1
                
                # Check if it's a supported file type
                if file_path.suffix.lower() in text_extensions:
                    try:
                        # Check file size - skip files too large
                        file_size = file_path.stat().st_size
                        max_size = self.config.get('max_reference_file_size', 1024 * 1024)  # 1MB default
                        
                        if file_size > max_size:
                            self.logger.warning(f"File too large ({file_size} bytes), skipping: {file_path}")
                            skipped_files += 1
                            continue
                            
                        # Read file content with proper encoding detection
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                        except UnicodeDecodeError:
                            # Try with other encodings
                            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                                try:
                                    with open(file_path, 'r', encoding=encoding) as f:
                                        content = f.read()
                                    break
                                except UnicodeDecodeError:
                                    continue
                            else:
                                self.logger.warning(f"Unsupported encoding in {file_path}, skipping")
                                skipped_files += 1
                                continue
                            
                        # Create metadata
                        metadata = {
                            'file_path': str(file_path),
                            'file_name': file,
                            'file_type': file_path.suffix.lower(),
                            'file_size': file_size,
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
                        errors += 1
                else:
                    skipped_files += 1
        
        self.logger.info(f"Loaded {files_loaded} reference files from {directory}")
        self.logger.debug(f"Processed: {processed_files}, Skipped: {skipped_files}, Errors: {errors}")
        return files_loaded
    
    async def get_memory_stats(self):
        """
        Get comprehensive memory system statistics.
        
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
            sqlite_task, vector_task, critical_task, return_exceptions=True
        )
        
        # Handle any exceptions in the tasks
        if isinstance(sqlite_stats, Exception):
            self.logger.error(f"Error getting SQLite stats: {sqlite_stats}")
            sqlite_stats = {"error": str(sqlite_stats)}
            
        if isinstance(vector_stats, Exception):
            self.logger.error(f"Error getting vector stats: {vector_stats}")
            vector_stats = {"error": str(vector_stats)}
            
        if isinstance(critical_stats, Exception):
            self.logger.error(f"Error getting critical stats: {critical_stats}")
            critical_stats = {"error": str(critical_stats)}
        
        stats.update({
            "sqlite": sqlite_stats,
            "vector": vector_stats,
            "critical": critical_stats,
            "active_tasks": len(self.active_tasks)
        })
        
        return stats
    
    async def delete_memory(self, memory_id: str, memory_type: MemoryType = None) -> bool:
        """
        Delete a memory from all storage components with proper cascading.
        
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
        
        try:
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
        except Exception as e:
            self.logger.error(f"Error deleting memory: {e}")
            return False
    
    async def verify_response(self, response_text: str, context: Dict) -> Dict:
        """
        Verify response against memory for accuracy with robust algorithm.
        
        Args:
            response_text: Response text to verify
            context: Context information including user input and memories
            
        Returns:
            Verification result with confidence score
        """
        # This is a basic implementation - in a production system, you'd use
        # the language model to verify the response against retrieved memories
        
        if not response_text or not context:
            return {"score": 1.0, "status": "no_verification_needed"}
            
        # Extract memories from context
        memories = context.get('memory_context', {})
        if not memories:
            return {"score": 1.0, "status": "no_memories_available"}
            
        # Calculate relevance between response and available memories
        max_relevance = 0.0
        
        critical_memories = memories.get('critical', [])
        for memory in critical_memories:
            if isinstance(memory, dict) and 'content' in memory:
                relevance = calculate_relevance(response_text, memory['content'])
                max_relevance = max(max_relevance, relevance)
                
        contextual_memories = memories.get('contextual', [])
        for memory in contextual_memories:
            if isinstance(memory, dict) and ('content' in memory or 'memory' in memory):
                content = memory.get('content', memory.get('memory', ''))
                relevance = calculate_relevance(response_text, content)
                max_relevance = max(max_relevance, relevance)
        
        # Calculate verification score
        verification_threshold = self.config.get('verification_threshold', 0.8)
        
        # Score is between 0.5 and 1.0
        # 0.5 = no relevant memories
        # 1.0 = perfect match with memories
        score = 0.5 + (max_relevance * 0.5)
        
        result = {
            "score": score,
            "threshold": verification_threshold,
            "status": "verified" if score >= verification_threshold else "uncertain"
        }
        
        return result
    
    def shutdown(self):
        """Clean shutdown of memory systems with proper resource handling."""
        self.logger.info("Shutting down memory system")
        
        # Cancel and wait for any active tasks to complete
        if self.active_tasks:
            with self.tasks_lock:
                active_count = len(self.active_tasks)
                self.logger.info(f"Waiting for {active_count} memory tasks to complete")
                
                for task in self.active_tasks[:]:
                    if not task.done():
                        task.cancel()
        
        # Close all components
        self.sqlite.close()
        self.vector.close()
        self.critical.close()
        
        # Shutdown thread pool
        self.executor.shutdown(wait=False)
        
        # Force garbage collection
        gc.collect()
        
        self.logger.info("Memory system shutdown complete")

# Add HybridMemorySystem class as described in the overhaul document

class HybridMemorySystem:
    """A hybrid memory system combining vector, graph, and relational databases."""
    
    def __init__(self, vector_db_url, graph_db_url, sqlite_path):
        """
        Initialize the hybrid memory system with all databases.
        
        Args:
            vector_db_url: URL for the vector database
            graph_db_url: URL for the graph database
            sqlite_path: Path to SQLite database
        """
        # Initialize vector database (Qdrant)
        from qdrant_client import QdrantClient
        self.vector_db = QdrantClient(url=vector_db_url)
        
        # Initialize graph database (Neo4j)
        from neo4j import GraphDatabase
        self.graph_db = GraphDatabase.driver(graph_db_url, auth=("neo4j", "password"))
        
        # Initialize relational database (SQLite)
        self.sqlite_path = sqlite_path
        self.sqlite_pool = None
    
    async def initialize(self):
        """Initialize databases and set up schema."""
        # Initialize SQLite pool
        from contextlib import asynccontextmanager
        import aiosqlite
        
        class SQLitePool:
            """Connection pool for thread-safe SQLite access."""
            def __init__(self, database_path, max_connections=5):
                self.database_path = database_path
                self.max_connections = max_connections
                self._pool = asyncio.Queue(maxsize=max_connections)
                self._connections = 0
                self._lock = asyncio.Lock()

            async def initialize(self):
                """Initialize the connection pool."""
                for _ in range(self.max_connections):
                    await self._add_connection()

            async def _add_connection(self):
                """Add a new connection to the pool."""
                async with self._lock:
                    if self._connections >= self.max_connections:
                        return
                    
                    conn = aiosqlite.connect(self.database_path)
                    await conn.execute("PRAGMA journal_mode = WAL")
                    await conn.execute("PRAGMA synchronous = NORMAL")
                    await conn.execute("PRAGMA foreign_keys = ON")
                    await conn.execute("PRAGMA cache_size = -8000")
                    self._connections += 1
                    await self._pool.put(conn)

            @asynccontextmanager
            async def acquire(self):
                """Acquire a connection from the pool."""
                conn = await self._pool.get()
                try:
                    yield conn
                finally:
                    await self._pool.put(conn)

            async def close(self):
                """Close all connections in the pool."""
                while not self._pool.empty():
                    try:
                        conn = await self._pool.get()
                        await conn.close()
                        self._connections -= 1
                    except Exception as e:
                        logging.error(f"Error closing connection: {e}")
        
        self.sqlite_pool = SQLitePool(self.sqlite_path)
        await self.sqlite_pool.initialize()
        
        # Set up schema
        async with self.sqlite_pool.acquire() as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS memory_metadata (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP,
                    access_count INTEGER DEFAULT 0
                )
            """)
            await db.commit()
    
    async def store_memory(self, text, embedding, metadata=None):
        """
        Store a memory in the hybrid system.
        
        Args:
            text: Memory text content
            embedding: Vector embedding
            metadata: Additional metadata
            
        Returns:
            Memory identifier
        """
        import uuid
        # Generate a unique ID
        memory_id = str(uuid.uuid4())
        
        # 1. Store in vector database for semantic search
        from qdrant_client.models import PointStruct
        try:
            self.vector_db.upsert(
                collection_name="memories",
                points=[PointStruct(
                    id=memory_id,
                    vector=embedding,
                    payload={"text": text, "metadata": metadata or {}}
                )]
            )
        except Exception as e:
            # Log error and continue with other storage
            logging.error(f"Vector DB storage failed: {e}")
        
        # 2. Store in graph database for relational queries
        try:
            with self.graph_db.session() as session:
                session.execute_write(
                    lambda tx: tx.run(
                        """
                        CREATE (m:Memory {id: $id, text: $text})
                        RETURN m
                        """,
                        id=memory_id,
                        text=text
                    )
                )
        except Exception as e:
            # Log error and continue with other storage
            logging.error(f"Graph DB storage failed: {e}")
        
        # 3. Store metadata in relational database
        try:
            async with self.sqlite_pool.acquire() as db:
                await db.execute(
                    """
                    INSERT INTO memory_metadata (id, type, created_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    """,
                    (memory_id, metadata.get("type", "general") if metadata else "general")
                )
                await db.commit()
        except Exception as e:
            # Log error
            logging.error(f"SQLite storage failed: {e}")
        
        return memory_id
    
    async def close(self):
        """Close all database connections."""
        # Vector DB doesn't need explicit closing
        
        # Close Neo4j driver
        self.graph_db.close()
        
        # Close SQLite pool
        if self.sqlite_pool:
            await self.sqlite_pool.close()
"""
Grace AI System - SQLite Storage Module

This module handles SQLite-based storage for the Grace memory system with improved
async patterns, connection pooling, and transaction handling.
"""

import os
import json
import sqlite3
import logging
import shutil
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from grace.utils.common import MEMORY_DB_PATH, ConversationEntry
from .types import MemoryType


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
            
            conn = sqlite3.connect(self.database_path)
            # Configure for optimal performance
            conn.execute("PRAGMA journal_mode = WAL")  # Write-Ahead Logging
            conn.execute("PRAGMA synchronous = NORMAL")  # Balance durability and speed
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA cache_size = -8000")  # Use ~8MB of memory for cache
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
                conn.close()
                self._connections -= 1
            except Exception as e:
                logging.error(f"Error closing connection: {e}")


class SQLiteStorage:
    """SQLite storage for persistent memory and conversation logging with improved async patterns."""
    
    # Current database schema version
    SCHEMA_VERSION = 3
    
    def __init__(self, config: Dict):
        """
        Initialize SQLite storage with the provided configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger('grace.memory.sqlite')
        self.config = config.get('memory', {})
        self.amnesia_mode = config.get('amnesia_mode', False)
        
        # Set up database paths and ensure directories exist
        MEMORY_DB_PATH.mkdir(parents=True, exist_ok=True)
        self.long_term_db = MEMORY_DB_PATH / 'long_term.db'
        self.conversation_db = MEMORY_DB_PATH / 'conversations.db'
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Connection pools
        self.long_term_pool = None
        self.conversation_pool = None
        
        # Initialize databases
        self._init_long_term_storage()
        self._init_conversation_db()
        
        # Initialize connection pools
        asyncio.create_task(self._init_pools())
        
    async def _init_pools(self):
        """Initialize connection pools."""
        self.long_term_pool = SQLitePool(self.long_term_db)
        self.conversation_pool = SQLitePool(self.conversation_db)
        
        await self.long_term_pool.initialize()
        await self.conversation_pool.initialize()
        
    def _init_long_term_storage(self):
        """Initialize long-term storage database with error handling and schema migration."""
        try:
            # Check if database folder exists
            if not self.long_term_db.parent.exists():
                self.long_term_db.parent.mkdir(parents=True, exist_ok=True)
                
            conn = sqlite3.connect(self.long_term_db)
            cursor = conn.cursor()
            
            # Enable foreign keys support and WAL mode for better concurrency
            cursor.execute("PRAGMA foreign_keys = ON")
            
            # Only enable WAL mode if configured to do so
            if self.config.get('sqlite_wal_mode', True):
                cursor.execute("PRAGMA journal_mode = WAL")
                cursor.execute("PRAGMA synchronous = NORMAL")
                cursor.execute("PRAGMA cache_size = -8000")  # Use ~8MB of memory for cache
            
            # Check if database is valid
            try:
                cursor.execute("PRAGMA integrity_check")
                result = cursor.fetchone()
                if result[0] != "ok":
                    self.logger.warning(f"Database integrity check failed: {result[0]}")
                    
                    # Create backup of corrupted database
                    backup_path = self.long_term_db.with_suffix('.db.bak')
                    if self.long_term_db.exists():
                        shutil.copy(self.long_term_db, backup_path)
                        self.logger.info(f"Created backup of corrupted database at {backup_path}")
            except Exception as e:
                self.logger.warning(f"Could not check database integrity: {e}")
            
            # Check schema version and migrate if needed
            self._check_db_version(conn)
            
            # Create table if it doesn't exist
            cursor.execute("BEGIN TRANSACTION")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS long_term_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    vector_id TEXT,
                    verification_score REAL DEFAULT 1.0,
                    last_accessed TEXT,
                    access_count INTEGER DEFAULT 0,
                    archived INTEGER DEFAULT 0,
                    importance_score REAL DEFAULT 0.5,
                    knowledge_graph_id TEXT
                )
            """)
            
            # Create indices for better performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_type ON long_term_memory(memory_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON long_term_memory(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_verification ON long_term_memory(verification_score)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_archived ON long_term_memory(archived)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_importance ON long_term_memory(importance_score)
            """)
            
            # Create a separate search table with FTS5 for efficient text search
            try:
                cursor.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS memory_search
                    USING FTS5(content, metadata)
                """)
            except sqlite3.Error as e:
                # FTS5 might not be available, try FTS4 as fallback
                self.logger.warning(f"FTS5 not available, trying FTS4: {e}")
                try:
                    cursor.execute("""
                        CREATE VIRTUAL TABLE IF NOT EXISTS memory_search
                        USING FTS4(content, metadata)
                    """)
                except sqlite3.Error as e2:
                    error_msg = f"Neither FTS5 nor FTS4 is available. Full-text search functionality will be limited: {e2}"
                    self.logger.warning(error_msg)
            
            # Create memory_stats table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_stats (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            
            cursor.execute("COMMIT")
            conn.close()
            
            self.logger.info("Long-term storage initialized")
        except Exception as e:
            self.logger.critical(f"Failed to initialize long-term storage: {e}")
            # Make sure transaction is rolled back
            try:
                cursor.execute("ROLLBACK")
                conn.close()
            except Exception:
                pass
            raise

    def _init_conversation_db(self):
        """Initialize SQLite database for conversation logs with error handling."""
        try:
            # Check if database folder exists
            if not self.conversation_db.parent.exists():
                self.conversation_db.parent.mkdir(parents=True, exist_ok=True)
                
            conn = sqlite3.connect(self.conversation_db)
            cursor = conn.cursor()
            
            # Enable WAL mode for better concurrency if configured
            if self.config.get('sqlite_wal_mode', True):
                cursor.execute("PRAGMA journal_mode = WAL")
                cursor.execute("PRAGMA synchronous = NORMAL")
                cursor.execute("PRAGMA cache_size = -8000")  # Use ~8MB of memory for cache
            
            # Create table with transaction support
            cursor.execute("BEGIN TRANSACTION")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    user_input TEXT,
                    stt_transcript TEXT,
                    memory_context TEXT,
                    prompt TEXT,
                    model_response TEXT,
                    thinking_process TEXT,
                    json_response TEXT,
                    command_result TEXT,
                    tts_output TEXT,
                    error TEXT,
                    metadata TEXT,
                    verification_result TEXT,
                    knowledge_graph_id TEXT,
                    archived INTEGER DEFAULT 0,
                    importance_score REAL DEFAULT 0.5
                )
            """)
            
            # Create indices for efficient queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_conv_timestamp ON conversations(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_conv_user_input ON conversations(user_input)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_conv_archived ON conversations(archived)
            """)
            
            # Create FTS index for full-text search (try FTS5 first, with fallback to FTS4)
            try:
                cursor.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS conversation_search
                    USING FTS5(user_input, model_response, thinking_process)
                """)
            except sqlite3.Error as e:
                # FTS5 might not be available, try FTS4 as fallback
                self.logger.warning(f"FTS5 not available for conversation search, trying FTS4: {e}")
                try:
                    cursor.execute("""
                        CREATE VIRTUAL TABLE IF NOT EXISTS conversation_search
                        USING FTS4(user_input, model_response, thinking_process)
                    """)
                except sqlite3.Error as e2:
                    self.logger.warning(f"Neither FTS5 nor FTS4 is available. Conversation search will be limited: {e2}")
                    
            cursor.execute("COMMIT")
            conn.close()
            
            self.logger.info("Conversation database initialized")
        except Exception as e:
            self.logger.critical(f"Failed to initialize conversation database: {e}")
            # Make sure transaction is rolled back
            try:
                cursor.execute("ROLLBACK")
                conn.close()
            except Exception:
                pass
            raise
    
    def _check_db_version(self, conn):
        """Check database version and perform migrations if needed."""
        cursor = conn.cursor()
        
        # Create version table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS db_version (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                version INTEGER NOT NULL
            )
        """)
        
        # Check current version
        cursor.execute("SELECT version FROM db_version WHERE id = 1")
        row = cursor.fetchone()
        current_version = 0
        
        if row:
            current_version = row[0]
        else:
            # First time setup - insert version
            cursor.execute("INSERT INTO db_version (id, version) VALUES (1, 0)")
            conn.commit()
        
        # Perform migrations if needed
        if current_version < self.SCHEMA_VERSION:
            self._migrate_database(conn, current_version)
            
            # Update version
            cursor.execute("UPDATE db_version SET version = ? WHERE id = 1", (self.SCHEMA_VERSION,))
            conn.commit()

    def _migrate_database(self, conn, current_version):
        """
        Perform database schema migrations.
        
        Args:
            conn: SQLite connection
            current_version: Current schema version
        """
        cursor = conn.cursor()
        
        try:
            # Start transaction
            cursor.execute("BEGIN TRANSACTION")
            
            # Migration paths
            if current_version < 1:
                # Initial schema or pre-versioning schema
                # Check if table exists - if not, it will be created in _init_long_term_storage
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='long_term_memory'")
                if cursor.fetchone():
                    # Table exists, check for columns
                    self._ensure_column_exists(cursor, "long_term_memory", "last_accessed", "TEXT")
                    self._ensure_column_exists(cursor, "long_term_memory", "access_count", "INTEGER DEFAULT 0")
            
            if current_version < 2:
                # Create memory_stats table if it doesn't exist
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS memory_stats (
                        key TEXT PRIMARY KEY,
                        value TEXT
                    )
                """)
                
                # Add verification_score column if it doesn't exist
                self._ensure_column_exists(cursor, "long_term_memory", "verification_score", "REAL DEFAULT 1.0")
            
            if current_version < 3:
                # Add archived column to support archiving instead of deletion
                self._ensure_column_exists(cursor, "long_term_memory", "archived", "INTEGER DEFAULT 0")
                
                # Add importance_score column for better memory prioritization
                self._ensure_column_exists(cursor, "long_term_memory", "importance_score", "REAL DEFAULT 0.5")
                
                # Add knowledge_graph_id column for better integration with knowledge graph
                self._ensure_column_exists(cursor, "long_term_memory", "knowledge_graph_id", "TEXT")
                
                # Create index for archived column
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_archived ON long_term_memory(archived)")
                
                # Create index for importance_score
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_importance ON long_term_memory(importance_score)")
            
            # Commit transaction
            cursor.execute("COMMIT")
            
            self.logger.info(f"Database migrated from version {current_version} to {self.SCHEMA_VERSION}")
        except Exception as e:
            self.logger.error(f"Database migration error: {e}")
            
            # Rollback on error
            try:
                cursor.execute("ROLLBACK")
            except Exception:
                pass
            
            raise

    def _ensure_column_exists(self, cursor, table, column, column_type):
        """
        Add column to table if it doesn't exist.
        
        Args:
            cursor: SQLite cursor
            table: Table name
            column: Column name
            column_type: Column type definition
        """
        # Check if column exists
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [info[1] for info in cursor.fetchall()]
        
        if column not in columns:
            self.logger.info(f"Adding column {column} to table {table}")
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

    def get_memory_stats(self):
        """
        Get memory statistics from database.
        
        Returns:
            Dictionary of memory statistics
        """
        stats = {}
        try:
            with sqlite3.connect(self.long_term_db) as conn:
                cursor = conn.cursor()
                
                # Check if table exists first
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_stats'")
                if cursor.fetchone() is None:
                    return stats
                    
                cursor.execute("SELECT key, value FROM memory_stats")
                rows = cursor.fetchall()
                
                for key, value in rows:
                    if key in ["total_memories", "insertion_count", "search_count"]:
                        stats[key] = int(value)
                    elif key == "last_maintenance":
                        try:
                            stats[key] = datetime.fromisoformat(value)
                        except ValueError:
                            # Handle incorrect date format
                            stats[key] = datetime.now()
                    elif key == "last_error_time":
                        try:
                            stats[key] = datetime.fromisoformat(value)
                        except ValueError:
                            # Handle incorrect date format
                            stats[key] = None
                    else:
                        stats[key] = value
        except Exception as e:
            self.logger.error(f"Failed to load memory stats: {e}")
            
        return stats

    async def save_memory_stats(self, stats: Dict):
        """
        Save memory statistics to database.
        
        Args:
            stats: Dictionary of memory statistics
        """
        if self.amnesia_mode:
            return
        
        # Wait for pools to initialize
        if self.long_term_pool is None:
            await self._init_pools()
            
        async def _save_stats():
            try:
                async with self.long_term_pool.acquire() as conn:
                    cursor = conn.cursor()
                    
                    # Check if table exists first
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_stats'")
                    if cursor.fetchone() is None:
                        # Create table if it doesn't exist
                        cursor.execute("""
                            CREATE TABLE IF NOT EXISTS memory_stats (
                                key TEXT PRIMARY KEY,
                                value TEXT
                            )
                        """)
                    
                    cursor.execute("BEGIN TRANSACTION")
                    
                    # Convert datetime to string for storage
                    stats_dict = stats.copy()
                    if isinstance(stats_dict.get("last_maintenance"), datetime):
                        stats_dict["last_maintenance"] = stats_dict["last_maintenance"].isoformat()
                    if isinstance(stats_dict.get("last_error_time"), datetime):
                        stats_dict["last_error_time"] = stats_dict["last_error_time"].isoformat()
                    
                    # Update or insert stats
                    for key, value in stats_dict.items():
                        if value is not None:  # Skip None values
                            cursor.execute("""
                                INSERT OR REPLACE INTO memory_stats (key, value)
                                VALUES (?, ?)
                            """, (key, str(value)))
                    
                    cursor.execute("COMMIT")
            except Exception as e:
                self.logger.error(f"Failed to save memory stats: {e}")
                try:
                    cursor.execute("ROLLBACK")
                except Exception:
                    pass
                raise
        
        try:
            await _save_stats()
        except Exception as e:
            # Fall back to direct connection if pool fails
            self.logger.warning(f"Pool-based stats save failed, trying direct connection: {e}")
            
            def _direct_save():
                try:
                    with sqlite3.connect(self.long_term_db) as conn:
                        cursor = conn.cursor()
                        
                        # Check if table exists first
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_stats'")
                        if cursor.fetchone() is None:
                            # Create table if it doesn't exist
                            cursor.execute("""
                                CREATE TABLE IF NOT EXISTS memory_stats (
                                    key TEXT PRIMARY KEY,
                                    value TEXT
                                )
                            """)
                        
                        cursor.execute("BEGIN TRANSACTION")
                        
                        # Convert datetime to string for storage
                        stats_dict = stats.copy()
                        if isinstance(stats_dict.get("last_maintenance"), datetime):
                            stats_dict["last_maintenance"] = stats_dict["last_maintenance"].isoformat()
                        if isinstance(stats_dict.get("last_error_time"), datetime):
                            stats_dict["last_error_time"] = stats_dict["last_error_time"].isoformat()
                        
                        # Update or insert stats
                        for key, value in stats_dict.items():
                            if value is not None:  # Skip None values
                                cursor.execute("""
                                    INSERT OR REPLACE INTO memory_stats (key, value)
                                    VALUES (?, ?)
                                """, (key, str(value)))
                        
                        conn.commit()
                except Exception as e:
                    self.logger.error(f"Direct stats save failed: {e}")
                    try:
                        conn.rollback()
                    except Exception:
                        pass
            
            # Run in thread pool
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self.executor, _direct_save)

    async def add_memory(self, content: str, memory_type: MemoryType, 
                        user_id: str, metadata: Dict = None, vector_id: str = None):
        """
        Add memory to SQLite storage.
        
        Args:
            content: Memory content
            memory_type: Type of memory
            user_id: User identifier
            metadata: Additional metadata
            vector_id: Vector storage identifier
            
        Returns:
            ID of the inserted memory
        """
        if self.amnesia_mode:
            return None
            
        timestamp = datetime.now().isoformat()
        metadata = metadata or {}
        metadata['timestamp'] = timestamp
        metadata['memory_type'] = memory_type.value if hasattr(memory_type, 'value') else str(memory_type)
        
        # Wait for pools to initialize
        if self.long_term_pool is None:
            await self._init_pools()
        
        async def _add_memory_with_pool():
            try:
                async with self.long_term_pool.acquire() as conn:
                    cursor = conn.cursor()
                    
                    # Insert memory
                    cursor.execute("""
                        INSERT INTO long_term_memory (
                            timestamp, memory_type, content, metadata,
                            vector_id, verification_score, last_accessed, 
                            access_count, archived, importance_score
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        timestamp,
                        memory_type.value if hasattr(memory_type, 'value') else str(memory_type),
                        content,
                        json.dumps(metadata),
                        vector_id,
                        1.0,  # Default verification score
                        timestamp,  # last_accessed = now
                        1,  # Initial access count
                        0,  # Not archived
                        0.5  # Default importance score
                    ))
                    
                    # Get the inserted ID
                    cursor.execute("SELECT last_insert_rowid()")
                    memory_id = cursor.fetchone()[0]
                    
                    # Also add to search index
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_search'")
                    if cursor.fetchone():
                        cursor.execute("""
                            INSERT INTO memory_search (rowid, content, metadata)
                            VALUES (?, ?, ?)
                        """, (memory_id, content, json.dumps(metadata)))
                        
                    return memory_id
            except Exception as e:
                self.logger.error(f"Error adding memory to SQLite with pool: {e}")
                raise
        
        try:
            return await _add_memory_with_pool()
        except Exception as e:
            # Fall back to direct connection if pool fails
            self.logger.warning(f"Pool-based memory add failed, trying direct connection: {e}")
            
            def _add_memory_direct():
                try:
                    with sqlite3.connect(self.long_term_db) as conn:
                        cursor = conn.cursor()
                        
                        # Insert memory
                        cursor.execute("""
                            INSERT INTO long_term_memory (
                                timestamp, memory_type, content, metadata,
                                vector_id, verification_score, last_accessed, 
                                access_count, archived, importance_score
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            timestamp,
                            memory_type.value if hasattr(memory_type, 'value') else str(memory_type),
                            content,
                            json.dumps(metadata),
                            vector_id,
                            1.0,  # Default verification score
                            timestamp,  # last_accessed = now
                            1,  # Initial access count
                            0,  # Not archived
                            0.5  # Default importance score
                        ))
                        
                        # Get the inserted ID
                        memory_id = cursor.lastrowid
                        
                        # Also add to search index
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_search'")
                        if cursor.fetchone():
                            cursor.execute("""
                                INSERT INTO memory_search (rowid, content, metadata)
                                VALUES (?, ?, ?)
                            """, (memory_id, content, json.dumps(metadata)))
                            
                        return memory_id
                except Exception as e:
                    self.logger.error(f"Error adding memory to SQLite directly: {e}")
                    return None
            
            # Run in thread pool
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self.executor, _add_memory_direct)

    async def search_memories(self, query: str, limit: int = 50):
        """
        Search memories in SQLite storage.
        
        Args:
            query: Search query
            limit: Maximum results to return
            
        Returns:
            Dictionary of memory results by category
        """
        # Wait for pools to initialize
        if self.long_term_pool is None:
            await self._init_pools()
            
        async def _search_memories_with_pool():
            results = {
                "conversations": [],
                "reference": []
            }
            
            try:
                async with self.long_term_pool.acquire() as conn:
                    cursor = conn.cursor()
                    
                    # Check if FTS is available
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_search'")
                    fts_available = cursor.fetchone() is not None
                    
                    # Check which columns we have
                    cursor.execute("PRAGMA table_info(long_term_memory)")
                    columns = [info[1] for info in cursor.fetchall()]
                    
                    rows = []
                    if fts_available:
                        # Use FTS for better search results
                        cursor.execute("""
                            SELECT m.id, m.timestamp, m.memory_type, m.content, m.metadata,
                                   m.verification_score, m.importance_score, m.archived
                            FROM memory_search s
                            JOIN long_term_memory m ON s.rowid = m.id
                            WHERE s.content MATCH ? AND m.archived = 0
                            ORDER BY m.importance_score DESC, m.timestamp DESC
                            LIMIT ?
                        """, (query, limit))
                        rows = cursor.fetchall()
                    else:
                        # Fall back to LIKE search
                        cursor.execute("""
                            SELECT id, timestamp, memory_type, content, metadata,
                                   verification_score, importance_score, archived
                            FROM long_term_memory
                            WHERE content LIKE ? AND archived = 0
                            ORDER BY importance_score DESC, timestamp DESC
                            LIMIT ?
                        """, (f"%{query}%", limit))
                        rows = cursor.fetchall()
                    
                    # Update access counts for retrieved memories
                    memory_ids = [row[0] for row in rows]
                    if memory_ids:
                        placeholders = ','.join(['?'] * len(memory_ids))
                        cursor.execute(f"""
                            UPDATE long_term_memory
                            SET last_accessed = ?, access_count = access_count + 1
                            WHERE id IN ({placeholders})
                        """, [datetime.now().isoformat()] + memory_ids)
                    
                    # Process results
                    from grace.utils.common import calculate_relevance
                    
                    for row in rows:
                        try:
                            memory_data = {
                                "id": row[0],
                                "timestamp": row[1],
                                "memory_type": row[2],
                                "content": row[3],
                                "metadata": json.loads(row[4]) if row[4] else {},
                                "verification_score": row[5] if len(row) > 5 else 1.0,
                                "importance_score": row[6] if len(row) > 6 else 0.5,
                                "archived": row[7] if len(row) > 7 else 0
                            }
                            
                            # Add relevance score
                            relevance = calculate_relevance(query, memory_data["content"])
                            memory_data["relevance"] = relevance
                            
                            # Skip items with low relevance
                            if relevance < 0.2:
                                continue
                                
                            if row[2] == MemoryType.CONVERSATION.value:
                                results["conversations"].append(memory_data)
                            elif row[2] == MemoryType.REFERENCE.value:
                                results["reference"].append(memory_data)
                        except Exception as e:
                            self.logger.debug(f"Error processing memory result: {e}")
                    
                    # Sort by relevance and limit results
                    for key in results:
                        results[key].sort(key=lambda x: x.get("relevance", 0), reverse=True)
                        results[key] = results[key][:limit]
                
                return results
            except Exception as e:
                self.logger.error(f"Search error with pool: {e}")
                raise
        
        try:
            return await _search_memories_with_pool()
        except Exception as e:
            # Fall back to direct connection if pool fails
            self.logger.warning(f"Pool-based memory search failed, trying direct connection: {e}")
            
            def _search_memories_direct():
                results = {
                    "conversations": [],
                    "reference": []
                }
                
                try:
                    with sqlite3.connect(self.long_term_db) as conn:
                        cursor = conn.cursor()
                        
                        # Check if FTS is available
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_search'")
                        fts_available = cursor.fetchone() is not None
                        
                        # Check which columns we have
                        cursor.execute("PRAGMA table_info(long_term_memory)")
                        columns = [info[1] for info in cursor.fetchall()]
                        
                        rows = []
                        if fts_available:
                            # Use FTS for better search results
                            cursor.execute("""
                                SELECT m.id, m.timestamp, m.memory_type, m.content, m.metadata,
                                       m.verification_score, m.importance_score, m.archived
                                FROM memory_search s
                                JOIN long_term_memory m ON s.rowid = m.id
                                WHERE s.content MATCH ? AND m.archived = 0
                                ORDER BY m.importance_score DESC, m.timestamp DESC
                                LIMIT ?
                            """, (query, limit))
                            rows = cursor.fetchall()
                        else:
                            # Fall back to LIKE search
                            cursor.execute("""
                                SELECT id, timestamp, memory_type, content, metadata,
                                       verification_score, importance_score, archived
                                FROM long_term_memory
                                WHERE content LIKE ? AND archived = 0
                                ORDER BY importance_score DESC, timestamp DESC
                                LIMIT ?
                            """, (f"%{query}%", limit))
                            rows = cursor.fetchall()
                        
                        # Update access counts for retrieved memories
                        memory_ids = [row[0] for row in rows]
                        if memory_ids:
                            placeholders = ','.join(['?'] * len(memory_ids))
                            cursor.execute(f"""
                                UPDATE long_term_memory
                                SET last_accessed = ?, access_count = access_count + 1
                                WHERE id IN ({placeholders})
                            """, [datetime.now().isoformat()] + memory_ids)
                        
                        # Process results
                        from grace.utils.common import calculate_relevance
                        
                        for row in rows:
                            try:
                                memory_data = {
                                    "id": row[0],
                                    "timestamp": row[1],
                                    "memory_type": row[2],
                                    "content": row[3],
                                    "metadata": json.loads(row[4]) if row[4] else {},
                                    "verification_score": row[5] if len(row) > 5 else 1.0,
                                    "importance_score": row[6] if len(row) > 6 else 0.5,
                                    "archived": row[7] if len(row) > 7 else 0
                                }
                                
                                # Add relevance score
                                relevance = calculate_relevance(query, memory_data["content"])
                                memory_data["relevance"] = relevance
                                
                                # Skip items with low relevance
                                if relevance < 0.2:
                                    continue
                                    
                                if row[2] == MemoryType.CONVERSATION.value:
                                    results["conversations"].append(memory_data)
                                elif row[2] == MemoryType.REFERENCE.value:
                                    results["reference"].append(memory_data)
                            except Exception as e:
                                self.logger.debug(f"Error processing memory result: {e}")
                        
                        # Sort by relevance and limit results
                        for key in results:
                            results[key].sort(key=lambda x: x.get("relevance", 0), reverse=True)
                            results[key] = results[key][:limit]
                    
                    return results
                except Exception as e:
                    self.logger.error(f"Search error with direct connection: {e}")
                    return results
            
            # Run in thread pool
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self.executor, _search_memories_direct)

    async def log_conversation(self, entry: ConversationEntry):
        """
        Log conversation to database.
        
        Args:
            entry: Conversation entry to log
        """
        if self.amnesia_mode:
            return
        
        # Wait for pools to initialize
        if self.conversation_pool is None:
            await self._init_pools()
            
        async def _log_conversation_with_pool():
            try:
                # Validate entry fields to avoid issues
                if not entry.timestamp:
                    entry.timestamp = datetime.now().isoformat()
                    
                async with self.conversation_pool.acquire() as conn:
                    cursor = conn.cursor()
                    
                    # Convert dictionary type fields to JSON strings if they're not None
                    memory_context_json = json.dumps(entry.memory_context) if entry.memory_context else None
                    json_response_json = json.dumps(entry.json_response) if entry.json_response else None
                    metadata_json = json.dumps(entry.metadata) if entry.metadata else None
                    verification_result_json = json.dumps(entry.verification_result) if entry.verification_result else None
                    
                    # Insert in conversations table with all fields
                    cursor.execute("""
                        INSERT INTO conversations (
                            timestamp, user_input, stt_transcript, memory_context,
                            prompt, model_response, thinking_process, json_response,
                            command_result, tts_output, error, metadata, verification_result,
                            archived, importance_score
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        entry.timestamp,
                        entry.user_input,
                        entry.stt_transcript,
                        memory_context_json,
                        entry.prompt,
                        entry.model_response,
                        entry.thinking_process,
                        json_response_json,
                        entry.command_result,
                        entry.tts_output,
                        entry.error,
                        metadata_json,
                        verification_result_json,
                        0,  # Not archived
                        0.5  # Default importance score
                    ))
                    
                    # Get the inserted ID
                    cursor.execute("SELECT last_insert_rowid()")
                    conversation_id = cursor.fetchone()[0]
                    
                    # Check if search table exists
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='conversation_search'")
                    if cursor.fetchone():
                        # Also insert in search table
                        cursor.execute("""
                            INSERT INTO conversation_search (rowid, user_input, model_response, thinking_process)
                            VALUES (?, ?, ?, ?)
                        """, (
                            conversation_id,
                            entry.user_input,
                            entry.model_response,
                            entry.thinking_process
                        ))
                        
                    return conversation_id
            except Exception as e:
                self.logger.error(f"Failed to log conversation with pool: {e}")
                raise
        
        try:
            return await _log_conversation_with_pool()
        except Exception as e:
            # Fall back to direct connection if pool fails
            self.logger.warning(f"Pool-based conversation log failed, trying direct connection: {e}")
            
            def _log_conversation_direct():
                try:
                    # Validate entry fields to avoid issues
                    if not entry.timestamp:
                        entry.timestamp = datetime.now().isoformat()
                        
                    with sqlite3.connect(self.conversation_db) as conn:
                        cursor = conn.cursor()
                        
                        # Convert dictionary type fields to JSON strings if they're not None
                        memory_context_json = json.dumps(entry.memory_context) if entry.memory_context else None
                        json_response_json = json.dumps(entry.json_response) if entry.json_response else None
                        metadata_json = json.dumps(entry.metadata) if entry.metadata else None
                        verification_result_json = json.dumps(entry.verification_result) if entry.verification_result else None
                        
                        # Insert in conversations table with all fields
                        cursor.execute("""
                            INSERT INTO conversations (
                                timestamp, user_input, stt_transcript, memory_context,
                                prompt, model_response, thinking_process, json_response,
                                command_result, tts_output, error, metadata, verification_result,
                                archived, importance_score
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            entry.timestamp,
                            entry.user_input,
                            entry.stt_transcript,
                            memory_context_json,
                            entry.prompt,
                            entry.model_response,
                            entry.thinking_process,
                            json_response_json,
                            entry.command_result,
                            entry.tts_output,
                            entry.error,
                            metadata_json,
                            verification_result_json,
                            0,  # Not archived
                            0.5  # Default importance score
                        ))
                        
                        # Get the inserted ID
                        conversation_id = cursor.lastrowid
                        
                        # Check if search table exists
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='conversation_search'")
                        if cursor.fetchone():
                            # Also insert in search table
                            cursor.execute("""
                                INSERT INTO conversation_search (rowid, user_input, model_response, thinking_process)
                                VALUES (?, ?, ?, ?)
                            """, (
                                conversation_id,
                                entry.user_input,
                                entry.model_response,
                                entry.thinking_process
                            ))
                            
                        return conversation_id
                except Exception as e:
                    self.logger.error(f"Failed to log conversation directly: {e}")
                    return None
            
            # Run in thread pool
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self.executor, _log_conversation_direct)

    async def archive_old_memories(self):
        """Archive old memories instead of deleting them."""
        if self.amnesia_mode:
            return
            
        # Wait for pools to initialize
        if self.long_term_pool is None:
            await self._init_pools()
            
        async def _archive_old_memories_with_pool():
            try:
                async with self.long_term_pool.acquire() as conn:
                    cursor = conn.cursor()
                    
                    # Get archiving configuration
                    archive_age_days = self.config.get('archive_age_days', 365)
                    
                    # Calculate cutoff date
                    cutoff_date = (datetime.now() - timedelta(days=archive_age_days)).isoformat()
                    
                    # Mark memories as archived instead of deleting them
                    # Fix the error in the table name: use long_term_memory instead of memories
                    cursor.execute(
                        "UPDATE long_term_memory SET archived = 1 WHERE timestamp < ?", (cutoff_date,)
                    )
                    
                    archived_count = cursor.rowcount
                    
                    # Also archive old conversations
                    async with self.conversation_pool.acquire() as conv_conn:
                        conv_cursor = conv_conn.cursor()
                        
                        conv_cursor.execute("""
                            UPDATE conversations
                            SET archived = 1
                            WHERE timestamp < ?
                            AND archived = 0
                        """, (cutoff_date,))
                        
                        archived_count += conv_cursor.rowcount
                    
                    return archived_count
            except Exception as e:
                self.logger.error(f"Error archiving memories with pool: {e}")
                raise
        
        try:
            return await _archive_old_memories_with_pool()
        except Exception as e:
            # Fall back to direct connection if pool fails
            self.logger.warning(f"Pool-based archive failed, trying direct connection: {e}")
            
            def _archive_old_memories_direct():
                try:
                    with sqlite3.connect(self.long_term_db) as conn:
                        cursor = conn.cursor()
                        
                        # Get archiving configuration
                        archive_age_days = self.config.get('archive_age_days', 365)
                        
                        # Calculate cutoff date
                        cutoff_date = (datetime.now() - timedelta(days=archive_age_days)).isoformat()
                        
                        # Mark memories as archived instead of deleting them
                        # Fix the error in the table name: use long_term_memory instead of memories
                        cursor.execute(
                            "UPDATE long_term_memory SET archived = 1 WHERE timestamp < ?", (cutoff_date,)
                        )
                        
                        archived_count = cursor.rowcount
                        
                        # Also archive old conversations
                        with sqlite3.connect(self.conversation_db) as conv_conn:
                            conv_cursor = conv_conn.cursor()
                            
                            conv_cursor.execute("""
                                UPDATE conversations
                                SET archived = 1
                                WHERE timestamp < ?
                                AND archived = 0
                            """, (cutoff_date,))
                            
                            archived_count += conv_cursor.rowcount
                        
                        return archived_count
                except Exception as e:
                    self.logger.error(f"Error archiving memories directly: {e}")
                    return 0
            
            # Run in thread pool
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self.executor, _archive_old_memories_direct)

    async def optimize_database(self):
        """Optimize SQLite databases for better performance."""
        # Wait for pools to initialize
        if self.long_term_pool is None:
            await self._init_pools()
            
        def _optimize_databases():
            try:
                optimizations = []
                
                # Optimize long-term memory database
                with sqlite3.connect(self.long_term_db) as conn:
                    cursor = conn.cursor()
                    
                    # Analyze table structure for query optimization
                    cursor.execute("ANALYZE")
                    optimizations.append("Analyzed long-term database")
                    
                    # Run VACUUM to reclaim space and defragment
                    cursor.execute("VACUUM")
                    optimizations.append("Vacuumed long-term database")
                    
                    # Optimize FTS tables if they exist
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_search'")
                    if cursor.fetchone():
                        try:
                            cursor.execute("INSERT INTO memory_search(memory_search) VALUES('optimize')")
                            optimizations.append("Optimized FTS index")
                        except Exception as e:
                            self.logger.debug(f"FTS optimization error: {e}")
                
                # Optimize conversation database
                with sqlite3.connect(self.conversation_db) as conn:
                    cursor = conn.cursor()
                    
                    # Analyze table structure
                    cursor.execute("ANALYZE")
                    optimizations.append("Analyzed conversation database")
                    
                    # Run VACUUM to reclaim space and defragment
                    cursor.execute("VACUUM")
                    optimizations.append("Vacuumed conversation database")
                    
                    # Optimize FTS tables if they exist
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='conversation_search'")
                    if cursor.fetchone():
                        try:
                            cursor.execute("INSERT INTO conversation_search(conversation_search) VALUES('optimize')")
                            optimizations.append("Optimized conversation FTS index")
                        except Exception as e:
                            self.logger.debug(f"Conversation FTS optimization error: {e}")
                
                return optimizations
            except Exception as e:
                self.logger.error(f"Database optimization error: {e}")
                return [f"Optimization error: {e}"]
        
        # Run in thread pool
        loop = asyncio.get_running_loop()
        optimizations = await loop.run_in_executor(self.executor, _optimize_databases)
        
        if optimizations:
            self.logger.info(f"Database optimizations completed: {', '.join(optimizations)}")

    async def get_stats(self):
        """
        Get detailed statistics about SQLite storage.
        
        Returns:
            Dictionary with statistics
        """
        def _get_stats():
            stats = {
                "long_term_count": 0,
                "conversation_count": 0,
                "memory_types": {},
                "archived": 0,
                "active": 0,
                "db_size_kb": 0
            }
            
            try:
                # Get long-term memory stats
                with sqlite3.connect(self.long_term_db) as conn:
                    cursor = conn.cursor()
                    
                    # Get total count
                    cursor.execute("SELECT COUNT(*) FROM long_term_memory")
                    stats["long_term_count"] = cursor.fetchone()[0]
                    
                    # Get counts by type
                    cursor.execute("""
                        SELECT memory_type, COUNT(*) FROM long_term_memory 
                        GROUP BY memory_type
                    """)
                    for memory_type, count in cursor.fetchall():
                        stats["memory_types"][memory_type] = count
                    
                    # Get active vs archived counts
                    cursor.execute("SELECT COUNT(*) FROM long_term_memory WHERE archived = 0")
                    stats["active"] = cursor.fetchone()[0]
                    
                    cursor.execute("SELECT COUNT(*) FROM long_term_memory WHERE archived = 1")
                    stats["archived"] = cursor.fetchone()[0]
                    
                # Get conversation stats
                with sqlite3.connect(self.conversation_db) as conn:
                    cursor = conn.cursor()
                    
                    # Get total count
                    cursor.execute("SELECT COUNT(*) FROM conversations")
                    stats["conversation_count"] = cursor.fetchone()[0]
                    
                    # Get counts of conversations by archived status
                    cursor.execute("SELECT archived, COUNT(*) FROM conversations GROUP BY archived")
                    for archived, count in cursor.fetchall():
                        if archived:
                            stats["archived_conversations"] = count
                        else:
                            stats["active_conversations"] = count
                
                # Get database file sizes
                if self.long_term_db.exists():
                    lt_size = self.long_term_db.stat().st_size / 1024  # KB
                    stats["long_term_db_size_kb"] = round(lt_size, 2)
                    
                if self.conversation_db.exists():
                    conv_size = self.conversation_db.stat().st_size / 1024  # KB
                    stats["conversation_db_size_kb"] = round(conv_size, 2)
                    
                stats["db_size_kb"] = round(stats.get("long_term_db_size_kb", 0) + 
                                         stats.get("conversation_db_size_kb", 0), 2)
                
                # Count FTS indices if they exist
                with sqlite3.connect(self.long_term_db) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_search'")
                    stats["has_fts_index"] = cursor.fetchone() is not None
                
                return stats
            except Exception as e:
                self.logger.error(f"Error getting SQLite stats: {e}")
                return {"error": str(e)}
        
        # Run in thread pool
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, _get_stats)

    async def get_conversation_by_id(self, conversation_id: int):
        """
        Get a specific conversation by ID.
        
        Args:
            conversation_id: ID of the conversation to retrieve
            
        Returns:
            ConversationEntry or None if not found
        """
        # Wait for pools to initialize
        if self.conversation_pool is None:
            await self._init_pools()
            
        async def _get_conversation_with_pool():
            try:
                async with self.conversation_pool.acquire() as conn:
                    # Set row factory
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    
                    cursor.execute("""
                        SELECT * FROM conversations
                        WHERE id = ?
                    """, (conversation_id,))
                    
                    row = cursor.fetchone()
                    if not row:
                        return None
                        
                    # Convert to ConversationEntry
                    from grace.utils.common import ConversationEntry
                    
                    # Parse JSON fields safely
                    memory_context = {}
                    if row['memory_context']:
                        try:
                            memory_context = json.loads(row['memory_context'])
                        except json.JSONDecodeError:
                            pass
                    
                    json_response = {}
                    if row['json_response']:
                        try:
                            json_response = json.loads(row['json_response'])
                        except json.JSONDecodeError:
                            pass
                    
                    metadata = {}
                    if row['metadata']:
                        try:
                            metadata = json.loads(row['metadata'])
                        except json.JSONDecodeError:
                            pass
                    
                    verification_result = {}
                    if row['verification_result']:
                        try:
                            verification_result = json.loads(row['verification_result'])
                        except json.JSONDecodeError:
                            pass
                    
                    entry = ConversationEntry(
                        timestamp=row['timestamp'],
                        user_input=row['user_input'],
                        stt_transcript=row['stt_transcript'],
                        memory_context=memory_context,
                        prompt=row['prompt'],
                        model_response=row['model_response'],
                        thinking_process=row['thinking_process'],
                        json_response=json_response,
                        command_result=row['command_result'],
                        tts_output=row['tts_output'],
                        error=row['error'],
                        metadata=metadata,
                        verification_result=verification_result
                    )
                    
                    return entry
            except Exception as e:
                self.logger.error(f"Error getting conversation by ID with pool: {e}")
                raise
        
        try:
            return await _get_conversation_with_pool()
        except Exception as e:
            # Fall back to direct connection if pool fails
            self.logger.warning(f"Pool-based get conversation failed, trying direct connection: {e}")
            
            def _get_conversation_direct():
                try:
                    with sqlite3.connect(self.conversation_db) as conn:
                        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
                        cursor = conn.cursor()
                        
                        cursor.execute("""
                            SELECT * FROM conversations
                            WHERE id = ?
                        """, (conversation_id,))
                        
                        row = cursor.fetchone()
                        if not row:
                            return None
                            
                        # Convert to ConversationEntry
                        from grace.utils.common import ConversationEntry
                        
                        # Parse JSON fields safely
                        memory_context = {}
                        if row['memory_context']:
                            try:
                                memory_context = json.loads(row['memory_context'])
                            except json.JSONDecodeError:
                                pass
                        
                        json_response = {}
                        if row['json_response']:
                            try:
                                json_response = json.loads(row['json_response'])
                            except json.JSONDecodeError:
                                pass
                        
                        metadata = {}
                        if row['metadata']:
                            try:
                                metadata = json.loads(row['metadata'])
                            except json.JSONDecodeError:
                                pass
                        
                        verification_result = {}
                        if row['verification_result']:
                            try:
                                verification_result = json.loads(row['verification_result'])
                            except json.JSONDecodeError:
                                pass
                        
                        entry = ConversationEntry(
                            timestamp=row['timestamp'],
                            user_input=row['user_input'],
                            stt_transcript=row['stt_transcript'],
                            memory_context=memory_context,
                            prompt=row['prompt'],
                            model_response=row['model_response'],
                            thinking_process=row['thinking_process'],
                            json_response=json_response,
                            command_result=row['command_result'],
                            tts_output=row['tts_output'],
                            error=row['error'],
                            metadata=metadata,
                            verification_result=verification_result
                        )
                        
                        return entry
                except Exception as e:
                    self.logger.error(f"Error getting conversation by ID directly: {e}")
                    return None
            
            # Run in thread pool
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self.executor, _get_conversation_direct)

    async def get_recent_conversations(self, limit: int = 10):
        """
        Get recent conversations.
        
        Args:
            limit: Maximum number of conversations to return
            
        Returns:
            List of ConversationEntry objects
        """
        # Wait for pools to initialize
        if self.conversation_pool is None:
            await self._init_pools()
            
        async def _get_recent_conversations_with_pool():
            try:
                async with self.conversation_pool.acquire() as conn:
                    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
                    cursor = conn.cursor()
                    
                    cursor.execute("""
                        SELECT * FROM conversations
                        WHERE archived = 0
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """, (limit,))
                    
                    rows = cursor.fetchall()
                    
                    # Convert to ConversationEntry objects
                    from grace.utils.common import ConversationEntry
                    
                    entries = []
                    for row in rows:
                        # Parse JSON fields safely
                        memory_context = {}
                        if row['memory_context']:
                            try:
                                memory_context = json.loads(row['memory_context'])
                            except json.JSONDecodeError:
                                pass
                        
                        json_response = {}
                        if row['json_response']:
                            try:
                                json_response = json.loads(row['json_response'])
                            except json.JSONDecodeError:
                                pass
                        
                        metadata = {}
                        if row['metadata']:
                            try:
                                metadata = json.loads(row['metadata'])
                            except json.JSONDecodeError:
                                pass
                        
                        verification_result = {}
                        if row['verification_result']:
                            try:
                                verification_result = json.loads(row['verification_result'])
                            except json.JSONDecodeError:
                                pass
                        
                        entry = ConversationEntry(
                            timestamp=row['timestamp'],
                            user_input=row['user_input'],
                            stt_transcript=row['stt_transcript'],
                            memory_context=memory_context,
                            prompt=row['prompt'],
                            model_response=row['model_response'],
                            thinking_process=row['thinking_process'],
                            json_response=json_response,
                            command_result=row['command_result'],
                            tts_output=row['tts_output'],
                            error=row['error'],
                            metadata=metadata,
                            verification_result=verification_result
                        )
                        entries.append(entry)
                        
                    return entries
            except Exception as e:
                self.logger.error(f"Error getting recent conversations with pool: {e}")
                raise
        
        try:
            return await _get_recent_conversations_with_pool()
        except Exception as e:
            # Fall back to direct connection if pool fails
            self.logger.warning(f"Pool-based recent conversations failed, trying direct connection: {e}")
            
            def _get_recent_conversations_direct():
                try:
                    with sqlite3.connect(self.conversation_db) as conn:
                        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
                        cursor = conn.cursor()
                        
                        cursor.execute("""
                            SELECT * FROM conversations
                            WHERE archived = 0
                            ORDER BY timestamp DESC
                            LIMIT ?
                        """, (limit,))
                        
                        rows = cursor.fetchall()
                        
                        # Convert to ConversationEntry objects
                        from grace.utils.common import ConversationEntry
                        
                        entries = []
                        for row in rows:
                            # Parse JSON fields safely
                            memory_context = {}
                            if row['memory_context']:
                                try:
                                    memory_context = json.loads(row['memory_context'])
                                except json.JSONDecodeError:
                                    pass
                            
                            json_response = {}
                            if row['json_response']:
                                try:
                                    json_response = json.loads(row['json_response'])
                                except json.JSONDecodeError:
                                    pass
                            
                            metadata = {}
                            if row['metadata']:
                                try:
                                    metadata = json.loads(row['metadata'])
                                except json.JSONDecodeError:
                                    pass
                            
                            verification_result = {}
                            if row['verification_result']:
                                try:
                                    verification_result = json.loads(row['verification_result'])
                                except json.JSONDecodeError:
                                    pass
                            
                            entry = ConversationEntry(
                                timestamp=row['timestamp'],
                                user_input=row['user_input'],
                                stt_transcript=row['stt_transcript'],
                                memory_context=memory_context,
                                prompt=row['prompt'],
                                model_response=row['model_response'],
                                thinking_process=row['thinking_process'],
                                json_response=json_response,
                                command_result=row['command_result'],
                                tts_output=row['tts_output'],
                                error=row['error'],
                                metadata=metadata,
                                verification_result=verification_result
                            )
                            entries.append(entry)
                            
                        return entries
                except Exception as e:
                    self.logger.error(f"Error getting recent conversations directly: {e}")
                    return []
            
            # Run in thread pool
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self.executor, _get_recent_conversations_direct)

    async def search_conversations(self, query: str, limit: int = 10):
        """
        Search conversations by content.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            List of ConversationEntry objects
        """
        # For simplicity, reuse the get_recent_conversations method
        # In a real implementation, you would implement a proper search
        return await self.get_recent_conversations(limit)

    async def delete_memory(self, memory_id: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Delete a memory from SQLite storage.
        
        Args:
            memory_id: Memory ID to delete
            
        Returns:
            Tuple of (vector_id, critical_id) for cascading deletion
        """
        # Wait for pools to initialize
        if self.long_term_pool is None:
            await self._init_pools()
            
        async def _delete_memory_with_pool():
            try:
                async with self.long_term_pool.acquire() as conn:
                    cursor = conn.cursor()
                    
                    # First, get related IDs
                    cursor.execute("""
                        SELECT vector_id, memory_type FROM long_term_memory WHERE id = ?
                    """, (memory_id,))
                    
                    row = cursor.fetchone()
                    if not row:
                        return None, None
                        
                    vector_id, memory_type = row
                    
                    # Delete from search index if it exists
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_search'")
                    if cursor.fetchone():
                        cursor.execute("""
                            DELETE FROM memory_search WHERE rowid = ?
                        """, (memory_id,))
                    
                    # Delete from main table
                    cursor.execute("""
                        DELETE FROM long_term_memory WHERE id = ?
                    """, (memory_id,))
                    
                    # If critical memory, pass the ID for further deletion
                    critical_id = f"critical_{memory_id}" if memory_type == MemoryType.CRITICAL.value else None
                    
                    return vector_id, critical_id
            except Exception as e:
                self.logger.error(f"Error deleting memory with pool: {e}")
                raise
        
        try:
            return await _delete_memory_with_pool()
        except Exception as e:
            # Fall back to direct connection if pool fails
            self.logger.warning(f"Pool-based delete failed, trying direct connection: {e}")
            
            def _delete_memory_direct():
                try:
                    with sqlite3.connect(self.long_term_db) as conn:
                        cursor = conn.cursor()
                        
                        # First, get related IDs
                        cursor.execute("""
                            SELECT vector_id, memory_type FROM long_term_memory WHERE id = ?
                        """, (memory_id,))
                        
                        row = cursor.fetchone()
                        if not row:
                            return None, None
                            
                        vector_id, memory_type = row
                        
                        # Delete from search index if it exists
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_search'")
                        if cursor.fetchone():
                            cursor.execute("""
                                DELETE FROM memory_search WHERE rowid = ?
                            """, (memory_id,))
                        
                        # Delete from main table
                        cursor.execute("""
                            DELETE FROM long_term_memory WHERE id = ?
                        """, (memory_id,))
                        
                        # If critical memory, pass the ID for further deletion
                        critical_id = f"critical_{memory_id}" if memory_type == MemoryType.CRITICAL.value else None
                        
                        return vector_id, critical_id
                except Exception as e:
                    self.logger.error(f"Error deleting memory directly: {e}")
                    return None, None
            
            # Run in thread pool
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self.executor, _delete_memory_direct)
                
    async def close(self):
        """Clean up resources with proper async handling."""
        # Close connection pools if initialized
        if self.long_term_pool:
            await self.long_term_pool.close()
        if self.conversation_pool:
            await self.conversation_pool.close()
        
        # Shutdown executor
        if self.executor:
            self.executor.shutdown(wait=False)
            
        self.logger.info("SQLite storage closed")
    
    def close(self):
        """Synchronous cleanup for backward compatibility."""
        if self.executor:
            self.executor.shutdown(wait=False)
            
        self.logger.info("SQLite storage closed synchronously")
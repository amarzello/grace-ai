"""
Grace AI System - Critical Memory Manager

This module handles critical memory management for the Grace AI system.
"""

import os
import json
import logging
import asyncio
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import threading

from grace.utils.common import MEMORY_DB_PATH, calculate_relevance
import hashlib


class CriticalMemoryManager:
    """Manager for critical memories that must be preserved."""
    
    def __init__(self, config: Dict):
        """
        Initialize critical memory manager with the provided configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger('grace.memory.critical')
        self.config = config.get('memory', {})
        self.amnesia_mode = config.get('amnesia_mode', False)
        
        # Critical memory is stored in a JSON file for security (replaces pickle)
        self.critical_memory_file = MEMORY_DB_PATH / 'critical_memory.json'
        
        # Lock for thread-safe access with RLock to prevent deadlocks
        self.critical_memory_lock = threading.RLock()
        
        # Thread pool for async operations with limited workers
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Load critical memory
        self.critical_memory = {}
        self._load_critical_memory()
        
        # Create backup schedule
        self.backup_interval = self.config.get('critical_backup_interval_hours', 24)
        self.last_backup_time = datetime.now()
        
    def _load_critical_memory(self):
        """Load critical memory from disk with error handling."""
        try:
            if self.critical_memory_file.exists():
                try:
                    with open(self.critical_memory_file, 'r', encoding='utf-8') as f:
                        self.critical_memory = json.load(f)
                        
                    self.logger.info(f"Loaded {len(self.critical_memory)} critical memories")
                except (json.JSONDecodeError, EOFError) as e:
                    self.logger.error(f"Error loading critical memory: {e}")
                    
                    # Create backup of corrupted file
                    if self.critical_memory_file.exists():
                        backup_path = self.critical_memory_file.with_suffix('.json.corrupted')
                        try:
                            import shutil
                            shutil.copy(self.critical_memory_file, backup_path)
                            self.logger.info(f"Created backup of corrupted critical memory at {backup_path}")
                            
                            # Try to recover partial content
                            self._try_recovery(self.critical_memory_file)
                        except Exception as be:
                            self.logger.error(f"Failed to create backup: {be}")
                    
                    # Reset to empty memory if recovery failed
                    if not self.critical_memory:
                        self.critical_memory = {}
            else:
                # If no critical memory file exists, initialize it as empty
                self.critical_memory = {}
                
                # Save the empty critical memory file
                asyncio.create_task(self._save_critical_memory())
                
            self.logger.info("Critical memory initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize critical memory: {e}")
            self.critical_memory = {}
            
    def _try_recovery(self, file_path: Path):
        """
        Try to recover partial content from a corrupted JSON file.
        
        Args:
            file_path: Path to the corrupted file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find valid JSON objects
            recovered_memories = {}
            start_pos = 0
            
            while True:
                # Find next object start
                obj_start = content.find('{', start_pos)
                if obj_start == -1:
                    break
                
                # Try to parse from this position
                for end_pos in range(obj_start + 1, len(content)):
                    try:
                        # Try to parse a complete object
                        obj_str = content[obj_start:end_pos + 1]
                        obj = json.loads(obj_str)
                        
                        # If successful, extract key-value pairs
                        if isinstance(obj, dict):
                            for key, value in obj.items():
                                if isinstance(value, dict) and 'content' in value:
                                    recovered_memories[key] = value
                        
                        # Move to next position
                        start_pos = end_pos + 1
                        break
                    except json.JSONDecodeError:
                        # Not a complete object yet, continue
                        continue
                
                # Move forward if we couldn't parse anything
                start_pos = obj_start + 1
            
            if recovered_memories:
                self.logger.info(f"Recovered {len(recovered_memories)} critical memories")
                self.critical_memory = recovered_memories
                
                # Save recovered memories
                asyncio.create_task(self._save_critical_memory())
            else:
                self.logger.warning("Could not recover any critical memories")
                
        except Exception as e:
            self.logger.error(f"Recovery attempt failed: {e}")
            
    async def _save_critical_memory(self):
        """Save critical memory to disk with improved error handling and locking."""
        if self.amnesia_mode:
            return
            
        # Use lock for thread safety
        with self.critical_memory_lock:
            try:
                # Create a temporary file first to avoid corruption
                with tempfile.NamedTemporaryFile('w', suffix='.json.tmp', delete=False, encoding='utf-8') as f:
                    temp_file = Path(f.name)
                    json.dump(self.critical_memory, f, indent=2)
                
                # Create parent directory if it doesn't exist
                self.critical_memory_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Replace the original file with the temporary file (atomic operation)
                os.replace(temp_file, self.critical_memory_file)
                
                self.logger.debug(f"Saved {len(self.critical_memory)} critical memories")
                
                # Check if it's time for a backup
                now = datetime.now()
                hours_since_backup = (now - self.last_backup_time).total_seconds() / 3600
                
                if hours_since_backup >= self.backup_interval:
                    await self._create_backup()
                    self.last_backup_time = now
            except Exception as e:
                self.logger.error(f"Error saving critical memory: {e}")
                
                # Try to clean up temp file if it exists
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except Exception:
                    pass
    
    async def _create_backup(self):
        """Create a backup of the critical memory file."""
        if not self.critical_memory_file.exists():
            return
            
        try:
            # Create backup directory if it doesn't exist
            backup_dir = MEMORY_DB_PATH / 'backups'
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Create backup with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = backup_dir / f"critical_memory_{timestamp}.json"
            
            # Copy the file
            import shutil
            shutil.copy2(self.critical_memory_file, backup_path)
            
            self.logger.info(f"Created critical memory backup at {backup_path}")
            
            # Remove old backups (keep last 5)
            backups = sorted(backup_dir.glob('critical_memory_*.json'))
            if len(backups) > 5:
                for old_backup in backups[:-5]:
                    old_backup.unlink()
                    self.logger.debug(f"Removed old backup: {old_backup}")
        except Exception as e:
            self.logger.error(f"Error creating backup: {e}")
                    
    async def add_memory(self, content: str, metadata: Dict = None, sql_id: int = None) -> str:
        """
        Add critical memory.
        
        Args:
            content: Memory content
            metadata: Additional metadata
            sql_id: SQLite storage ID
            
        Returns:
            Memory identifier
        """
        if self.amnesia_mode:
            return None
            
        if not content or not content.strip():
            return None
            
        timestamp = datetime.now().isoformat()
        metadata = metadata or {}
        metadata['timestamp'] = timestamp
        
        # Generate a unique key for this memory
        if sql_id:
            key = f"critical_{sql_id}"
        else:
            # Create a deterministic but unique key based on content and timestamp
            hash_input = (content + timestamp).encode('utf-8')
            key = f"critical_{hashlib.md5(hash_input).hexdigest()}"
        
        # Store memory data
        memory_data = {
            'content': content,
            'metadata': metadata,
            'sql_id': sql_id,
            'verification_score': 1.0,
            'created_at': timestamp,
            'accessed_at': timestamp,
            'access_count': 0
        }
        
        # Update critical memory dictionary with thread safety
        with self.critical_memory_lock:
            self.critical_memory[key] = memory_data
            
        # Save to disk (non-blocking)
        save_task = asyncio.create_task(
            self._save_critical_memory(),
            name=f"save_critical_memory_{timestamp}"
        )
        
        return key
        
    async def search_memories(self, query: str, limit: int = 50) -> List[Dict]:
        """
        Search critical memories.
        
        Args:
            query: Search query
            limit: Maximum results to return
            
        Returns:
            List of matching memories
        """
        results = []
        
        # Run in executor to avoid blocking
        loop = asyncio.get_running_loop()
        
        def _search_memories():
            search_results = []
            
            # Search through critical memories with thread safety
            with self.critical_memory_lock:
                for key, memory in self.critical_memory.items():
                    try:
                        content = memory.get('content', '')
                        if content:
                            # Calculate relevance score
                            relevance = calculate_relevance(query, content)
                            
                            # Only include if relevance is above threshold
                            if relevance > 0.2:
                                # Create a result entry
                                result = {
                                    'id': key,
                                    'content': content,
                                    'metadata': memory.get('metadata', {}),
                                    'verification_score': memory.get('verification_score', 1.0),
                                    'relevance': relevance,
                                    'sql_id': memory.get('sql_id')
                                }
                                
                                search_results.append(result)
                                
                                # Update access statistics
                                memory['access_count'] = memory.get('access_count', 0) + 1
                                memory['accessed_at'] = datetime.now().isoformat()
                    except Exception as e:
                        self.logger.debug(f"Error processing memory {key}: {e}")
            
            # Sort by relevance and limit results
            search_results.sort(key=lambda x: x.get('relevance', 0), reverse=True)
            return search_results[:limit]
        
        results = await loop.run_in_executor(self.executor, _search_memories)
        
        # Save updated access counts (non-blocking)
        if results:
            save_task = asyncio.create_task(
                self._save_critical_memory(),
                name=f"save_critical_memory_access_{datetime.now().isoformat()}"
            )
        
        return results
    
    async def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a critical memory.
        
        Args:
            memory_id: ID of the memory to delete
            
        Returns:
            Success status
        """
        if self.amnesia_mode:
            return False
            
        with self.critical_memory_lock:
            if memory_id in self.critical_memory:
                # Delete the memory
                del self.critical_memory[memory_id]
                
                # Save changes
                await self._save_critical_memory()
                return True
            else:
                return False
        
    async def get_stats(self) -> Dict:
        """
        Get statistics about critical memory.
        
        Returns:
            Dictionary with statistics
        """
        with self.critical_memory_lock:
            stats = {
                "total_memories": len(self.critical_memory),
                "file_exists": self.critical_memory_file.exists()
            }
            
            # Calculate size on disk if file exists
            if self.critical_memory_file.exists():
                stats["file_size_bytes"] = self.critical_memory_file.stat().st_size
                stats["file_size_kb"] = stats["file_size_bytes"] / 1024
            
            # Calculate average memory size
            if self.critical_memory:
                total_content_length = sum(len(memory.get('content', '')) 
                                          for memory in self.critical_memory.values())
                stats["avg_memory_size"] = total_content_length / len(self.critical_memory)
                
                # Count memories by verification score
                verification_counts = {}
                for memory in self.critical_memory.values():
                    score = memory.get('verification_score', 1.0)
                    score_bucket = round(score * 10) / 10  # Round to nearest 0.1
                    verification_counts[score_bucket] = verification_counts.get(score_bucket, 0) + 1
                
                stats["verification_distribution"] = verification_counts
            
            return stats
            
    def close(self):
        """Clean up resources."""
        # Save critical memory one last time
        try:
            # Run in a thread to avoid blocking
            asyncio.run_coroutine_threadsafe(
                self._save_critical_memory(),
                asyncio.get_event_loop()
            )
        except Exception as e:
            self.logger.error(f"Error saving critical memory during shutdown: {e}")
            
            # Fallback: Save synchronously
            try:
                with self.critical_memory_lock:
                    with open(self.critical_memory_file, 'w', encoding='utf-8') as f:
                        json.dump(self.critical_memory, f, indent=2)
            except Exception as e2:
                self.logger.error(f"Fallback save failed: {e2}")
            
        # Shutdown executor
        self.executor.shutdown(wait=True)
"""Vector memory interface for Grace using mem0."""

import asyncio
import logging
from functools import cached_property
from typing import Dict, List, Any, Optional

# Import mem0 for vector memory management
from mem0 import AsyncMemory
from grace.config.mem0_config import DEFAULT_CONFIG as MEM0_CFG

# Create a logger
logger = logging.getLogger('grace.memory.vector')

class VectorStorage:
    """
    Vector storage implementation using mem0's AsyncMemory.
    
    This class provides vector-based memory storage and retrieval
    capabilities using the mem0 library, which internally uses
    Qdrant for vector storage.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize vector storage with the provided configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger('grace.memory.vector')
        self.config = config
        self.mem0_config = config.get('mem0', MEM0_CFG)
        
        # Initialize memory with appropriate configuration
        self.mem = AsyncMemory.from_config(self.mem0_config)
        self.logger.info("Vector storage initialized with mem0")
    
    async def add_memory(self, content: str, memory_type: Any = None, 
                        user_id: str = "default", metadata: Dict = None,
                        encoding: Dict = None) -> str:
        """
        Add a memory to vector storage.
        
        Args:
            content: Memory content
            memory_type: Type of memory (used in metadata)
            user_id: User identifier
            metadata: Additional metadata
            encoding: Optional pre-computed vector encoding
            
        Returns:
            Memory identifier
        """
        try:
            # Format as message for mem0
            messages = [{"role": "user", "content": content}]
            
            # Prepare metadata
            meta = metadata or {}
            if memory_type:
                meta["memory_type"] = str(memory_type)
            
            # Add to mem0 storage
            result = await self.mem.add(
                messages=messages,
                user_id=user_id,
                metadata=meta
            )
            
            # Return the memory ID
            if isinstance(result, list) and len(result) > 0:
                memory_id = result[0].get("id")
                return memory_id
            elif isinstance(result, dict):
                return result.get("id")
            else:
                return str(result)
                
        except Exception as e:
            self.logger.error(f"Failed to add memory to vector storage: {e}")
            return None
    
    async def search_memories(self, query: str, limit: int = 10, 
                             user_id: str = "default") -> Dict:
        """
        Search for memories in vector storage.
        
        Args:
            query: Search query
            limit: Maximum number of results
            user_id: User identifier
            
        Returns:
            Dictionary with search results
        """
        try:
            # Perform search using mem0
            results = await self.mem.search(
                query=query,
                user_id=user_id,
                limit=limit
            )
            
            # Format results for compatibility
            if isinstance(results, dict) and "results" in results:
                return results
            else:
                return {"results": results}
                
        except Exception as e:
            self.logger.error(f"Vector memory search error: {e}")
            return {"results": []}
    
    async def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory from vector storage.
        
        Args:
            memory_id: Memory identifier
            
        Returns:
            Success status
        """
        try:
            # Delete using mem0
            await self.mem.delete(memory_id)
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete memory from vector storage: {e}")
            return False
    
    async def get_stats(self) -> Dict:
        """
        Get statistics about vector storage.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "provider": "mem0",
            "vector_count": -1,  # Not directly exposed by mem0
            "status": "active"
        }
    
    async def optimize_indexes(self) -> bool:
        """
        Optimize vector indexes for better performance.
        
        Returns:
            Success status
        """
        # Mem0 handles optimization internally
        return True
    
    def close(self):
        """Clean up resources."""
        # Mem0 doesn't require explicit cleanup
        pass


# For backward compatibility with the old functions
async def add_memory(text: str, user_id: str = "default") -> None:
    """
    Legacy function for adding memory.
    
    Args:
        text: Memory content
        user_id: User identifier
    """
    storage = VectorStorage({})
    await storage.add_memory(text, user_id=user_id)


async def search_memories(query: str, user_id: str = "default", k: int = 5) -> List[str]:
    """
    Legacy function for searching memories.
    
    Args:
        query: Search query
        user_id: User identifier
        k: Maximum number of results
        
    Returns:
        List of memory contents
    """
    storage = VectorStorage({})
    results = await storage.search_memories(query, limit=k, user_id=user_id)
    
    # Extract content from results
    memory_list = []
    for item in results.get("results", []):
        if isinstance(item, dict) and "memory" in item:
            memory_list.append(item["memory"])
        elif isinstance(item, str):
            memory_list.append(item)
    
    return memory_list
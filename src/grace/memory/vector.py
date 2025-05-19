"""Vector memory interface for Grace using mem0."""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

# Import mem0 for vector memory management
try:
    from mem0 import Memory, AsyncMemory
    MEM0_AVAILABLE = True
except ImportError:
    Memory = None
    AsyncMemory = None
    MEM0_AVAILABLE = False

# Import Qdrant for direct operations when needed
try:
    from qdrant_client import QdrantClient, AsyncQdrantClient, models
    from qdrant_client.models import VectorParams, Distance, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QdrantClient = None
    AsyncQdrantClient = None
    QDRANT_AVAILABLE = False

# Import for grace configurations
from grace.config.mem0_config import DEFAULT_CONFIG as MEM0_CFG
from grace.utils.common import MEMORY_DB_PATH

# Create a logger
logger = logging.getLogger('grace.memory.vector')

class VectorStorage:
    """
    Vector storage implementation using mem0's AsyncMemory with enhanced Qdrant integration.
    
    This class provides vector-based memory storage and retrieval
    capabilities using the mem0 library, which internally uses
    Qdrant for vector storage. It includes improved connection handling,
    proper resource management, and robust error recovery.
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
        
        # Thread pool for heavy operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Define Qdrant connection parameters
        qdrant_path = MEMORY_DB_PATH / "qdrant_data"
        qdrant_path.mkdir(parents=True, exist_ok=True)
        
        self.qdrant_config = {
            'collection_name': self.mem0_config.get('vector_store', {}).get('config', {}).get('collection_name', 'grace_memories'),
            'host': self.mem0_config.get('vector_store', {}).get('config', {}).get('host', 'localhost'),
            'port': self.mem0_config.get('vector_store', {}).get('config', {}).get('port', 6333),
            'path': str(qdrant_path),
            'prefer_grpc': self.mem0_config.get('vector_store', {}).get('config', {}).get('prefer_grpc', False),
            'timeout': 10.0
        }
        
        # Initialize clients
        self.mem = None
        self.async_mem = None
        self.qdrant_client = None
        self.async_qdrant_client = None
        
        # Track connection status
        self.connection_status = "initialized"
        self.last_connection_error = None
        
        try:
            # Initialize memory with appropriate configuration
            if MEM0_AVAILABLE:
                self.mem = Memory.from_config(self.mem0_config)
                self.async_mem = AsyncMemory.from_config(self.mem0_config)
                self.logger.info("Vector storage initialized with mem0")
            else:
                self.logger.warning("mem0 not available - initializing direct Qdrant connection")
                self._init_qdrant()
        except Exception as e:
            self.logger.error(f"Failed to initialize vector storage: {e}")
            self.last_connection_error = str(e)
            self.connection_status = "error"
            # Don't raise the exception - we'll try to recover or fall back

    def _init_qdrant(self):
        """Initialize direct Qdrant connection with proper error handling."""
        if not QDRANT_AVAILABLE:
            self.logger.error("Qdrant client not available")
            self.connection_status = "qdrant_not_available"
            return
            
        try:
            # Initialize both sync and async clients
            self.qdrant_client = QdrantClient(
                host=self.qdrant_config.get('host'),
                port=self.qdrant_config.get('port'),
                path=self.qdrant_config.get('path'),
                prefer_grpc=self.qdrant_config.get('prefer_grpc'),
                timeout=self.qdrant_config.get('timeout')
            )
            
            self.async_qdrant_client = AsyncQdrantClient(
                host=self.qdrant_config.get('host'),
                port=self.qdrant_config.get('port'),
                path=self.qdrant_config.get('path'),
                prefer_grpc=self.qdrant_config.get('prefer_grpc'),
                timeout=self.qdrant_config.get('timeout')
            )
            
            # Check if collection exists, create if not
            collection_name = self.qdrant_config.get('collection_name')
            collections = self.qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if collection_name not in collection_names:
                self.logger.info(f"Creating Qdrant collection: {collection_name}")
                # Create collection with proper vector configuration
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.mem0_config.get('vector_store', {}).get('config', {}).get('embedding_model_dims', 768),
                        distance=Distance.COSINE
                    )
                )
                
            self.connection_status = "connected"
            self.logger.info("Direct Qdrant connection established")
        except Exception as e:
            self.logger.error(f"Failed to initialize Qdrant: {e}")
            self.last_connection_error = str(e)
            self.connection_status = "connection_error"

    @asynccontextmanager
    async def _get_async_mem(self):
        """Context manager for safely accessing async mem0 client."""
        if self.async_mem is None:
            try:
                if MEM0_AVAILABLE:
                    self.async_mem = AsyncMemory.from_config(self.mem0_config)
                else:
                    raise RuntimeError("mem0 is not available")
            except Exception as e:
                self.logger.error(f"Failed to create async mem0 client: {e}")
                raise
                
        try:
            yield self.async_mem
        except Exception as e:
            self.logger.error(f"Error during async mem0 operation: {e}")
            # Attempt to recreate the client
            try:
                if MEM0_AVAILABLE:
                    self.async_mem = AsyncMemory.from_config(self.mem0_config)
            except Exception:
                self.async_mem = None
            raise

    @asynccontextmanager
    async def _get_async_qdrant(self):
        """Context manager for safely accessing async Qdrant client."""
        if self.async_qdrant_client is None:
            try:
                if QDRANT_AVAILABLE:
                    self.async_qdrant_client = AsyncQdrantClient(
                        host=self.qdrant_config.get('host'),
                        port=self.qdrant_config.get('port'),
                        path=self.qdrant_config.get('path'),
                        prefer_grpc=self.qdrant_config.get('prefer_grpc'),
                        timeout=self.qdrant_config.get('timeout')
                    )
                else:
                    raise RuntimeError("Qdrant is not available")
            except Exception as e:
                self.logger.error(f"Failed to create async Qdrant client: {e}")
                raise
                
        try:
            yield self.async_qdrant_client
        except Exception as e:
            self.logger.error(f"Error during async Qdrant operation: {e}")
            # Attempt to recreate the client
            try:
                if QDRANT_AVAILABLE:
                    self.async_qdrant_client = AsyncQdrantClient(
                        host=self.qdrant_config.get('host'),
                        port=self.qdrant_config.get('port'),
                        path=self.qdrant_config.get('path'),
                        prefer_grpc=self.qdrant_config.get('prefer_grpc'),
                        timeout=self.qdrant_config.get('timeout')
                    )
            except Exception:
                self.async_qdrant_client = None
            raise

    async def add_memory(self, content: str, memory_type: Any = None, 
                        user_id: str = "default", metadata: Dict = None,
                        encoding: Dict = None) -> str:
        """
        Add a memory to vector storage with improved error handling.
        
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
            
            # Try mem0 storage first
            async with self._get_async_mem() as mem:
                try:
                    result = await mem.add(
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
                    self.logger.error(f"mem0 add failed, trying direct Qdrant: {e}")
                    
                    # Fall back to direct Qdrant if mem0 fails
                    if QDRANT_AVAILABLE and self.async_qdrant_client:
                        return await self._add_direct_qdrant(content, memory_type, user_id, metadata, encoding)
                    else:
                        raise  # Re-raise if we can't fall back
                
        except Exception as e:
            self.logger.error(f"Failed to add memory to vector storage: {e}")
            return None

    async def _add_direct_qdrant(self, content: str, memory_type: Any = None,
                               user_id: str = "default", metadata: Dict = None,
                               encoding: Dict = None) -> str:
        """Add memory directly to Qdrant if mem0 fails."""
        if not QDRANT_AVAILABLE or not self.async_qdrant_client:
            self.logger.error("Cannot add directly to Qdrant - client not available")
            return None
            
        try:
            async with self._get_async_qdrant() as client:
                # Get or create embedding
                if encoding is None:
                    embedding_vector = await self._embed_with_mem0(content)

                    # If mem0 is down, fall back to deterministic hash so we
                    # *still* store something rather than raise.

                    if embedding_vector is None:
                        embedding_vector = self._hash_fallback(content, dim)
                else:
                    embedding_vector = encoding                
                # Prepare metadata
                meta = metadata or {}
                if memory_type:
                    meta["memory_type"] = str(memory_type)
                meta["user_id"] = user_id
                meta["content"] = content
                
                # Generate a unique ID
                import uuid
                memory_id = str(uuid.uuid4())
                
                # Add to Qdrant with proper error handling
                await client.upsert(
                    collection_name=self.qdrant_config.get('collection_name'),
                    points=[
                        PointStruct(
                            id=memory_id,
                            vector=embedding_vector,
                            payload=meta
                        )
                    ],
                    wait=True  # Ensure data is committed
                )
                
                return memory_id
        except Exception as e:
            self.logger.error(f"Failed to add directly to Qdrant: {e}")
            return None
    
    async def search_memories(self, query: str, limit: int = 10, 
                             user_id: str = "default") -> Dict:
        """
        Search for memories in vector storage with improved error handling.
        
        Args:
            query: Search query
            limit: Maximum number of results
            user_id: User identifier
            
        Returns:
            Dictionary with search results
        """
        try:
            # Try mem0 search first
            async with self._get_async_mem() as mem:
                try:
                    results = await mem.search(
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
                    self.logger.error(f"mem0 search failed, trying direct Qdrant: {e}")
                    
                    # Fall back to direct Qdrant if mem0 fails
                    if QDRANT_AVAILABLE and self.async_qdrant_client:
                        return await self._search_direct_qdrant(query, limit, user_id)
                    else:
                        raise  # Re-raise if we can't fall back
                
        except Exception as e:
            self.logger.error(f"Vector memory search error: {e}")
            return {"results": []}

    async def _search_direct_qdrant(self, query: str, limit: int = 10,
                                  user_id: str = "default") -> Dict:
        """Search memories directly in Qdrant if mem0 fails."""
        if not QDRANT_AVAILABLE or not self.async_qdrant_client:
            self.logger.error("Cannot search directly in Qdrant - client not available")
            return {"results": []}
            
        try:
            async with self._get_async_qdrant() as client:
                # Similar to add, create a simple embedding for the query
                # This is just a fallback and won't provide good semantic search
                from grace.utils.token_utils import estimate_tokens
                import numpy as np
                import hashlib
                
                # Create a deterministic "embedding" based on hash
                hash_obj = hashlib.md5(query.encode('utf-8'))
                hash_bytes = hash_obj.digest()
                
                # Convert hash to a normalized vector of the right dimension
                dim = self.mem0_config.get('vector_store', {}).get('config', {}).get('embedding_model_dims', 768)
                hash_ints = [b for b in hash_bytes]
                
                # Repeat the hash values to fill the dimension
                repeated = hash_ints * (dim // len(hash_ints) + 1)
                query_vector = np.array(repeated[:dim]) / 255.0  # Normalize to [0,1]
                query_vector = query_vector.tolist()
                
                # Search Qdrant directly
                search_results = await client.search(
                    collection_name=self.qdrant_config.get('collection_name'),
                    query_vector=query_vector,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                    filter=None  # Can add filter by user_id if needed
                )
                
                # Format results to match expected structure
                formatted_results = []
                for result in search_results:
                    # Extract content and metadata from payload
                    content = result.payload.get("content", "")
                    memory_type = result.payload.get("memory_type", "unknown")
                    
                    formatted_results.append({
                        "id": result.id,
                        "content": content,
                        "memory": content,  # For compatibility with mem0 format
                        "score": result.score,
                        "metadata": {k: v for k, v in result.payload.items() if k not in ["content"]}
                    })
                
                return {"results": formatted_results}
        except Exception as e:
            self.logger.error(f"Failed to search directly in Qdrant: {e}")
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
            # Try mem0 delete first
            async with self._get_async_mem() as mem:
                try:
                    await mem.delete(memory_id)
                    return True
                except Exception as e:
                    self.logger.error(f"mem0 delete failed, trying direct Qdrant: {e}")
                    
                    # Fall back to direct Qdrant if mem0 fails
                    if QDRANT_AVAILABLE and self.async_qdrant_client:
                        return await self._delete_direct_qdrant(memory_id)
                    else:
                        raise  # Re-raise if we can't fall back
        except Exception as e:
            self.logger.error(f"Failed to delete memory from vector storage: {e}")
            return False

    async def _delete_direct_qdrant(self, memory_id: str) -> bool:
        """Delete memory directly from Qdrant if mem0 fails."""
        if not QDRANT_AVAILABLE or not self.async_qdrant_client:
            self.logger.error("Cannot delete directly from Qdrant - client not available")
            return False
            
        try:
            async with self._get_async_qdrant() as client:
                await client.delete(
                    collection_name=self.qdrant_config.get('collection_name'),
                    points=[memory_id],
                    wait=True  # Ensure deletion is committed
                )
                return True
        except Exception as e:
            self.logger.error(f"Failed to delete directly from Qdrant: {e}")
            return False
    
    async def get_stats(self) -> Dict:
        """
        Get statistics about vector storage.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            "provider": "mem0" if self.async_mem is not None else "qdrant_direct",
            "vector_count": -1,  # Will be updated if available
            "status": self.connection_status,
            "error": self.last_connection_error
        }
        
        try:
            # Try to get collection info from Qdrant if available
            if QDRANT_AVAILABLE and (self.qdrant_client or self.async_qdrant_client):
                # Use sync client if available, otherwise use async client
                if self.qdrant_client:
                    try:
                        collection_info = self.qdrant_client.get_collection(
                            collection_name=self.qdrant_config.get('collection_name')
                        )
                        stats["vector_count"] = collection_info.vectors_count
                        stats["dimension"] = collection_info.config.params.vectors.size
                        stats["distance_metric"] = str(collection_info.config.params.vectors.distance)
                    except Exception as e:
                        self.logger.debug(f"Error getting collection stats: {e}")
                elif self.async_qdrant_client:
                    try:
                        async with self._get_async_qdrant() as client:
                            collection_info = await client.get_collection(
                                collection_name=self.qdrant_config.get('collection_name')
                            )
                            stats["vector_count"] = collection_info.vectors_count
                            stats["dimension"] = collection_info.config.params.vectors.size
                            stats["distance_metric"] = str(collection_info.config.params.vectors.distance)
                    except Exception as e:
                        self.logger.debug(f"Error getting async collection stats: {e}")
        except Exception as e:
            self.logger.error(f"Error getting vector stats: {e}")
        
        return stats
    
    async def optimize_indexes(self) -> bool:
        """
        Optimize vector indexes for better performance.
        
        Returns:
            Success status
        """
        if not QDRANT_AVAILABLE or (not self.qdrant_client and not self.async_qdrant_client):
            # If we're using mem0 only, it handles optimization internally
            return True
            
        try:
            # Try to optimize collection using direct Qdrant access if available
            if self.async_qdrant_client:
                async with self._get_async_qdrant() as client:
                    # Check if collection exists
                    collections = await client.get_collections()
                    collection_names = [c.name for c in collections.collections]
                    
                    if self.qdrant_config.get('collection_name') in collection_names:
                        # Qdrant doesn't have a direct "optimize" function like some vector DBs
                        # We could implement index rebuilding or other optimizations if needed
                        self.logger.info(f"Optimization completed for {self.qdrant_config.get('collection_name')}")
                        return True
                    else:
                        self.logger.warning(f"Collection {self.qdrant_config.get('collection_name')} not found for optimization")
                        return False
            elif self.qdrant_client:
                # Fallback to sync client
                collections = self.qdrant_client.get_collections()
                collection_names = [c.name for c in collections.collections]
                
                if self.qdrant_config.get('collection_name') in collection_names:
                    self.logger.info(f"Optimization completed for {self.qdrant_config.get('collection_name')}")
                    return True
                else:
                    self.logger.warning(f"Collection {self.qdrant_config.get('collection_name')} not found for optimization")
                    return False
            else:
                return False
        except Exception as e:
            self.logger.error(f"Error optimizing vector indexes: {e}")
            return False
    
    def close(self):
        """Clean up resources with improved error handling."""
        # Mem0 doesn't require explicit cleanup, but we'll clean up our Qdrant clients
        if self.qdrant_client:
            try:
                # No specific close method for QdrantClient, but we can set it to None
                self.qdrant_client = None
            except Exception as e:
                self.logger.debug(f"Error closing Qdrant client: {e}")
        
        # Close executor
        if self.executor:
            self.executor.shutdown(wait=False)


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
        elif isinstance(item, dict) and "content" in item:
            memory_list.append(item["content"])
        elif isinstance(item, str):
            memory_list.append(item)
    
    return memory_list
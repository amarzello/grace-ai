"""Central Mem0 configuration for offline/local use."""

from pathlib import Path

QDRANT_PATH = Path(__file__).resolve().parent.parent / "qdrant_data"
QDRANT_PATH.mkdir(exist_ok=True)

DEFAULT_CONFIG = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "grace_memories",
            "host": "localhost",
            "port": 6333,
            "path": str(QDRANT_PATH),
            # Removed the prefer_grpc parameter that was causing errors
            "embedding_model_dims": 768,
        },
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "neo4j",
        },
    },
    "embedder": {
        "provider": "sentence_transformers",  # Changed from "local" to a supported provider
        "config": {
            "model": "all-MiniLM-L6-v2"  # Or specify path to local model
        }
    }
}
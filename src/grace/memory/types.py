"""
Grace AI System - Memory Types Module

This module defines the types of memories used in the Grace AI system with clear
categorization and descriptions for better integration with the hybrid memory
architecture.
"""

from enum import Enum


class MemoryType(Enum):
    """Types of memories in the system with descriptions and storage strategies."""
    # General conversational context - stored in vector database for similarity search
    CONTEXTUAL = "contextual"
    
    # Important information that must be retained - stored in all three systems
    # for maximum durability and multiple access methods
    CRITICAL = "critical"
    
    # Past conversations - stored primarily in SQLite with vector embeddings
    CONVERSATION = "conversation"
    
    # External reference materials - stored in both vector and relational
    # databases for semantic search and structural queries
    REFERENCE = "reference"
    
    # User preferences and settings - stored primarily in relational database
    # with references in graph database for relationship tracking
    USER_PREFERENCE = "user_preference"
    
    # Verification data - used to verify the accuracy of responses
    # stored in relational database with links to source evidence
    VERIFICATION = "verification"
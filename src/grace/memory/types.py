"""
Grace AI System - Memory Types Module

This module defines the types of memories used in the Grace AI system.
"""

from enum import Enum


class MemoryType(Enum):
    """Types of memories in the system with descriptions."""
    CONTEXTUAL = "contextual"     # General conversational context
    CRITICAL = "critical"         # Important information that must be retained
    CONVERSATION = "conversation" # Past conversations
    REFERENCE = "reference"       # External reference materials
    USER_PREFERENCE = "user_preference"  # User preferences and settings
    VERIFICATION = "verification"  # Verification data

"""
Grace AI System - Common Utilities

This module contains common utilities and shared components for the Grace AI system.
"""

import os
import sys
import json
import logging
import yaml
import re
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from datetime import datetime


# Configure logging
def setup_logging(debug: bool = False):
    """Configure logging with appropriate levels and handlers."""
    log_level = logging.DEBUG if debug else logging.INFO
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    
    # Configure grace logger
    grace_logger = logging.getLogger('grace')
    grace_logger.setLevel(log_level)
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Create handlers if they don't exist
    if not grace_logger.handlers:
        # File handler with rotation
        try:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                'logs/grace_system.log',
                maxBytes=10*1024*1024,  # 10 MB
                backupCount=5
            )
        except Exception:
            # Fall back to simple file handler if rotation not available
            file_handler = logging.FileHandler('logs/grace_system.log')
            
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        grace_logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        grace_logger.addHandler(console_handler)
    
    # Silence noisy loggers, not all logs
    for logger_name in ['ovos_config', 'ovos_utils', 'ovos_bus_client', 'qdrant_client']:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    
    return grace_logger


# Paths configuration
GRACE_HOME = Path.home() / '.grace'
MEMORY_DB_PATH = GRACE_HOME / 'memory'
LOGS_PATH = GRACE_HOME / 'logs'
CONFIG_PATH = GRACE_HOME / 'config'
MODELS_PATH = GRACE_HOME / 'models'
REFERENCE_PATH = GRACE_HOME / 'reference'

# Create directories
for path in [MEMORY_DB_PATH, LOGS_PATH, CONFIG_PATH, MODELS_PATH, REFERENCE_PATH]:
    path.mkdir(parents=True, exist_ok=True)


@dataclass
class ConversationEntry:
    """Structure for logging conversations with improved field organization."""
    timestamp: str = ""
    user_input: str = ""
    stt_transcript: str = ""
    memory_context: Dict = field(default_factory=dict)
    prompt: str = ""
    model_response: str = ""
    thinking_process: str = ""
    json_response: Dict = field(default_factory=dict)
    command_result: str = ""
    tts_output: str = ""
    error: str = ""
    metadata: Dict = field(default_factory=dict)
    verification_result: Dict = field(default_factory=dict)
    
    def to_dict(self):
        """Convert to dictionary for storage."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data):
        """Create instance from dictionary."""
        return cls(**data)


class MemoryType(Enum):
    """Types of memories in the system with descriptions."""
    CONTEXTUAL = "contextual"     # General conversational context
    CRITICAL = "critical"         # Important information that must be retained
    CONVERSATION = "conversation" # Past conversations
    REFERENCE = "reference"       # External reference materials
    USER_PREFERENCE = "user_preference"  # User preferences and settings
    VERIFICATION = "verification"  # Verification data


# Default configuration
DEFAULT_CONFIG = {
    'llama': {
        'model_path': str(MODELS_PATH / 'QWQ-32B-Q4_K_M.gguf'),
        'n_ctx': 32768,           # Context window size
        'n_gpu_layers': 31,       # Number of layers to offload to GPU
        'n_threads': 6,           # CPU threads
        'n_batch': 512,           # Batch size for prompt processing
        'rope_freq_base': 10000,  # RoPE frequency base
        'rope_freq_scale': 1.0,   # RoPE frequency scaling factor
        'temperature': 0.6,       # Sampling temperature - optimized for QWQ-32B
        'top_p': 0.95,            # Top-p sampling parameter for QWQ-32B
        'top_k': 30,              # Top-k sampling parameter for QWQ-32B
        'min_p': 0.0,             # Min-p parameter for QWQ-32B
        'presence_penalty': 1.0,  # Presence penalty for QWQ-32B
        'max_tokens': 2048,       # Maximum tokens to generate
        'use_mlock': True,        # Lock memory to prevent swapping
        'use_mmap': False,        # Memory map model
        'verbose': False          # Verbose llama.cpp output
    },
    'whisper': {
        'model_size': 'large-v2', # Whisper model size
        'device': 'cuda',         # Device to run model on
        'compute_type': 'float16', # Computation precision
        'language': 'en',         # Language code
        'beam_size': 5,           # Beam search size
        'timeout': 30             # Timeout for transcription in seconds
    },
    'piper': {
        'model_path': None,       # Piper model path (will auto-detect)
        'sample_rate': 22050      # Audio sample rate
    },
    'audio': {
        'use_microphone': True,   # Use microphone input
        'vad_aggressiveness': 3,  # Voice Activity Detection aggressiveness (0-3)
        'silence_threshold': 0.1, # Silence threshold (0-1)
        'silence_duration': 1.0,  # Silence duration to end recording (seconds)
        'sample_rate': 16000,     # Audio sample rate for recording
        'channels': 1,            # Audio channels (1=mono, 2=stereo)
        'input_device': None,     # Input device name or index (None for default)
        'mute': False             # Mute text-to-speech output
    },
    'memory': {
        'max_context_tokens': 20000,  # Maximum tokens for context
        'search_limit': 50,       # Maximum memories to return in search
        'qdrant_host': 'localhost', # Qdrant server host
        'qdrant_port': 6333,      # Qdrant server port
        'use_in_memory': True,    # Use in-memory storage
        'verification_threshold': 0.85, # Verification confidence threshold
        'archive_age_days': 365,  # Days before archiving memories
        'memory_cache_size': 1000,  # Size of memory cache
        'sqlite_wal_mode': True   # Enable WAL mode for SQLite
    },
    'ovos': {
        'host': 'localhost',      # OVOS messagebus host
        'port': 8181,             # OVOS messagebus port
        'retry_attempts': 3,      # Connection retry attempts
        'retry_delay': 1,         # Delay between retries (seconds)
        'disable_ovos': False,    # Set to True to disable OVOS integration
        'reconnect_delay': 30     # Seconds between reconnection attempts
    },
    'system': {
        'error_fallback': True,   # Enable fallback mechanisms on errors
        'debug_json': False,      # Print raw JSON responses for debugging
        'reconnect_attempts': 3,  # Number of times to try reconnecting to services
        'input_mode': 'both',     # Default input mode: 'text', 'voice', or 'both'
        'log_rotation': True,     # Enable log rotation
        'max_log_size_mb': 10,    # Maximum log file size before rotation
        'max_log_files': 5,       # Maximum number of log files to keep
        'maintenance_interval_hours': 24,  # Hours between automatic maintenance
        'show_raw_output': True,  # Show raw model output when parsing fails
        'backup_config': True     # Create backups of config files
    },
    'amnesia_mode': False,        # Privacy mode - don't store memories
    'debug': False                # Enable debug mode for more detailed output
}


def load_config(config_file=None, config_override=None):
    """
    Load configuration from file and apply overrides.
    
    Args:
        config_file: Path to config file
        config_override: Dictionary of config overrides
        
    Returns:
        Merged configuration dictionary
    """
    logger = logging.getLogger('grace.config')
    config = DEFAULT_CONFIG.copy()
    
    # Create default config if none exists and no override provided
    if not config_file and not Path(CONFIG_PATH / 'config.yaml').exists():
        try:
            # Create config directory if it doesn't exist
            CONFIG_PATH.mkdir(parents=True, exist_ok=True)
            
            with open(CONFIG_PATH / 'config.yaml', 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            logger.info(f"Created default configuration at {CONFIG_PATH / 'config.yaml'}")
        except Exception as e:
            logger.warning(f"Failed to create default config file: {e}")
            logger.info("No configuration file found. Using default configuration.")
    
    # Load config file if provided
    config_path = None
    if config_file and Path(config_file).exists():
        config_path = config_file
    elif Path(CONFIG_PATH / 'config.yaml').exists():
        config_path = CONFIG_PATH / 'config.yaml'
        
    if config_path:
        try:
            # Create backup of config file if enabled
            if config.get('system', {}).get('backup_config', True):
                backup_path = str(config_path) + ".backup"
                try:
                    import shutil
                    shutil.copy2(config_path, backup_path)
                    logger.debug(f"Created backup of config file at {backup_path}")
                except Exception as e:
                    logger.debug(f"Failed to create config backup: {e}")
            
            # Load the config file
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f) or {}
                config = deep_merge(config, loaded_config)
                logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Error loading config file {config_path}: {e}")
    else:
        logger.info("No configuration file found. Using default configuration.")
    
    # Apply overrides
    if config_override:
        config = deep_merge(config, config_override)
        logger.info("Applied configuration overrides")
    
    # Update QWQ default parameters if using QWQ model
    if 'QWQ' in config.get('llama', {}).get('model_path', ''):
        # Set QWQ-specific parameters if not already set
        llama_config = config.get('llama', {})
        
        # Add guard clause for config_override
        if config_override and 'llama' in config_override:
            llama_override = config_override.get('llama', {})
            if 'temperature' not in llama_override:
                llama_config['temperature'] = 0.6
            if 'top_p' not in llama_override:
                llama_config['top_p'] = 0.95
            if 'top_k' not in llama_override:
                llama_config['top_k'] = 30
            if 'min_p' not in llama_override:
                llama_config['min_p'] = 0.0
            if 'presence_penalty' not in llama_override:
                llama_config['presence_penalty'] = 1.0
        else:
            # No overrides specified, just set defaults
            llama_config['temperature'] = 0.6
            llama_config['top_p'] = 0.95
            llama_config['top_k'] = 30
            llama_config['min_p'] = 0.0
            llama_config['presence_penalty'] = 1.0
        
        logger.info("Applied QWQ-specific model parameters")
    
    return config


def deep_merge(base, override):
    """
    Recursively merge dictionaries.
    
    Args:
        base: Base dictionary
        override: Override dictionary
        
    Returns:
        Merged dictionary
    """
    if isinstance(base, dict) and isinstance(override, dict):
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    return override


def calculate_relevance(query: str, content: str) -> float:
    """
    Calculate relevance score between query and content.
    
    Args:
        query: Search query
        content: Content to check against
        
    Returns:
        Relevance score (0.0-1.0)
    """
    if not query or not content:
        return 0.0
        
    # Normalize and tokenize query and content
    query_tokens = set(_tokenize(query))
    content_tokens = set(_tokenize(content))
    
    if not query_tokens or not content_tokens:
        return 0.0
    
    # Base relevance: Jaccard similarity with TF-IDF inspired weighting
    # Jaccard similarity: intersection / union
    intersection = query_tokens.intersection(content_tokens)
    union = query_tokens.union(content_tokens)
    
    if not union:
        return 0.0
        
    # Basic Jaccard similarity
    jaccard = len(intersection) / len(union)
    
    # Additional relevance factors:
    
    # 1. Check for exact phrase matches
    phrase_bonus = 0.0
    for phrase in _extract_phrases(query):
        if len(phrase) > 3 and phrase in content:
            phrase_bonus += 0.2
    
    # 2. Check for semantic key terms
    semantic_bonus = 0.0
    key_terms = _extract_key_terms(query)
    content_lower = content.lower()
    for term in key_terms:
        if term in content_lower:
            semantic_bonus += 0.1
    
    # 3. Length ratio factor - prefer matching with similarly sized content
    length_ratio = min(len(query), len(content)) / max(len(query), len(content))
    length_factor = 0.5 + (0.5 * length_ratio)  # Range: 0.5 - 1.0
    
    # Combine factors (weighted sum)
    relevance = (
        (0.5 * jaccard) + 
        (0.3 * min(phrase_bonus, 0.4)) +  # Cap at 0.4
        (0.2 * min(semantic_bonus, 0.3))  # Cap at 0.3
    ) * length_factor
    
    return min(relevance, 1.0)  # Ensure it doesn't exceed 1.0


def _tokenize(text: str) -> List[str]:
    """
    Tokenize text for relevance calculation.
    
    Args:
        text: Text to tokenize
        
    Returns:
        List of tokens
    """
    if not text:
        return []
        
    # Convert to lowercase and clean special characters
    text = text.lower()
    
    # Remove punctuation but keep words together
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Split on whitespace and filter out stop words and very short words
    stop_words = {'the', 'and', 'is', 'in', 'to', 'a', 'of', 'for', 'with', 'on', 'at', 'from'}
    tokens = [word for word in text.split() if word not in stop_words and len(word) > 1]
    
    return tokens


def _extract_phrases(text: str) -> List[str]:
    """
    Extract meaningful phrases from text.
    
    Args:
        text: Text to process
        
    Returns:
        List of phrases
    """
    # Split into potential phrases
    phrases = []
    
    # Add full text as a phrase if not too long
    if 3 < len(text) < 50:
        phrases.append(text)
    
    # Split on punctuation and common conjunctions
    splits = re.split(r'[,.;!?]|\s(?:and|but|or|because|however)\s', text)
    for split in splits:
        split = split.strip()
        if split and 3 < len(split) < 30:
            phrases.append(split)
    
    return phrases


def _extract_key_terms(text: str) -> Set[str]:
    """
    Extract key terms from text.
    
    Args:
        text: Text to process
        
    Returns:
        Set of key terms
    """
    # Start with full tokenization
    tokens = _tokenize(text)
    
    # Identify potential key terms (longer words are more likely to be important)
    key_terms = {token for token in tokens if len(token) > 4}
    
    # Add bigrams (pairs of adjacent words) for more context
    words = [w for w in text.lower().split() if len(w) > 1]
    if len(words) > 1:
        bigrams = {f"{words[i]} {words[i+1]}" for i in range(len(words)-1)}
        key_terms.update(bigrams)
    
    return key_terms


def print_debug_separator(title="", char="=", width=80):
    """
    Print a debug separator line with an optional title.
    
    Args:
        title: Optional title to display in the separator
        char: Character to use for the separator line
        width: Width of the separator line
    """
    if not title:
        print(char * width)
        return
        
    title = f" {title} "
    padding = (width - len(title)) // 2
    if padding > 0:
        line = char * padding + title + char * (width - padding - len(title))
        print(line)
    else:
        print(title)


def check_package_available(package_name):
    """
    Check if a Python package is available.
    
    Args:
        package_name: Package name to check
        
    Returns:
        True if available, False otherwise
    """
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

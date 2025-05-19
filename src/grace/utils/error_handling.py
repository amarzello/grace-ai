"""
Grace AI System - Error Handling and Resource Management Module

This module provides standardized error handling and resource management
patterns for use throughout the Grace AI system.
"""

import logging
import traceback
import json
import datetime
import threading
import tempfile
import shutil
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, asynccontextmanager
from typing import Dict, Any, Optional, Callable

# Define a base exception class for the project
class GraceAIError(Exception):
    """Base exception class for all Grace AI errors."""
    pass

# Create specific exception types inheriting from the base class
class ConfigurationError(GraceAIError):
    """Raised when there's an error in the configuration."""
    pass

class ResourceError(GraceAIError):
    """Raised when there's an error with system resources."""
    pass

class MemoryError(GraceAIError):
    """Raised when there's an error with memory systems."""
    pass

class ModelError(GraceAIError):
    """Raised when there's an error with ML models."""
    pass

class ErrorLogger:
    """Comprehensive error logging system with structured format and persistence."""
    
    def __init__(self, log_file="grace_ai_errors.log"):
        self.logger = logging.getLogger("grace_ai_error_logger")
        self.logger.setLevel(logging.ERROR)
        
        # File handler for persistent error logs
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.ERROR)
        
        # Create a JSON formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
    
    def log_error(self, error, context=None):
        """Log an error with full traceback and contextual information"""
        error_info = {
            "timestamp": datetime.datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": context or {}
        }
        
        self.logger.error(json.dumps(error_info))
        return error_info

@contextmanager
def temp_workspace():
    """Creates a temporary workspace directory that's automatically cleaned up"""
    temp_dir = tempfile.mkdtemp(prefix="grace_ai_workspace_")
    try:
        yield temp_dir
    finally:
        # Clean up the directory and all its contents
        shutil.rmtree(temp_dir, ignore_errors=True)

class ResourcePool:
    """Generic resource pool to limit and reuse expensive resources."""
    
    def __init__(self, create_func, max_resources=10, timeout=5):
        self.create_func = create_func
        self.max_resources = max_resources
        self.timeout = timeout
        self.available = []
        self.in_use = set()
        self._lock = threading.RLock()
    
    def acquire(self):
        """Acquire a resource from the pool."""
        start_time = time.time()
        while True:
            with self._lock:
                if self.available:
                    resource = self.available.pop()
                    self.in_use.add(resource)
                    return resource
                
                if len(self.in_use) < self.max_resources:
                    # Create a new resource if under the limit
                    resource = self.create_func()
                    self.in_use.add(resource)
                    return resource
            
            # Wait if we've reached max resources
            if time.time() - start_time > self.timeout:
                raise TimeoutError("Timed out waiting for a resource")
            
            time.sleep(0.1)
    
    def release(self, resource):
        """Release a resource back to the pool."""
        with self._lock:
            if resource in self.in_use:
                self.in_use.remove(resource)
                self.available.append(resource)
    
    def close(self):
        """Close all resources in the pool."""
        with self._lock:
            for resource in self.available:
                self._close_resource(resource)
            for resource in self.in_use:
                self._close_resource(resource)
            self.available = []
            self.in_use = set()
    
    def _close_resource(self, resource):
        """Close a single resource."""
        try:
            if hasattr(resource, 'close'):
                resource.close()
            elif hasattr(resource, '__del__'):
                del resource
        except Exception as e:
            logging.error(f"Error closing resource: {e}")

@asynccontextmanager
async def async_database_connection(connection_string):
    """Async context manager for database connections"""
    connection = await create_async_db_connection(connection_string)
    try:
        yield connection
    finally:
        await connection.close()

async def process_items_with_limit(items, process_func, max_concurrent=5):
    """Process items with a limit on concurrent operations"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_item_with_semaphore(item):
        async with semaphore:
            return await process_func(item)
    
    # Process all items concurrently, but with limited parallelism
    results = await asyncio.gather(
        *[process_item_with_semaphore(item) for item in items],
        return_exceptions=True
    )
    
    # Handle any exceptions
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logging.error(f"Error processing item {items[i]}: {result}")
            processed_results.append(None)
        else:
            processed_results.append(result)
    
    return processed_results

async def fetch_with_timeout(url, timeout=10):
    """Fetch data with a timeout to prevent indefinite waiting"""
    try:
        async with asyncio.timeout(timeout):
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    response.raise_for_status()
                    return await response.json()
    except asyncio.TimeoutError:
        logging.warning(f"Request to {url} timed out after {timeout} seconds")
        raise TimeoutError(f"Request to {url} timed out")
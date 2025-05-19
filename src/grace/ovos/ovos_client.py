"""
Grace AI System - OVOS Client Module

This module implements core client functionality for OpenVoiceOS messagebus integration
with improved thread safety and connection handling.
"""

import logging
import time
import threading
import socket
import os
import json
import asyncio
import queue
from typing import Dict, Optional, List, Any, Callable, Tuple
from pathlib import Path

# Try to import OVOS dependencies
try:
    from ovos_bus_client import MessageBusClient, Message
    OVOS_AVAILABLE = True
except ImportError:
    OVOS_AVAILABLE = False
    # Create improved fallback classes
    class Message:
        def __init__(self, message_type, data=None, context=None):
            self.type = message_type
            self.data = data or {}
            self.context = context or {}
    
    class MessageBusClient:
        def __init__(self, *args, **kwargs):
            self.connected = False
            # Track connection timeout for better error reporting
            self.connection_timeout = kwargs.get('timeout', 5)
       
        def run_in_thread(self):
            # Indicate that the client is in a disconnected but defined state
            # This should still return False for is_connected() checks
            pass
           
        def emit(self, message):
            # Log that the message couldn't be sent
            logging.getLogger('grace.ovos').debug(
                f"Attempted to emit message '{message.type}' but OVOS client not available")
            return False
           
        def wait_for_response(self, message, timeout=None):
            # Log that no response can be received
            logging.getLogger('grace.ovos').debug(
                f"Attempted to wait for response to '{message.type}' but OVOS client not available")
            return None
           
        def close(self):
            # Add method for proper resource cleanup
            pass
        
        def on(self, message_type, handler):
            # Register handlers (no-op in fallback)
            logging.getLogger('grace.ovos').debug(
                f"Attempted to register handler for '{message_type}' but OVOS client not available")
            return False


class OVOSClient:
    """
    OpenVoiceOS messagebus client with improved connection handling and thread safety.
    
    This class manages the connection to the OVOS messagebus and provides
    methods for message transmission and reception with proper resource
    management and error handling.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the OVOS client with the provided configuration.
        
        Args:
            config: OVOS client configuration
        """
        self.logger = logging.getLogger('grace.ovos.client')
        self.config = config
        self.ovos_config = config.get('ovos', {})
        self.client = None
        self.connected = False
        self.reconnect_thread = None
        self.running = True
        
        # Track connection attempts and last error
        self.connection_attempts = 0
        self.connection_failures = 0
        self.last_connection_error = None
        self.connection_status = "not_initialized"
        
        # Connection lock to prevent race conditions during reconnection
        self.connection_lock = threading.RLock()
        
        # Message queue for when connection is lost temporarily
        self.message_queue = []
        self.message_queue_limit = self.ovos_config.get('message_queue_limit', 100)
        self.message_queue_lock = threading.RLock()
        
        # Handler registry for auto-reregistration after reconnection
        self.handler_registry = {}
        self.handler_lock = threading.RLock()
        
        # Initialize connection if not explicitly disabled
        if not self.ovos_config.get('disable_ovos', False):
            self._connect()
    
    def _check_port_available(self, host, port):
        """
        Check if a port is in use (indicating the service might be running).
        
        Args:
            host: Host to check
            port: Port to check
            
        Returns:
            True if something is listening on the port
        """
        try:
            # Try to connect to the port - if connection succeeds, something is there
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0  # 0 = success, meaning port is open
        except Exception:
            return False
    
    def _connect(self):
        """Connect to OVOS messagebus with retry mechanism and diagnostics."""
        if not OVOS_AVAILABLE:
            self.logger.warning("ovos_bus_client not available. Running without OVOS integration.")
            self.connection_status = "module_not_available"
            return
            
        # Use connection lock to prevent race conditions
        with self.connection_lock:
            # Silence OVOS config errors
            os.environ['OVOS_CONFIG_LOG_LEVEL'] = 'CRITICAL'
            
            # Check if OVOS should be disabled
            if self.ovos_config.get('disable_ovos', False):
                self.logger.info("OVOS integration disabled by configuration")
                self.connection_status = "disabled"
                return
                
            max_attempts = self.ovos_config.get('retry_attempts', 3)
            retry_delay = self.ovos_config.get('retry_delay', 1)
            host = self.ovos_config.get('host', 'localhost')
            port = self.ovos_config.get('port', 8181)
            
            self.logger.info(f"Attempting to connect to OVOS messagebus at {host}:{port}")
            
            self.connection_attempts = 0
            
            # Try multiple ports if the configured one fails
            alternate_ports = [8181, 8080, 8000, 8765]
            if port in alternate_ports:
                alternate_ports.remove(port)
                alternate_ports.insert(0, port)  # Try configured port first
            else:
                alternate_ports.insert(0, port)
                
            for port in alternate_ports:
                if not self._check_port_available(host, port):
                    self.logger.debug(f"Port {port} not available, skipping")
                    continue
                    
                for attempt in range(max_attempts):
                    self.connection_attempts += 1
                    try:
                        self.logger.debug(f"Trying to connect to {host}:{port} (attempt {attempt+1})")
                        
                        # Clean up any existing client
                        if self.client:
                            try:
                                if hasattr(self.client, 'close'):
                                    self.client.close()
                            except Exception:
                                pass
                            self.client = None
                            
                        self.client = MessageBusClient(
                            host=host,
                            port=port,
                            ssl=False,  # Added to ensure no SSL connection is attempted
                            timeout=3   # Reduced timeout for faster failure detection
                        )
                        self.client.run_in_thread()
                        
                        # Verify connection with a timeout
                        start_time = time.time()
                        while time.time() - start_time < 3:  # Reduced timeout from 5 to 3 seconds
                            if hasattr(self.client, 'connected') and self.client.connected:
                                self.connected = True
                                self.connection_status = "connected"
                                self.logger.info(f"Connected to OVOS messagebus at {host}:{port}")
                                
                                # Update config with working port
                                self.ovos_config['port'] = port
                                
                                # Re-register handlers
                                self._reregister_handlers()
                                
                                # Process any queued messages
                                self._process_message_queue()
                                
                                return
                            time.sleep(0.1)
                            
                        # If we got here, connection wasn't established in time
                        self.last_connection_error = f"Connection timeout on port {port}"
                        self.logger.debug(f"OVOS connection timed out on port {port} (attempt {attempt + 1}/{max_attempts})")
                        self.connection_status = "timeout"
                        
                        # Clean up client that timed out
                        if self.client:
                            try:
                                if hasattr(self.client, 'close'):
                                    self.client.close()
                            except Exception:
                                pass
                            self.client = None
                            
                    except Exception as e:
                        self.last_connection_error = f"Error on port {port}: {str(e)}"
                        self.logger.debug(f"OVOS connection attempt {attempt + 1}/{max_attempts} failed on port {port}: {e}")
                        self.connection_status = "connection_error"
                        
                        # Clean up failed connection attempt
                        if self.client:
                            try:
                                if hasattr(self.client, 'close'):
                                    self.client.close()
                            except Exception:
                                pass
                            self.client = None
                        
                    if attempt < max_attempts - 1:
                        time.sleep(retry_delay)
                        
                # If we've tried all attempts on this port, move to next port
                if not self.connected:
                    self.logger.debug(f"Failed to connect on port {port}, trying next port if available")
                    
            # If we reach here, all connection attempts on all ports failed
            self.connection_failures += 1
            self.logger.warning(f"Failed to connect to OVOS messagebus after trying multiple ports - running in standalone mode")
            self.connected = False
            
            # Only start reconnection thread if OVOS isn't explicitly disabled
            if not self.ovos_config.get('disable_ovos', False):
                self._start_reconnect_thread()
                
    def _start_reconnect_thread(self):
        """Start a background thread for reconnection attempts."""
        if self.reconnect_thread and self.reconnect_thread.is_alive():
            return
            
        self.reconnect_thread = threading.Thread(
            target=self._reconnect_loop,
            daemon=True,
            name="OVOS-Reconnect"
        )
        self.reconnect_thread.start()
        
    def _reconnect_loop(self):
        """Background loop for reconnection attempts with improved diagnostics."""
        reconnect_delay = self.ovos_config.get('reconnect_delay', 30)  # 30 seconds between attempts
        max_reconnect_attempts = self.ovos_config.get('max_reconnect_attempts', 0)  # 0 = infinite
        
        reconnect_count = 0
        
        while self.running and not self.connected and OVOS_AVAILABLE:
            # Check if we've hit the max reconnect attempts (if configured)
            if max_reconnect_attempts > 0 and reconnect_count >= max_reconnect_attempts:
                self.logger.warning(f"Giving up reconnection after {reconnect_count} attempts")
                # Disable OVOS integration after max attempts
                self.ovos_config['disable_ovos'] = True
                self.connection_status = "disabled_max_attempts"
                break
                
            # Check if OVOS has been disabled
            if self.ovos_config.get('disable_ovos', False):
                self.logger.debug("OVOS integration now disabled, stopping reconnection attempts")
                break
                
            reconnect_count += 1
            self.logger.debug(f"Attempting to reconnect to OVOS messagebus (attempt {reconnect_count})")
            
            # Use connection lock to prevent race conditions
            with self.connection_lock:
                try:
                    host = self.ovos_config.get('host', 'localhost')
                    port = self.ovos_config.get('port', 8181)
                    
                    # Check if port is available
                    if not self._check_port_available(host, port):
                        self.logger.debug(f"Port {port} not available during reconnection")
                        continue
                    
                    # Clean up any existing client
                    if self.client:
                        try:
                            if hasattr(self.client, 'close'):
                                self.client.close()
                        except Exception:
                            pass
                        self.client = None
                        
                    self.client = MessageBusClient(
                        host=host,
                        port=port,
                        ssl=False,
                        timeout=3
                    )
                    self.client.run_in_thread()
                    
                    # Verify connection
                    start_time = time.time()
                    connection_verified = False
                    while time.time() - start_time < 3:  # Reduced timeout for reconnect
                        if hasattr(self.client, 'connected') and self.client.connected:
                            self.connected = True
                            self.connection_status = "connected"
                            self.logger.info(f"Reconnected to OVOS messagebus after {reconnect_count} attempts")
                            
                            # Re-register any handlers
                            self._reregister_handlers()
                            
                            # Process any queued messages
                            self._process_message_queue()
                            
                            connection_verified = True
                            break
                        time.sleep(0.1)
                        
                    if connection_verified:
                        # Successfully reconnected
                        continue
                        
                    # Connection timed out
                    self.last_connection_error = "Reconnection timeout"
                    self.connection_status = "timeout"
                    
                    # Clean up client that timed out
                    if self.client:
                        try:
                            if hasattr(self.client, 'close'):
                                self.client.close()
                        except Exception:
                            pass
                        self.client = None
                    
                except Exception as e:
                    self.last_connection_error = str(e)
                    self.logger.debug(f"OVOS reconnection attempt failed: {e}")
                    self.connection_status = "reconnection_error"
                    
                    # Clean up failed connection attempt
                    if self.client:
                        try:
                            if hasattr(self.client, 'close'):
                                self.client.close()
                        except Exception:
                            pass
                        self.client = None
                    
            # After several failed attempts, try reducing reconnection frequency
            if reconnect_count > 5:
                reconnect_delay = min(reconnect_delay * 1.5, 300)  # Gradually increase up to 5 minutes
                
            # Wait before retrying
            time.sleep(reconnect_delay)
            
    def _reregister_handlers(self):
        """Re-register all handlers after reconnection."""
        with self.handler_lock:
            if not self.handler_registry:
                return
                
            self.logger.info(f"Re-registering {len(self.handler_registry)} message handlers")
            
            for message_type, handler in self.handler_registry.items():
                try:
                    self.client.on(message_type, handler)
                except Exception as e:
                    self.logger.error(f"Error re-registering handler for {message_type}: {e}")
                    
    def _process_message_queue(self):
        """Process any messages that were queued during disconnection."""
        with self.message_queue_lock:
            if not self.message_queue:
                return
                
            self.logger.info(f"Processing {len(self.message_queue)} queued messages")
            
            # Process all queued messages
            success_count = 0
            for message_type, data in self.message_queue:
                try:
                    message = Message(message_type, data=data or {})
                    if self.client.emit(message):
                        success_count += 1
                except Exception as e:
                    self.logger.debug(f"Error sending queued message: {e}")
                    
            self.logger.info(f"Processed {success_count}/{len(self.message_queue)} queued messages")
            
            # Clear the queue after processing
            self.message_queue.clear()
    
    def register_handler(self, message_type: str, handler: Callable):
        """
        Register a handler for a specific message type with thread safety.
        
        Args:
            message_type: Type of message to handle
            handler: Callback function for handling the message
            
        Returns:
            Success status
        """
        if not self.is_connected():
            # Store handler for when connection is established
            with self.handler_lock:
                self.handler_registry[message_type] = handler
            return False
            
        try:
            # Register handler with client
            self.client.on(message_type, handler)
            
            # Store in registry for reconnection
            with self.handler_lock:
                self.handler_registry[message_type] = handler
                
            return True
        except Exception as e:
            self.logger.error(f"Failed to register handler for {message_type}: {e}")
            return False
        
    def is_connected(self) -> bool:
        """
        Check if connected to OVOS messagebus.
        
        Returns:
            Connection status
        """
        if not OVOS_AVAILABLE or self.ovos_config.get('disable_ovos', False):
            return False
            
        try:
            return self.client is not None and hasattr(self.client, 'connected') and self.client.connected
        except Exception:
            return False
            
    def get_connection_status(self) -> Dict:
        """
        Get detailed connection status information.
        
        Returns:
            Dictionary with connection status details
        """
        status = {
            "connected": self.connected,
            "status": self.connection_status,
            "attempts": self.connection_attempts,
            "failures": self.connection_failures,
            "last_error": self.last_connection_error,
            "ovos_available": OVOS_AVAILABLE,
            "disabled": self.ovos_config.get('disable_ovos', False),
            "queue_size": len(self.message_queue),
            "registered_handlers": len(self.handler_registry)
        }
        
        # Add basic OVOS info
        if OVOS_AVAILABLE and not self.ovos_config.get('disable_ovos', False):
            status["host"] = self.ovos_config.get('host', 'localhost')
            status["port"] = self.ovos_config.get('port', 8181)
                
        return status
        
    def send_message(self, message_type: str, data: Dict = None) -> bool:
        """
        Send message to OVOS messagebus with improved error handling.
        
        Args:
            message_type: Type of message to send
            data: Message data
            
        Returns:
            Success status
        """
        if not self.is_connected():
            # If not connected, queue the message for later delivery if enabled
            if self.ovos_config.get('queue_messages_when_disconnected', True):
                with self.message_queue_lock:
                    # Check if queue is full
                    if len(self.message_queue) < self.message_queue_limit:
                        self.message_queue.append((message_type, data))
                        self.logger.debug(f"Queued message {message_type} for later delivery")
                        
                        # If queue is getting large, log a warning
                        if len(self.message_queue) > self.message_queue_limit * 0.8:
                            self.logger.warning(f"Message queue is at {len(self.message_queue)}/{self.message_queue_limit}")
                    else:
                        self.logger.warning(f"Message queue full, dropping message: {message_type}")
                
            self.logger.debug(f"Not connected to OVOS - message not sent immediately: {message_type}")
            return False
            
        try:
            message = Message(message_type, data=data or {})
            success = self.client.emit(message)
            
            if not success:
                self.logger.debug(f"Failed to emit message {message_type}")
                
                # Connection might be lost
                if not self.is_connected():
                    self.connected = False
                    self._start_reconnect_thread()
                    
                    # Queue the message if enabled
                    if self.ovos_config.get('queue_messages_when_disconnected', True):
                        with self.message_queue_lock:
                            if len(self.message_queue) < self.message_queue_limit:
                                self.message_queue.append((message_type, data))
                                self.logger.debug(f"Queued message {message_type} after failed emit")
                
            return success
        except Exception as e:
            self.logger.error(f"Failed to send message {message_type}: {e}")
            
            # Connection might be lost
            if not self.is_connected():
                self.connected = False
                self._start_reconnect_thread()
                
            return False
            
    def wait_for_response(self, message_type: str, data: Dict = None, 
                        response_type: str = None, timeout: int = 10) -> Optional[Message]:
        """
        Send a message and wait for a response with improved error handling.
        
        Args:
            message_type: Type of message to send
            data: Message data
            response_type: Expected response type
            timeout: Maximum wait time in seconds
            
        Returns:
            Response message or None if timed out
        """
        if not self.is_connected():
            return None
            
        try:
            message = Message(message_type, data=data or {})
            response = self.client.wait_for_response(
                message, 
                response_type=response_type,
                timeout=timeout
            )
            return response
        except Exception as e:
            self.logger.error(f"Error waiting for response to {message_type}: {e}")
            
            # Check if connection was lost
            if not self.is_connected():
                self.connected = False
                self._start_reconnect_thread()
                
            return None
        
    def send_and_wait(self, message_type: str, data: Dict = None, 
                     response_type: str = None, timeout: int = 10) -> Tuple[bool, Optional[Dict]]:
        """
        Send a message and wait for a response with better error handling.
        
        Args:
            message_type: Type of message to send
            data: Message data
            response_type: Expected response type
            timeout: Maximum wait time in seconds
            
        Returns:
            Tuple of (success, response_data)
        """
        if not self.is_connected():
            return (False, None)
            
        try:
            message = Message(message_type, data=data or {})
            response = self.client.wait_for_response(
                message, 
                response_type=response_type,
                timeout=timeout
            )
            
            if response:
                return (True, response.data)
            return (False, None)
            
        except Exception as e:
            self.logger.error(f"Error in send_and_wait for {message_type}: {e}")
            
            # Check if connection was lost
            if not self.is_connected():
                self.connected = False
                self._start_reconnect_thread()
                
            return (False, {"error": str(e)})
    
    def reset_connection(self):
        """
        Force a connection reset and reconnect attempt.
        
        Returns:
            Success status
        """
        with self.connection_lock:
            if self.client:
                try:
                    if hasattr(self.client, 'close'):
                        self.client.close()
                except Exception as e:
                    self.logger.debug(f"Error closing client: {e}")
                    
            self.client = None
            self.connected = False
            self.connection_status = "reset"
            
            # Reconnect
            self._connect()
            return self.is_connected()
        
    def shutdown(self):
        """Clean shutdown of OVOS client with proper resource management."""
        self.logger.info("Shutting down OVOS client")
        
        # Stop reconnection thread
        self.running = False
        
        # Wait for reconnect thread to exit
        if self.reconnect_thread and self.reconnect_thread.is_alive():
            try:
                self.reconnect_thread.join(timeout=1.0)  # Wait for thread to exit with timeout
            except Exception as e:
                self.logger.debug(f"Error waiting for reconnect thread: {e}")
        
        # Close client with proper thread safety
        with self.connection_lock:
            if self.client:
                try:
                    # Properly disconnect
                    if hasattr(self.client, 'close'):
                        self.client.close()
                except Exception as e:
                    self.logger.debug(f"Error closing OVOS client: {e}")
                    
            self.client = None
            self.connected = False
        
        # Clear message queue
        with self.message_queue_lock:
            queue_size = len(self.message_queue)
            if queue_size > 0:
                self.logger.info(f"Clearing {queue_size} queued messages")
                self.message_queue.clear()
                
        # Clear handler registry
        with self.handler_lock:
            self.handler_registry.clear()
"""
Grace AI System - OVOS Integration

This module integrates Grace AI with OpenVoiceOS through messagebus communication
with improved thread safety and connection handling.
"""

import logging
import threading
import time
import json
import asyncio
from typing import Dict, Optional, List, Any, Callable, Tuple

# Import OVOS components
from .ovos_client import OVOSClient, OVOS_AVAILABLE
from .ovos_handlers import OVOSHandlers
from .ovos_commands import OVOSCommands
from .ovos_message import OVOSMessage


class OVOSInterface:
    """
    OpenVoiceOS integration for the Grace AI system with improved thread safety.
    
    This class coordinates the different components needed for OVOS integration,
    providing a unified interface for interaction with OpenVoiceOS with proper
    resource management and error handling.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the OVOS interface with the provided configuration.
        
        Args:
            config: OVOS configuration
        """
        self.logger = logging.getLogger('grace.ovos')
        self.config = config
        self.ovos_config = config.get('ovos', {})
        
        # Check if OVOS integration is disabled
        if self.ovos_config.get('disable_ovos', False):
            self.logger.info("OVOS integration disabled by configuration")
            self.client = None
            self.message = OVOSMessage()
            self.handlers = None
            self.commands = None
            return
            
        # Initialize components with proper error handling
        self.client = OVOSClient(config)
        self.message = OVOSMessage()
        
        # Only initialize handlers and commands if client is connected
        if self.client.is_connected():
            self.handlers = OVOSHandlers(self.client)
            self.commands = OVOSCommands(self.client)
            
            # Register default handlers
            self.handlers.register_default_handlers()
        else:
            self.handlers = None
            self.commands = None
            
        # Use locks for thread-safe operations
        self.lock = threading.RLock()
        
        # Track OVOS ready state
        self.is_ready = False
        self.setup_complete = threading.Event()
        
        # Start initialization process if connected
        if self.client.is_connected():
            self._initialize()
            
        # Callback registry for external components to receive OVOS events
        # with lock protection for thread safety
        self.callbacks = {}
        self.callback_lock = threading.RLock()
    
    def _initialize(self):
        """Initialize OVOS interface by gathering system information with proper error handling."""
        # Setup ready flag
        self.is_ready = False
        self.setup_complete.clear()
        
        def on_system_ready(message):
            """Handle system ready message with thread safety."""
            with self.lock:
                self.logger.info("OVOS system is ready")
                self.is_ready = True
                self.setup_complete.set()
                
                # Trigger any registered callbacks with proper thread safety
                with self.callback_lock:
                    if 'system_ready' in self.callbacks:
                        for callback in self.callbacks['system_ready']:
                            try:
                                callback(message)
                            except Exception as e:
                                self.logger.error(f"Error in system_ready callback: {e}")
            
        # Register for system ready message
        if self.handlers:
            self.handlers.register_handler('ovos.ready', on_system_ready)
            self.handlers.register_callback('skills_initialized', self._on_skills_initialized)
            
        # Wait for setup completion in background thread
        setup_thread = threading.Thread(
            target=self._wait_for_setup_completion,
            daemon=True,
            name="OVOS-Setup"
        )
        setup_thread.start()
    
    def _wait_for_setup_completion(self):
        """Wait for OVOS setup to complete with timeout."""
        # Wait up to 30 seconds for setup to complete
        if self.setup_complete.wait(timeout=30):
            self.logger.info("OVOS interface initialization complete")
        else:
            self.logger.warning("OVOS interface initialization timed out, continuing without full setup")
            # Set ready state even though we timed out
            with self.lock:
                self.is_ready = True
    
    def _on_skills_initialized(self, message):
        """Handle skills initialized message with proper thread safety."""
        with self.lock:
            self.logger.info("OVOS skills are initialized")
            
            # Update ready status
            if not self.is_ready:
                self.is_ready = True
                self.setup_complete.set()
                
            # Trigger any registered callbacks with proper thread safety
            with self.callback_lock:
                if 'skills_initialized' in self.callbacks:
                    for callback in self.callbacks['skills_initialized']:
                        try:
                            callback(message)
                        except Exception as e:
                            self.logger.error(f"Error in skills_initialized callback: {e}")
    
    def wait_for_ready(self, timeout: float = 10.0) -> bool:
        """
        Wait for OVOS interface to be ready with proper timeout handling.
        
        Args:
            timeout: Maximum wait time in seconds
            
        Returns:
            True if ready, False if timeout
        """
        if not self.client or not self.client.is_connected():
            return False
            
        return self.setup_complete.wait(timeout=timeout)
    
    def is_connected(self) -> bool:
        """
        Check if connected to OVOS messagebus.
        
        Returns:
            Connection status
        """
        return self.client and self.client.is_connected()
    
    def get_connection_status(self) -> Dict:
        """
        Get connection status details.
        
        Returns:
            Dictionary with status information
        """
        if not self.client:
            return {
                "connected": False,
                "status": "client_not_initialized",
                "ovos_available": OVOS_AVAILABLE,
                "disabled": self.ovos_config.get('disable_ovos', False)
            }
            
        return self.client.get_connection_status()
    
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
            return False
            
        return self.client.send_message(message_type, data)
    
    def send_utterance(self, utterance: str) -> bool:
        """
        Send an utterance for intent processing with improved error handling.
        
        Args:
            utterance: Utterance text
            
        Returns:
            Success status
        """
        if not self.is_connected():
            return False
            
        # Create utterance message
        utterance_msg = self.message.format_utterance(utterance)
        
        # Send to messagebus
        return self.client.send_message(utterance_msg.type, utterance_msg.data)
    
    def speak(self, text: str) -> bool:
        """
        Make OVOS speak text.
        
        Args:
            text: Text to speak
            
        Returns:
            Success status
        """
        if not self.is_connected() or not self.commands:
            return False
            
        return self.commands.speak(text)
    
    def execute_command(self, command: Dict) -> bool:
        """
        Execute an OVOS command with improved error handling and thread safety.
        
        Args:
            command: Command dictionary with 'type' and 'data'
            
        Returns:
            Success status
        """
        if not self.is_connected():
            return False
            
        # Create command message
        command_msg = self.message.create_command_message(command)
        if not command_msg:
            return False
            
        # Send to messagebus
        return self.client.send_message(command_msg.type, command_msg.data)
    
    def control_volume(self, action: str, level: int = None) -> Union[bool, Optional[int]]:
        """
        Control OVOS volume.
        
        Args:
            action: Volume action ('get', 'set', 'up', 'down', 'mute', 'unmute')
            level: Volume level for 'set' action
            
        Returns:
            Success status for actions, volume level for 'get'
        """
        if not self.is_connected() or not self.commands:
            return False
            
        if action == 'get':
            return self.commands.get_volume()
        elif action == 'set' and level is not None:
            return self.commands.set_volume(level)
        elif action == 'up':
            return self.commands.volume_up()
        elif action == 'down':
            return self.commands.volume_down()
        elif action == 'mute':
            return self.commands.mute()
        elif action == 'unmute':
            return self.commands.unmute()
        else:
            self.logger.error(f"Invalid volume action: {action}")
            return False
    
    def get_skills(self) -> List[Dict]:
        """
        Get list of available skills.
        
        Returns:
            List of skill information
        """
        if not self.is_connected() or not self.handlers:
            return []
            
        return self.handlers.get_skill_list()
    
    async def get_skills_async(self) -> List[Dict]:
        """
        Get list of available skills asynchronously.
        
        Returns:
            List of skill information
        """
        if not self.is_connected() or not self.handlers:
            return []
            
        # Run in a thread to avoid blocking
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.get_skills)
    
    def register_callback(self, event: str, callback: Callable) -> bool:
        """
        Register a callback for an event with improved thread safety.
        
        Args:
            event: Event name
            callback: Callback function
            
        Returns:
            Success status
        """
        with self.callback_lock:
            # Store callback in local registry for reconnect scenarios
            if event not in self.callbacks:
                self.callbacks[event] = []
            
            # Avoid duplicate callbacks
            if callback not in self.callbacks[event]:
                self.callbacks[event].append(callback)
            
            # Register with handlers if connected
            if self.is_connected() and self.handlers:
                return self.handlers.register_callback(event, callback)
                
            return True
    
    def handle_fallback_intent(self, utterance: str) -> Dict:
        """
        Handle intent processing when OVOS is not connected.
        
        Args:
            utterance: User utterance
            
        Returns:
            Intent processing result
        """
        # Simple fallback intent processing when OVOS is unavailable
        # This provides basic functionality without OVOS
        
        # Check for simple commands
        utterance = utterance.lower().strip()
        
        if any(word in utterance for word in ['time', 'clock', 'hour']):
            import datetime
            now = datetime.datetime.now()
            return {
                "handled": True,
                "intent_type": "time.inquiry",
                "response": f"It's {now.strftime('%I:%M %p')}."
            }
            
        if any(word in utterance for word in ['date', 'day', 'month', 'year']):
            import datetime
            now = datetime.datetime.now()
            return {
                "handled": True,
                "intent_type": "date.inquiry",
                "response": f"Today is {now.strftime('%A, %B %d, %Y')}."
            }
            
        # Weather, news, etc. would require external APIs and are not implemented here
        
        return {
            "handled": False,
            "intent_type": "fallback",
            "response": None
        }
    
    def test_connection(self) -> Tuple[bool, str]:
        """
        Test the connection to OVOS with detailed diagnostics.
        
        Returns:
            Tuple of (success, message)
        """
        if not OVOS_AVAILABLE:
            return (False, "OVOS client library not available")
            
        if self.ovos_config.get('disable_ovos', False):
            return (False, "OVOS integration disabled in configuration")
            
        if not self.is_connected():
            status = self.get_connection_status()
            
            # Detailed diagnostics
            message_parts = [
                f"OVOS connection status: {status.get('status', 'unknown')}",
                f"Last error: {status.get('last_error', 'none')}"
            ]
                
            return (False, ". ".join(message_parts))
            
        # We're connected - try a simple message to verify
        try:
            # Send a ping message
            success = self.send_message("ping")
            if success:
                return (True, "Connected and able to send messages")
            else:
                return (False, "Connected but failed to send test message")
        except Exception as e:
            return (False, f"Connection test failed: {e}")
    
    async def test_connection_async(self) -> Tuple[bool, str]:
        """
        Test the connection to OVOS with detailed diagnostics asynchronously.
        
        Returns:
            Tuple of (success, message)
        """
        # Run in a thread to avoid blocking
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.test_connection)
    
    def reset_connection(self) -> bool:
        """
        Force a connection reset and reconnect attempt.
        
        Returns:
            Success status
        """
        if not self.client:
            return False
            
        success = self.client.reset_connection()
        
        # If reconnected, reinitialize handlers and commands
        if success:
            with self.lock:
                self.handlers = OVOSHandlers(self.client)
                self.commands = OVOSCommands(self.client)
                
                # Register default handlers
                self.handlers.register_default_handlers()
                
                # Register stored callbacks with thread safety
                with self.callback_lock:
                    for event, callbacks in self.callbacks.items():
                        for callback in callbacks:
                            if self.handlers:
                                self.handlers.register_callback(event, callback)
                
                # Restart initialization
                self._initialize()
            
        return success
    
    async def reset_connection_async(self) -> bool:
        """
        Force a connection reset and reconnect attempt asynchronously.
        
        Returns:
            Success status
        """
        # Run in a thread to avoid blocking
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.reset_connection)
    
    def shutdown(self):
        """Clean shutdown of OVOS interface with proper resource management."""
        self.logger.info("Shutting down OVOS interface")
        
        with self.lock:
            # Clear event
            self.setup_complete.clear()
            
            # Set not ready
            self.is_ready = False
            
            # Clean shutdown of client
            if self.client:
                self.client.shutdown()
                
            # Clear handlers and commands
            self.handlers = None
            self.commands = None
            
            # Clear callbacks for good measure
            with self.callback_lock:
                self.callbacks.clear()
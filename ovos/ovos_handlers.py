"""
Grace AI System - OVOS Handlers Module

This module implements handlers for OpenVoiceOS messagebus messages.
"""

import logging
import asyncio
import time
from typing import Dict, Any, Optional, Callable, List, Union

# Import OVOS client for type hints
from .ovos_client import OVOSClient, Message


class OVOSHandlers:
    """
    Handlers for OpenVoiceOS messagebus messages.
    
    This class manages handlers for different message types received
    from the OpenVoiceOS messagebus.
    """
    
    def __init__(self, ovos_client: OVOSClient):
        """
        Initialize OVOS handlers with the provided client.
        
        Args:
            ovos_client: OVOS client
        """
        self.logger = logging.getLogger('grace.ovos.handlers')
        self.client = ovos_client
        self.handlers = {}
        self.callback_registry = {}
        self.skill_registry = {}
        
        # Track latest message data for debugging
        self.last_messages = {}
        self.message_history = []
        self.max_message_history = 100
        
    def register_default_handlers(self):
        """Register default message handlers."""
        if not self.client or not self.client.is_connected():
            return
            
        try:
            # Register for skill status messages
            self.register_handler('ovos.skills.initialized', self._handle_skills_initialized)
            
            # Register for intent failure messages
            self.register_handler('ovos.skills.fallback', self._handle_intent_failure)
            
            # Register for system status messages
            self.register_handler('system.heartbeat', self._handle_heartbeat)
            
            # Register for question response
            self.register_handler('question:query.response', self._handle_question_response)
            
            # Register for error messages
            self.register_handler('recognizer_loop.error', self._handle_recognizer_error)
            
            # Register for volume change feedback
            self.register_handler('ovos.volume.get.response', self._handle_volume_response)
            
            # Register for device settings
            self.register_handler('ovos.device.settings', self._handle_device_settings)
            
            # Register for speech start and end
            self.register_handler('recognizer_loop:audio_output_start', self._handle_audio_start)
            self.register_handler('recognizer_loop:audio_output_end', self._handle_audio_end)
            
            # Register for utterance handling
            self.register_handler('recognizer_loop:utterance', self._handle_utterance)
            
            # Register for intent handling success
            self.register_handler('recognizer_loop:utterance.handled', self._handle_utterance_handled)
            
            self.logger.info("Registered default message handlers")
        except Exception as e:
            self.logger.error(f"Failed to register default handlers: {e}")
    
    def _handle_skills_initialized(self, message):
        """Handle skills initialized message."""
        self.logger.info("OVOS skills initialized")
        
        # Get available skills
        self.client.send_message('skillmanager.list')
        
        # Track message data
        self.last_messages['skills_initialized'] = {
            'timestamp': message.context.get('timestamp', time.time()),
            'data': message.data
        }
        self._add_to_message_history('skills_initialized', message)
        
        # Trigger callback if registered
        if 'skills_initialized' in self.callback_registry:
            for callback in self.callback_registry.get('skills_initialized', []):
                try:
                    callback(message)
                except Exception as e:
                    self.logger.error(f"Error in callback for skills_initialized: {e}")
    
    def _handle_intent_failure(self, message):
        """Handle intent failure message."""
        utterance = message.data.get('utterance', '')
        self.logger.debug(f"Intent failure for: {utterance}")
        
        # Track message data
        self.last_messages['intent_failure'] = {
            'timestamp': message.context.get('timestamp', time.time()),
            'utterance': utterance
        }
        self._add_to_message_history('intent_failure', message)
        
        # Trigger callback if registered
        if 'intent_failure' in self.callback_registry:
            for callback in self.callback_registry.get('intent_failure', []):
                try:
                    callback(message)
                except Exception as e:
                    self.logger.error(f"Error in callback for intent_failure: {e}")
    
    def _handle_heartbeat(self, message):
        """Handle system heartbeat message."""
        # This confirms the system is still connected and responsive
        self.last_messages['heartbeat'] = {
            'timestamp': message.context.get('timestamp', time.time())
        }
        
        # Trigger callback if registered
        if 'heartbeat' in self.callback_registry:
            for callback in self.callback_registry.get('heartbeat', []):
                try:
                    callback(message)
                except Exception as e:
                    self.logger.error(f"Error in callback for heartbeat: {e}")
        
    def _handle_question_response(self, message):
        """Handle responses to questions."""
        self.logger.debug(f"Received question response")
        
        # Track message data
        self.last_messages['question_response'] = {
            'timestamp': message.context.get('timestamp', time.time()),
            'data': message.data
        }
        self._add_to_message_history('question_response', message)
        
        # Trigger callback if registered
        if 'question_response' in self.callback_registry:
            for callback in self.callback_registry.get('question_response', []):
                try:
                    callback(message)
                except Exception as e:
                    self.logger.error(f"Error in callback for question_response: {e}")
    
    def _handle_recognizer_error(self, message):
        """Handle recognizer errors."""
        error = message.data.get('error', '')
        self.logger.warning(f"Recognizer error: {error}")
        
        # Track message data
        self.last_messages['recognizer_error'] = {
            'timestamp': message.context.get('timestamp', time.time()),
            'error': error
        }
        self._add_to_message_history('recognizer_error', message)
        
        # Trigger callback if registered
        if 'recognizer_error' in self.callback_registry:
            for callback in self.callback_registry.get('recognizer_error', []):
                try:
                    callback(message)
                except Exception as e:
                    self.logger.error(f"Error in callback for recognizer_error: {e}")
    
    def _handle_volume_response(self, message):
        """Handle volume response message."""
        # Store volume level
        volume = message.data.get('volume', 0)
        self.logger.debug(f"Current volume is {volume}")
        
        # Track message data
        self.last_messages['volume'] = {
            'timestamp': message.context.get('timestamp', time.time()),
            'volume': volume
        }
        self._add_to_message_history('volume_response', message)
        
        # Trigger callback if registered
        if 'volume_response' in self.callback_registry:
            for callback in self.callback_registry.get('volume_response', []):
                try:
                    callback(message)
                except Exception as e:
                    self.logger.error(f"Error in callback for volume_response: {e}")
                
    def _handle_device_settings(self, message):
        """Handle device settings message."""
        settings = message.data.get('settings', {})
        self.logger.debug("Received device settings")
        
        # Track message data
        self.last_messages['device_settings'] = {
            'timestamp': message.context.get('timestamp', time.time()),
            'settings': settings
        }
        self._add_to_message_history('device_settings', message)
        
        # Trigger callback if registered
        if 'device_settings' in self.callback_registry:
            for callback in self.callback_registry.get('device_settings', []):
                try:
                    callback(message)
                except Exception as e:
                    self.logger.error(f"Error in callback for device_settings: {e}")
    
    def _handle_audio_start(self, message):
        """Handle audio output start message."""
        self.logger.debug("Audio output started")
        
        # Track message data
        self.last_messages['audio_start'] = {
            'timestamp': message.context.get('timestamp', time.time())
        }
        
        # Trigger callback if registered
        if 'audio_start' in self.callback_registry:
            for callback in self.callback_registry.get('audio_start', []):
                try:
                    callback(message)
                except Exception as e:
                    self.logger.error(f"Error in callback for audio_start: {e}")
    
    def _handle_audio_end(self, message):
        """Handle audio output end message."""
        self.logger.debug("Audio output ended")
        
        # Track message data
        self.last_messages['audio_end'] = {
            'timestamp': message.context.get('timestamp', time.time())
        }
        
        # Trigger callback if registered
        if 'audio_end' in self.callback_registry:
            for callback in self.callback_registry.get('audio_end', []):
                try:
                    callback(message)
                except Exception as e:
                    self.logger.error(f"Error in callback for audio_end: {e}")
    
    def _handle_utterance(self, message):
        """Handle new utterance message."""
        utterances = message.data.get('utterances', [])
        if utterances:
            utterance = utterances[0]
            self.logger.debug(f"New utterance: {utterance}")
            
            # Track message data
            self.last_messages['utterance'] = {
                'timestamp': message.context.get('timestamp', time.time()),
                'utterance': utterance
            }
            self._add_to_message_history('utterance', message)
            
            # Trigger callback if registered
            if 'utterance' in self.callback_registry:
                for callback in self.callback_registry.get('utterance', []):
                    try:
                        callback(message)
                    except Exception as e:
                        self.logger.error(f"Error in callback for utterance: {e}")
    
    def _handle_utterance_handled(self, message):
        """Handle utterance handled message."""
        intent = message.data.get('handler', 'unknown')
        self.logger.debug(f"Utterance handled by: {intent}")
        
        # Track message data
        self.last_messages['utterance_handled'] = {
            'timestamp': message.context.get('timestamp', time.time()),
            'handler': intent
        }
        self._add_to_message_history('utterance_handled', message)
        
        # Trigger callback if registered
        if 'utterance_handled' in self.callback_registry:
            for callback in self.callback_registry.get('utterance_handled', []):
                try:
                    callback(message)
                except Exception as e:
                    self.logger.error(f"Error in callback for utterance_handled: {e}")
    
    def _add_to_message_history(self, message_type: str, message):
        """Add message to history with proper truncation."""
        try:
            # Create history entry with relevant information
            history_entry = {
                'type': message_type,
                'timestamp': message.context.get('timestamp', time.time()),
                'data': message.data
            }
            
            # Add to history
            self.message_history.append(history_entry)
            
            # Truncate if needed
            if len(self.message_history) > self.max_message_history:
                self.message_history = self.message_history[-self.max_message_history:]
        except Exception as e:
            self.logger.error(f"Error adding message to history: {e}")
    
    def register_callback(self, event_name: str, callback: Callable) -> bool:
        """
        Register a callback for a specific event.
        
        Args:
            event_name: Name of the event
            callback: Callback function
            
        Returns:
            Success status
        """
        try:
            if event_name not in self.callback_registry:
                self.callback_registry[event_name] = []
                
            # Avoid duplicate callbacks
            if callback not in self.callback_registry[event_name]:
                self.callback_registry[event_name].append(callback)
            return True
        except Exception as e:
            self.logger.error(f"Failed to register callback for {event_name}: {e}")
            return False
    
    def deregister_callback(self, event_name: str, callback: Callable) -> bool:
        """
        Deregister a callback for a specific event.
        
        Args:
            event_name: Name of the event
            callback: Callback function
            
        Returns:
            Success status
        """
        try:
            if event_name in self.callback_registry:
                if callback in self.callback_registry[event_name]:
                    self.callback_registry[event_name].remove(callback)
                    return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to deregister callback for {event_name}: {e}")
            return False
    
    def register_handler(self, message_type: str, handler: Callable) -> bool:
        """
        Register a handler for a specific message type.
        
        Args:
            message_type: Type of message to handle
            handler: Handler function
            
        Returns:
            Success status
        """
        if not self.client or not self.client.is_connected():
            self.logger.debug(f"Not connected to OVOS - handler not registered: {message_type}")
            return False
            
        try:
            success = self.client.register_handler(message_type, handler)
            if success:
                self.handlers[message_type] = handler
            return success
        except Exception as e:
            self.logger.error(f"Failed to register handler for {message_type}: {e}")
            return False
    
    def get_skill_list(self) -> List[Dict]:
        """
        Get list of available skills.
        
        Returns:
            List of skill information dictionaries
        """
        if not self.client or not self.client.is_connected():
            return []
            
        try:
            success, data = self.client.send_and_wait(
                'skillmanager.list', 
                response_type='skillmanager.list.response', 
                timeout=5
            )
            
            if success and data and 'skills' in data:
                # Update skill registry
                self.skill_registry = {skill['name']: skill for skill in data['skills']}
                return data['skills']
            return []
        except Exception as e:
            self.logger.error(f"Failed to get skill list: {e}")
            return []
    
    async def get_skill_list_async(self) -> List[Dict]:
        """
        Get list of available skills asynchronously.
        
        Returns:
            List of skill information dictionaries
        """
        # Run in a thread pool
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.get_skill_list)
    
    def execute_skill_command(self, skill_id: str, command: str, data: Dict = None) -> bool:
        """
        Execute a specific skill command.
        
        Args:
            skill_id: Skill identifier
            command: Command to execute
            data: Command data
            
        Returns:
            Success status
        """
        if not self.client or not self.client.is_connected():
            return False
            
        message_type = f"{skill_id}.{command}"
        return self.client.send_message(message_type, data or {})
    
    def get_last_message(self, message_type: str) -> Union[Dict, None]:
        """
        Get the last message of a specific type.
        
        Args:
            message_type: Type of message to retrieve
            
        Returns:
            Message data or None if not found
        """
        return self.last_messages.get(message_type)
    
    def get_recent_messages(self, count: int = 10) -> List[Dict]:
        """
        Get recent messages from history.
        
        Args:
            count: Number of messages to retrieve
            
        Returns:
            List of message data
        """
        return self.message_history[-count:] if self.message_history else []

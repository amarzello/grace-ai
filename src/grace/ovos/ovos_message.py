"""
Grace AI System - OVOS Message Module

This module implements message handling for OpenVoiceOS integration.
"""

import logging
import json
import time
import re
from typing import Dict, Any, Optional, Union, List

# Try to import OVOS dependencies
try:
    from ovos_bus_client import Message
    OVOS_AVAILABLE = True
except ImportError:
    OVOS_AVAILABLE = False
    # Create fallback Message class
    class Message:
        def __init__(self, message_type, data=None, context=None):
            self.type = message_type
            self.data = data or {}
            self.context = context or {}


class OVOSMessage:
    """
    OpenVoiceOS message handling and formatting.
    
    This class provides utilities for creating, formatting and parsing 
    messages for the OpenVoiceOS messagebus.
    """
    
    def __init__(self):
        """Initialize the OVOS message handler."""
        self.logger = logging.getLogger('grace.ovos.message')
        
        # Cache for command templates
        self.command_templates = {}
        
        # Initialize command templates
        self._init_command_templates()
    
    def _init_command_templates(self):
        """Initialize command templates for common operations."""
        # Volume commands
        self.command_templates["volume.set"] = {
            "type": "ovos.volume.set",
            "data_template": {"level": "{level}"}
        }
        self.command_templates["volume.get"] = {
            "type": "ovos.volume.get",
            "data_template": {}
        }
        self.command_templates["volume.increase"] = {
            "type": "ovos.volume.increase",
            "data_template": {}
        }
        self.command_templates["volume.decrease"] = {
            "type": "ovos.volume.decrease",
            "data_template": {}
        }
        
        # Media commands
        self.command_templates["media.play"] = {
            "type": "ovos.audio.service.play",
            "data_template": {"uri": "{uri}"}
        }
        self.command_templates["media.pause"] = {
            "type": "ovos.audio.service.pause",
            "data_template": {}
        }
        self.command_templates["media.resume"] = {
            "type": "ovos.audio.service.resume",
            "data_template": {}
        }
        
        # System commands
        self.command_templates["system.reboot"] = {
            "type": "system.reboot",
            "data_template": {}
        }
        self.command_templates["system.shutdown"] = {
            "type": "system.shutdown",
            "data_template": {}
        }
    
    def create_message(self, message_type: str, data: Dict = None, context: Dict = None) -> Message:
        """
        Create a new Message object.
        
        Args:
            message_type: Type of message
            data: Message data
            context: Message context
            
        Returns:
            Message object
        """
        # Ensure data is a dictionary
        if data is None:
            data = {}
            
        # Add timestamp to context if not provided
        if context is None:
            context = {"timestamp": time.time()}
        elif "timestamp" not in context:
            context["timestamp"] = time.time()
            
        return Message(message_type, data=data, context=context)
    
    def format_utterance(self, text: str) -> Message:
        """
        Format an utterance message.
        
        Args:
            text: Utterance text
            
        Returns:
            Formatted message
        """
        # Sanitize input
        if not text:
            text = ""
            
        utterance = text.strip()
        return Message('recognizer_loop:utterance', {'utterances': [utterance]})
    
    def format_speak(self, text: str) -> Message:
        """
        Format a speak message.
        
        Args:
            text: Text to speak
            
        Returns:
            Formatted message
        """
        # Sanitize input
        if not text:
            text = ""
            
        return Message('speak', {'utterance': text.strip()})
    
    def create_command_message(self, command: Dict) -> Optional[Message]:
        """
        Create a message from a command dictionary.
        
        Args:
            command: Command dictionary with 'type' and optionally 'data'
            
        Returns:
            Message object or None if invalid
        """
        if not command or not isinstance(command, dict):
            return None
            
        # Extract command fields
        message_type = command.get('type')
        if not message_type:
            self.logger.error("Missing message type in command")
            return None
            
        message_data = command.get('data', {})
        
        # Create message
        return Message(message_type, data=message_data)
    
    def create_command_from_template(self, template_name: str, **kwargs) -> Optional[Message]:
        """
        Create a message from a predefined template.
        
        Args:
            template_name: Name of the template to use
            **kwargs: Parameters to fill in the template
            
        Returns:
            Message object or None if template not found
        """
        if template_name not in self.command_templates:
            self.logger.error(f"Template not found: {template_name}")
            return None
            
        template = self.command_templates[template_name]
        message_type = template["type"]
        data_template = template["data_template"].copy()
        
        # Fill in template parameters
        data = {}
        for key, value in data_template.items():
            if isinstance(value, str) and "{" in value and "}" in value:
                # Extract parameter name
                param_name = re.search(r'\{(.*?)\}', value).group(1)
                if param_name in kwargs:
                    data[key] = kwargs[param_name]
                else:
                    self.logger.error(f"Missing required parameter for template {template_name}: {param_name}")
                    return None
            else:
                data[key] = value
                
        # Create message
        return Message(message_type, data=data)
    
    def parse_response(self, message: Message) -> Dict:
        """
        Parse a response message.
        
        Args:
            message: Message to parse
            
        Returns:
            Parsed data
        """
        if not message:
            return {}
            
        try:
            result = {
                'type': message.type,
                'data': message.data
            }
            
            if hasattr(message, 'context') and message.context:
                result['context'] = message.context
                
            return result
        except Exception as e:
            self.logger.error(f"Error parsing message: {e}")
            return {'error': str(e)}
    
    def extract_utterance(self, message: Message) -> Optional[str]:
        """
        Extract utterance from message.
        
        Args:
            message: Message to extract from
            
        Returns:
            Utterance text or None
        """
        if not message or not hasattr(message, 'data'):
            return None
            
        # Handle different message formats
        if 'utterance' in message.data:
            return message.data['utterance']
        elif 'utterances' in message.data and message.data['utterances']:
            return message.data['utterances'][0]
        return None
    
    def json_to_message(self, json_data: Union[str, Dict]) -> Optional[Message]:
        """
        Convert JSON to Message object.
        
        Args:
            json_data: JSON string or dictionary
            
        Returns:
            Message object or None if conversion fails
        """
        try:
            # Parse JSON if string
            if isinstance(json_data, str):
                data = json.loads(json_data)
            else:
                data = json_data
                
            # Extract fields
            message_type = data.get('type')
            if not message_type:
                self.logger.error("Missing message type in JSON data")
                return None
                
            message_data = data.get('data', {})
            message_context = data.get('context', {})
            
            # Create message
            return Message(message_type, data=message_data, context=message_context)
        except Exception as e:
            self.logger.error(f"Error converting JSON to message: {e}")
            return None
    
    def message_to_json(self, message: Message) -> str:
        """
        Convert Message object to JSON string.
        
        Args:
            message: Message to convert
            
        Returns:
            JSON string
        """
        try:
            data = {
                'type': message.type,
                'data': message.data
            }
            
            if hasattr(message, 'context') and message.context:
                data['context'] = message.context
                
            return json.dumps(data)
        except Exception as e:
            self.logger.error(f"Error converting message to JSON: {e}")
            return json.dumps({'error': str(e)})
    
    def validate_message(self, message: Message) -> bool:
        """
        Validate a message.
        
        Args:
            message: Message to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not message:
            return False
            
        # Check minimum required fields
        if not hasattr(message, 'type') or not message.type:
            return False
            
        # Check data is a dictionary
        if hasattr(message, 'data') and not isinstance(message.data, dict):
            return False
            
        # Check context is a dictionary if present
        if hasattr(message, 'context') and message.context and not isinstance(message.context, dict):
            return False
            
        return True
    
    def create_response_message(self, request_message: Message, response_data: Dict) -> Message:
        """
        Create a response message for a request.
        
        Args:
            request_message: Original request message
            response_data: Response data
            
        Returns:
            Response message
        """
        # Determine response type based on request type
        response_type = request_message.type
        
        # For specific request types, use a different response type
        if request_message.type.endswith('.get'):
            response_type = f"{request_message.type}.response"
        elif request_message.type.endswith('.request'):
            response_type = request_message.type.replace('.request', '.response')
            
        # Create context from request
        context = {}
        if hasattr(request_message, 'context') and request_message.context:
            context = request_message.context.copy()
            
        # Update timestamp
        context['timestamp'] = time.time()
        
        # Create response message
        return Message(response_type, data=response_data, context=context)

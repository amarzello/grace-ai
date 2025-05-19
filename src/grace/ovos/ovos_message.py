"""
Grace AI System - OVOS Message Module

This module implements message handling for OpenVoiceOS integration with improved
error handling and robust parsing.
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
    # Create fallback Message class with better compatibility
    class Message:
        def __init__(self, message_type, data=None, context=None):
            self.type = message_type
            self.data = data or {}
            self.context = context or {}
            
        def response(self, data=None):
            """Create a response message."""
            response_type = f"{self.type}.response"
            response_data = data or {}
            response_context = self.context.copy() if hasattr(self, 'context') else {}
            return Message(response_type, response_data, response_context)


class OVOSMessage:
    """
    OpenVoiceOS message handling and formatting.
    
    This class provides utilities for creating, formatting and parsing 
    messages for the OpenVoiceOS messagebus with improved error handling
    and fallback mechanisms.
    """
    
    def __init__(self):
        """Initialize the OVOS message handler."""
        self.logger = logging.getLogger('grace.ovos.message')
        
        # Cache for command templates with improved structure
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
        self.command_templates["volume.mute"] = {
            "type": "ovos.volume.mute",
            "data_template": {}
        }
        self.command_templates["volume.unmute"] = {
            "type": "ovos.volume.unmute",
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
        self.command_templates["media.stop"] = {
            "type": "ovos.audio.service.stop",
            "data_template": {}
        }
        self.command_templates["media.next"] = {
            "type": "ovos.audio.service.next",
            "data_template": {}
        }
        self.command_templates["media.prev"] = {
            "type": "ovos.audio.service.prev",
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
        self.command_templates["system.restart_services"] = {
            "type": "system.ovos.service.restart",
            "data_template": {}
        }
        
        # Home Assistant commands
        self.command_templates["ha.turn_on"] = {
            "type": "ovos.phal.plugin.homeassistant.device.turn_on",
            "data_template": {"device_id": "{device_id}"}
        }
        self.command_templates["ha.turn_off"] = {
            "type": "ovos.phal.plugin.homeassistant.device.turn_off",
            "data_template": {"device_id": "{device_id}"}
        }
        self.command_templates["ha.get_devices"] = {
            "type": "ovos.phal.plugin.homeassistant.get.devices",
            "data_template": {}
        }
    
    def create_message(self, message_type: str, data: Dict = None, context: Dict = None) -> Message:
        """
        Create a new Message object with improved validation.
        
        Args:
            message_type: Type of message
            data: Message data
            context: Message context
            
        Returns:
            Message object
        """
        # Validate message type
        if not message_type:
            message_type = "grace.unknown"
            self.logger.warning("Creating message with empty message type")
        
        # Ensure data is a dictionary
        if data is None:
            data = {}
        elif not isinstance(data, dict):
            self.logger.warning(f"Data is not a dictionary, converting: {type(data)}")
            try:
                # Try to convert to dict if it's a string
                if isinstance(data, str):
                    data = json.loads(data)
                else:
                    # Fall back to empty dict if conversion fails
                    data = {"value": str(data)}
            except Exception:
                data = {"value": str(data)}
            
        # Add timestamp to context if not provided
        if context is None:
            context = {"timestamp": time.time()}
        elif "timestamp" not in context:
            context["timestamp"] = time.time()
            
        return Message(message_type, data=data, context=context)
    
    def format_utterance(self, text: str) -> Message:
        """
        Format an utterance message with proper validation.
        
        Args:
            text: Utterance text
            
        Returns:
            Formatted message
        """
        # Sanitize input
        if not text:
            text = ""
            
        # Clean and normalize text
        utterance = text.strip()
        
        # Ensure maximum length to avoid messaging issues
        if len(utterance) > 500:
            self.logger.warning(f"Truncating long utterance: {len(utterance)} chars")
            utterance = utterance[:497] + "..."
            
        return Message('recognizer_loop:utterance', {'utterances': [utterance]})
    
    def format_speak(self, text: str) -> Message:
        """
        Format a speak message with proper validation.
        
        Args:
            text: Text to speak
            
        Returns:
            Formatted message
        """
        # Sanitize input
        if not text:
            text = ""
            
        # Clean text
        text = text.strip()
        
        # Ensure maximum length
        if len(text) > 1000:
            self.logger.warning(f"Truncating long speak text: {len(text)} chars")
            text = text[:997] + "..."
            
        return Message('speak', {'utterance': text})
    
    def create_command_message(self, command: Dict) -> Optional[Message]:
        """
        Create a message from a command dictionary with improved validation.
        
        Args:
            command: Command dictionary with 'type' and optionally 'data'
            
        Returns:
            Message object or None if invalid
        """
        if not command or not isinstance(command, dict):
            self.logger.error(f"Invalid command format: {type(command)}")
            return None
            
        # Extract command fields
        message_type = command.get('type')
        if not message_type:
            self.logger.error("Missing message type in command")
            return None
            
        message_data = command.get('data', {})
        
        # Validate data is a dictionary
        if message_data and not isinstance(message_data, dict):
            self.logger.warning(f"Command data is not a dictionary: {type(message_data)}")
            try:
                # Try to parse if it's a string
                if isinstance(message_data, str):
                    message_data = json.loads(message_data)
                else:
                    message_data = {"value": str(message_data)}
            except Exception:
                message_data = {"value": str(message_data)}
        
        # Create message
        return Message(message_type, data=message_data)
    
    def create_command_from_template(self, template_name: str, **kwargs) -> Optional[Message]:
        """
        Create a message from a predefined template with improved error handling.
        
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
        
        # Fill in template parameters with better validation
        data = {}
        for key, value in data_template.items():
            if isinstance(value, str) and "{" in value and "}" in value:
                # Extract parameter name
                param_match = re.search(r'\{(.*?)\}', value)
                if param_match:
                    param_name = param_match.group(1)
                    if param_name in kwargs:
                        # Replace with provided parameter
                        param_value = kwargs[param_name]
                        # Handle non-string values
                        if isinstance(param_value, (int, float, bool)):
                            data[key] = param_value
                        else:
                            data[key] = str(param_value)
                    else:
                        self.logger.error(f"Missing required parameter for template {template_name}: {param_name}")
                        return None
            else:
                data[key] = value
                
        # Create message
        return Message(message_type, data=data)
    
    def parse_response(self, message: Message) -> Dict:
        """
        Parse a response message with improved error handling.
        
        Args:
            message: Message to parse
            
        Returns:
            Parsed data
        """
        if not message:
            return {"error": "No message provided"}
            
        try:
            result = {
                'type': message.type,
                'data': message.data if hasattr(message, 'data') else {}
            }
            
            if hasattr(message, 'context') and message.context:
                result['context'] = message.context
                
            return result
        except Exception as e:
            self.logger.error(f"Error parsing message: {e}")
            return {'error': str(e)}
    
    def extract_utterance(self, message: Message) -> Optional[str]:
        """
        Extract utterance from message with robust error handling.
        
        Args:
            message: Message to extract from
            
        Returns:
            Utterance text or None
        """
        if not message:
            return None
            
        try:
            # Handle different message formats
            if hasattr(message, 'data'):
                data = message.data
                if 'utterance' in data:
                    return data['utterance']
                elif 'utterances' in data and data['utterances']:
                    if isinstance(data['utterances'], list) and data['utterances']:
                        return data['utterances'][0]
                    elif isinstance(data['utterances'], str):
                        return data['utterances']
            return None
        except Exception as e:
            self.logger.error(f"Error extracting utterance: {e}")
            return None
    
    def json_to_message(self, json_data: Union[str, Dict]) -> Optional[Message]:
        """
        Convert JSON to Message object with improved validation and error handling.
        
        Args:
            json_data: JSON string or dictionary
            
        Returns:
            Message object or None if conversion fails
        """
        try:
            # Parse JSON if string
            if isinstance(json_data, str):
                try:
                    data = json.loads(json_data)
                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid JSON format: {e}")
                    return None
            else:
                data = json_data
                
            # Validate data structure
            if not isinstance(data, dict):
                self.logger.error(f"JSON data is not a dictionary: {type(data)}")
                return None
                
            # Extract fields
            message_type = data.get('type')
            if not message_type:
                self.logger.error("Missing message type in JSON data")
                return None
                
            message_data = data.get('data', {})
            message_context = data.get('context', {})
            
            # Validate data and context are dictionaries
            if message_data and not isinstance(message_data, dict):
                self.logger.warning("Message data is not a dictionary, converting")
                message_data = {"value": str(message_data)}
                
            if message_context and not isinstance(message_context, dict):
                self.logger.warning("Message context is not a dictionary, converting")
                message_context = {"value": str(message_context)}
            
            # Create message
            return Message(message_type, data=message_data, context=message_context)
        except Exception as e:
            self.logger.error(f"Error converting JSON to message: {e}")
            return None
    
    def message_to_json(self, message: Message) -> str:
        """
        Convert Message object to JSON string with improved error handling.
        
        Args:
            message: Message to convert
            
        Returns:
            JSON string
        """
        try:
            # Extract message components with validation
            if not message:
                return json.dumps({"error": "No message provided"})
                
            message_type = message.type if hasattr(message, 'type') else "unknown"
            
            # Ensure data is a dictionary
            message_data = {}
            if hasattr(message, 'data'):
                if isinstance(message.data, dict):
                    message_data = message.data
                else:
                    message_data = {"value": str(message.data)}
            
            # Ensure context is a dictionary
            message_context = {}
            if hasattr(message, 'context') and message.context:
                if isinstance(message.context, dict):
                    message_context = message.context
                else:
                    message_context = {"value": str(message.context)}
            
            # Build data structure
            data = {
                'type': message_type,
                'data': message_data
            }
            
            if message_context:
                data['context'] = message_context
                
            # Convert to JSON with error handling
            return json.dumps(data)
        except Exception as e:
            self.logger.error(f"Error converting message to JSON: {e}")
            return json.dumps({'error': str(e)})
    
    def validate_message(self, message: Message) -> bool:
        """
        Validate a message with detailed checks.
        
        Args:
            message: Message to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not message:
            return False
            
        # Check minimum required fields
        if not hasattr(message, 'type') or not message.type:
            self.logger.debug("Invalid message: missing type")
            return False
            
        # Check data is a dictionary
        if hasattr(message, 'data') and not isinstance(message.data, dict):
            self.logger.debug("Invalid message: data is not a dictionary")
            return False
            
        # Check context is a dictionary if present
        if hasattr(message, 'context') and message.context and not isinstance(message.context, dict):
            self.logger.debug("Invalid message: context is not a dictionary")
            return False
            
        return True
    
    def create_response_message(self, request_message: Message, response_data: Dict) -> Message:
        """
        Create a response message for a request with improved validation.
        
        Args:
            request_message: Original request message
            response_data: Response data
            
        Returns:
            Response message
        """
        if not request_message:
            self.logger.warning("Creating response for empty request message")
            return Message("grace.unknown.response", data=response_data)
            
        # Determine response type based on request type
        response_type = request_message.type
        
        # For specific request types, use a different response type
        if response_type.endswith('.get'):
            response_type = f"{response_type}.response"
        elif response_type.endswith('.request'):
            response_type = response_type.replace('.request', '.response')
        else:
            # Default to adding .response suffix
            response_type = f"{response_type}.response"
            
        # Create context from request
        context = {}
        if hasattr(request_message, 'context') and request_message.context:
            if isinstance(request_message.context, dict):
                context = request_message.context.copy()
            else:
                self.logger.warning("Request context is not a dictionary")
                context = {"original_context": str(request_message.context)}
            
        # Update timestamp
        context['timestamp'] = time.time()
        
        # Ensure response_data is a dictionary
        if not isinstance(response_data, dict):
            self.logger.warning("Response data is not a dictionary, converting")
            response_data = {"value": str(response_data)}
        
        # Create response message
        return Message(response_type, data=response_data, context=context)
    
    def extract_command_from_text(self, text: str) -> Optional[Dict]:
        """
        Extract command information from natural language text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Command dictionary or None if no command found
        """
        if not text:
            return None
            
        # Look for common command patterns
        # Volume commands
        if re.search(r'(set|change)\s+volume\s+to\s+(\d+)', text, re.IGNORECASE):
            match = re.search(r'(set|change)\s+volume\s+to\s+(\d+)', text, re.IGNORECASE)
            level = int(match.group(2))
            # Ensure volume is in valid range
            level = max(0, min(100, level))
            return {
                "type": "ovos.volume.set",
                "data": {"level": level}
            }
        elif re.search(r'volume\s+up|increase\s+volume|turn\s+(it|the volume)\s+up', text, re.IGNORECASE):
            return {
                "type": "ovos.volume.increase"
            }
        elif re.search(r'volume\s+down|decrease\s+volume|lower\s+volume|turn\s+(it|the volume)\s+down', text, re.IGNORECASE):
            return {
                "type": "ovos.volume.decrease"
            }
        elif re.search(r'mute|silence', text, re.IGNORECASE) and not re.search(r'unmute', text, re.IGNORECASE):
            return {
                "type": "ovos.volume.mute"
            }
        elif re.search(r'unmute', text, re.IGNORECASE):
            return {
                "type": "ovos.volume.unmute"
            }
        
        # Media playback commands
        elif re.search(r'(play|start)\s+music', text, re.IGNORECASE):
            return {
                "type": "ovos.audio.service.play"
            }
        elif re.search(r'pause|stop\s+music', text, re.IGNORECASE):
            return {
                "type": "ovos.audio.service.pause"
            }
        elif re.search(r'resume|continue', text, re.IGNORECASE):
            return {
                "type": "ovos.audio.service.resume"
            }
        elif re.search(r'next\s+(song|track)', text, re.IGNORECASE):
            return {
                "type": "ovos.audio.service.next"
            }
        elif re.search(r'previous\s+(song|track)|last\s+(song|track)', text, re.IGNORECASE):
            return {
                "type": "ovos.audio.service.prev"
            }
            
        # System commands
        elif re.search(r'restart\s+(system|services)', text, re.IGNORECASE):
            return {
                "type": "system.ovos.service.restart"
            }
        elif re.search(r'(shut\s*down|power\s*off)\s+(system|computer)', text, re.IGNORECASE):
            return {
                "type": "system.shutdown"
            }
        elif re.search(r'reboot\s+(system|computer)', text, re.IGNORECASE):
            return {
                "type": "system.reboot"
            }
            
        # No command found
        return None
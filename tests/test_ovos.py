#!/usr/bin/env python3
# tests/test_ovos.py

import os
import sys
import time
import threading
import json
from pathlib import Path

# Import base test class
from test_base import BaseTest, run_async_tests

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

class OVOSTest(BaseTest):
    """Tests for the Grace OVOS integration."""
    
    def __init__(self, verbose=False):
        super().__init__("ovos", verbose)
        
        # Create a test config
        self.test_config = {
            'ovos': {
                'host': 'localhost',
                'port': 8181,
                'retry_attempts': 1,
                'retry_delay': 0.1,
                'disable_ovos': False,
                'reconnect_delay': 1
            },
            'debug': verbose
        }
    
    def test_ovos_message_formatting(self):
        """Test OVOS message formatting."""
        from grace.ovos.ovos_message import OVOSMessage
        
        message_handler = OVOSMessage()
        
        # Test utterance formatting
        utterance_msg = message_handler.format_utterance("Hello world")
        assert utterance_msg is not None, "format_utterance should return a Message object"
        assert hasattr(utterance_msg, 'type'), "Message should have a type attribute"
        assert utterance_msg.type == 'recognizer_loop:utterance', "Utterance message should have correct type"
        assert 'utterances' in utterance_msg.data, "Utterance message should have utterances in data"
        assert utterance_msg.data['utterances'][0] == "Hello world", "Utterance message should contain the original text"
        
        # Test speak formatting
        speak_msg = message_handler.format_speak("Hello world")
        assert speak_msg is not None, "format_speak should return a Message object"
        assert hasattr(speak_msg, 'type'), "Message should have a type attribute"
        assert speak_msg.type == 'speak', "Speak message should have correct type"
        assert 'utterance' in speak_msg.data, "Speak message should have utterance in data"
        assert speak_msg.data['utterance'] == "Hello world", "Speak message should contain the original text"
        
        # Test command message creation
        command = {
            'type': 'test.command',
            'data': {
                'param1': 'value1',
                'param2': 'value2'
            }
        }
        command_msg = message_handler.create_command_message(command)
        assert command_msg is not None, "create_command_message should return a Message object"
        assert hasattr(command_msg, 'type'), "Message should have a type attribute"
        assert command_msg.type == 'test.command', "Command message should have correct type"
        assert 'param1' in command_msg.data, "Command message should have param1 in data"
        assert command_msg.data['param1'] == 'value1', "Command message should have correct param1 value"
        
        # Test JSON extraction from message
        json_str = message_handler.message_to_json(command_msg)
        assert json_str is not None, "message_to_json should return a JSON string"
        parsed = json.loads(json_str)
        assert 'type' in parsed, "JSON should have a type field"
        assert parsed['type'] == 'test.command', "JSON should have correct type"
        assert 'data' in parsed, "JSON should have a data field"
        assert 'param1' in parsed['data'], "JSON data should have param1"
        
        # Test message validation
        is_valid = message_handler.validate_message(command_msg)
        assert is_valid, "validate_message should return True for valid message"
    
    def test_ovos_disabled_mode(self):
        """Test OVOS integration with disabled mode."""
        from grace.ovos.ovos_integration import OVOSInterface
        
        # Create config with OVOS disabled
        disabled_config = self.test_config.copy()
        disabled_config['ovos'] = self.test_config['ovos'].copy()
        disabled_config['ovos']['disable_ovos'] = True
        
        ovos = OVOSInterface(disabled_config)
        
        # Check that OVOS is disabled
        assert not ovos.is_connected(), "OVOS should not be connected when disabled"
        
        # Get connection status
        status = ovos.get_connection_status()
        assert status is not None, "get_connection_status should return a dictionary"
        assert "disabled" in status, "Status should include disabled field"
        assert status["disabled"], "disabled should be True when OVOS is disabled"
        
        # Try to send a message (should fail gracefully)
        result = ovos.send_message("test.message", {"data": "value"})
        assert not result, "send_message should return False when OVOS is disabled"
        
        # Try to send an utterance (should fail gracefully)
        result = ovos.send_utterance("test utterance")
        assert not result, "send_utterance should return False when OVOS is disabled"
        
        # Test shutdown
        ovos.shutdown()
    
    def test_ovos_client_queue_messages(self):
        """Test message queuing in OVOS client."""
        from grace.ovos.ovos_client import OVOSClient
        
        ovos_client = OVOSClient(self.test_config)
        
        # Make sure OVOS client is disconnected (likely already is since server probably isn't running)
        assert not ovos_client.is_connected(), "OVOS should not be connected for this test"
        
        # Send a message (should be queued)
        ovos_client.send_message("test.message", {"data": "value"})
        
        # Check that message was queued
        with ovos_client.message_queue_lock:
            assert len(ovos_client.message_queue) > 0, "Message should be added to queue when disconnected"
        
        # Test queue limit
        # Set a small queue limit
        ovos_client.message_queue_limit = 5
        
        # Add messages to fill the queue
        for i in range(10):
            ovos_client.send_message(f"test.message.{i}", {"data": f"value{i}"})
        
        # Check that queue size is limited
        with ovos_client.message_queue_lock:
            assert len(ovos_client.message_queue) <= 5, "Queue size should be limited"
        
        # Test shutdown
        ovos_client.shutdown()
    
    def test_ovos_handlers_callbacks(self):
        """Test callback registration and handling in OVOS handlers."""
        from grace.ovos.ovos_handlers import OVOSHandlers
        from grace.ovos.ovos_client import OVOSClient
        
        # Create a client
        client = OVOSClient(self.test_config)
        
        # Create handlers
        handlers = OVOSHandlers(client)
        
        # Create a callback function
        callback_called = False
        
        def test_callback(message):
            nonlocal callback_called
            callback_called = True
        
        # Register the callback
        handlers.register_callback("test_event", test_callback)
        
        # Check that callback was registered
        assert "test_event" in handlers.callback_registry, "Callback should be registered"
        assert test_callback in handlers.callback_registry["test_event"], "Callback function should be in registry"
        
        # Simulate a message that triggers the callback
        # Since we don't have an actual messagebus connection,
        # we'll call the callback directly through the registry
        for callback in handlers.callback_registry["test_event"]:
            callback({"data": {"test": True}})
        
        # Check that callback was called
        assert callback_called, "Callback should be called"
        
        # Test deregistering the callback
        handlers.deregister_callback("test_event", test_callback)
        
        # Check that callback was deregistered
        assert test_callback not in handlers.callback_registry["test_event"], "Callback should be deregistered"
    
    def test_ovos_commands_volume_control(self):
        """Test OVOS commands volume control functions."""
        from grace.ovos.ovos_commands import OVOSCommands
        from grace.ovos.ovos_client import OVOSClient
        
        # Create a client
        client = OVOSClient(self.test_config)
        
        # Create commands
        commands = OVOSCommands(client)
        
        # Test volume control functions
        # Since we don't have an actual OVOS instance, these will fail
        # but should fail gracefully
        
        # Get volume
        volume = commands.get_volume()
        assert volume is None, "get_volume should return None when not connected"
        
        # Set volume
        result = commands.set_volume(50)
        assert not result, "set_volume should return False when not connected"
        
        # Volume up
        result = commands.volume_up()
        assert not result, "volume_up should return False when not connected"
        
        # Volume down
        result = commands.volume_down()
        assert not result, "volume_down should return False when not connected"
        
        # Mute
        result = commands.mute()
        assert not result, "mute should return False when not connected"
        
        # Unmute
        result = commands.unmute()
        assert not result, "unmute should return False when not connected"
    
    def test_ovos_integration_handle_fallback_intent(self):
        """Test OVOS integration fallback intent handling."""
        from grace.ovos.ovos_integration import OVOSInterface
        
        ovos = OVOSInterface(self.test_config)
        
        # Test fallback intent handler with time query
        result = ovos.handle_fallback_intent("what time is it")
        assert result is not None, "handle_fallback_intent should return a result"
        assert "handled" in result, "Result should include handled field"
        assert result["handled"], "Time query should be handled"
        assert "intent_type" in result, "Result should include intent_type field"
        assert "response" in result, "Result should include response field"
        
        # Test fallback intent handler with date query
        result = ovos.handle_fallback_intent("what is today's date")
        assert result is not None, "handle_fallback_intent should return a result"
        assert "handled" in result, "Result should include handled field"
        assert result["handled"], "Date query should be handled"
        assert "intent_type" in result, "Result should include intent_type field"
        assert "response" in result, "Result should include response field"
        
        # Test fallback intent handler with unknown query
        result = ovos.handle_fallback_intent("what is the meaning of life")
        assert result is not None, "handle_fallback_intent should return a result"
        assert "handled" in result, "Result should include handled field"
        assert not result["handled"], "Unknown query should not be handled"
        
        # Test shutdown
        ovos.shutdown()
    
    async def test_async_operations(self):
        """Test OVOS integration async operations."""
        from grace.ovos.ovos_integration import OVOSInterface
        
        ovos = OVOSInterface(self.test_config)
        
        # Test async test_connection
        success, message = await ovos.test_connection_async()
        assert not success, "test_connection_async should return False when not connected"
        assert message is not None, "test_connection_async should return a message"
        
        # Test reset_connection_async
        result = await ovos.reset_connection_async()
        assert not result, "reset_connection_async should return False when not initially connected"
        
        # Test get_skills_async
        skills = await ovos.get_skills_async()
        assert isinstance(skills, list), "get_skills_async should return a list"
        assert len(skills) == 0, "get_skills_async should return an empty list when not connected"
        
        # Test shutdown
        ovos.shutdown()
    
    async def run_all_tests(self):
        """Run all OVOS integration tests."""
        self.test_ovos_message_formatting()
        self.test_ovos_disabled_mode()
        self.test_ovos_client_queue_messages()
        self.test_ovos_handlers_callbacks()
        self.test_ovos_commands_volume_control()
        self.test_ovos_integration_handle_fallback_intent()
        await self.test_async_operations()
        
        return self.print_results()

def run_tests(verbose=False):
    """Run OVOS integration tests."""
    test = OVOSTest(verbose=verbose)
    return run_async_tests(test.run_all_tests())

if __name__ == "__main__":
    run_tests(verbose=True)
"""
Grace AI System - System Orchestrator

This module provides a central orchestrator for coordinating all system components.
"""

import asyncio
import logging
import json
import time
import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from grace.utils.common import ConversationEntry, calculate_relevance, print_debug_separator


class SystemOrchestrator:
    """
    Central orchestrator for the Grace AI system.
    
    This class coordinates the interaction between all system components,
    including memory, language model, audio, and OVOS integration.
    """
    
    def __init__(self, config, memory_system, language_model, audio_system=None, ovos_interface=None):
        """
        Initialize the system orchestrator.
        
        Args:
            config: System configuration
            memory_system: Memory system instance
            language_model: Language model instance
            audio_system: Optional audio system instance
            ovos_interface: Optional OVOS interface instance
        """
        self.logger = logging.getLogger('grace.orchestrator')
        self.config = config
        self.memory_system = memory_system
        self.language_model = language_model
        self.audio_system = audio_system
        self.ovos_interface = ovos_interface
        
        # System status
        self.running = False
        self.system_ready = False
        self.ready_event = asyncio.Event()
        
        # Session context tracking
        self.session_context = {
            'conversation_history': [],
            'last_memory_search': {},
            'last_command_result': "",
            'system_state': {}
        }
        
        # Command router for handling JSON command output
        self.command_routes = {
            'memory_add': self._handle_memory_add,
            'memory_search': self._handle_memory_search,
            'speak': self._handle_speak_command,
            'ovos_command': self._handle_ovos_command,
            'system_command': self._handle_system_command
        }
        
        # Debug settings
        self.debug_mode = config.get('debug', False)
    
    async def start(self):
        """Start the orchestrator and initialize all components."""
        self.logger.info("Starting system orchestrator")
        self.running = True
        
        # Initialize connection to OVOS if available
        if self.ovos_interface:
            # Register callbacks for OVOS events
            self.ovos_interface.register_callback('skills_initialized', self._on_ovos_ready)
            self.ovos_interface.register_callback('intent_failure', self._on_intent_failure)
            
            # Wait for OVOS to be ready (with timeout)
            if self.ovos_interface.is_connected():
                self.logger.info("Waiting for OVOS to be ready...")
                await asyncio.to_thread(self.ovos_interface.wait_for_ready, 10)
        
        # Initialize audio system if available
        if self.audio_system:
            if hasattr(self.audio_system, 'get_status'):
                status = self.audio_system.get_status()
                self.logger.info(f"Audio system initialized. Recognition ready: {status.get('recognition_ready', False)}")
        
        # System is ready
        self.system_ready = True
        self.ready_event.set()
        self.logger.info("System orchestrator ready")
        
    async def stop(self):
        """Stop the orchestrator and cleanup."""
        self.logger.info("Stopping system orchestrator")
        self.running = False
        
        # Cancel any pending tasks
        tasks = [task for task in asyncio.all_tasks() 
                if task is not asyncio.current_task() and 
                task.get_name().startswith('grace_')]
        
        for task in tasks:
            task.cancel()
            
        await asyncio.gather(*tasks, return_exceptions=True)
        
        self.logger.info("System orchestrator stopped")
    
    async def voice_interaction_loop(self):
        """Run the voice interaction loop."""
        if not self.audio_system:
            self.logger.error("Cannot start voice interaction loop: Audio system not available")
            print("Error: Audio system not available. Use text mode instead.")
            await self.text_interaction_loop()
            return
        
        self.logger.info("Starting voice interaction loop")
        print("\nGrace AI Voice Interaction Mode")
        print("Speak to interact or type 'exit' to quit\n")
        
        while self.running:
            try:
                # Wait until system is ready
                if not self.system_ready:
                    await self.ready_event.wait()
                
                print("\nListening... (or type a command)")
                
                # Create a task for listening
                listen_task = asyncio.create_task(
                    self.audio_system.listen_and_transcribe_async(),
                    name="grace_listen_task"
                )
                
                # Also check for text input
                input_task = asyncio.create_task(
                    self._async_input(),
                    name="grace_input_task"
                )
                
                # Wait for either voice or text input with a timeout
                done, pending = await asyncio.wait(
                    [listen_task, input_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel the pending task
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                
                # Process the completed task
                completed_task = next(iter(done))
                
                if completed_task == listen_task:
                    # Voice input
                    transcript, audio_data = completed_task.result()
                    
                    if transcript:
                        print(f"\nYou: {transcript}")
                        response = await self.process_user_input(transcript, "voice")
                        
                        # Speak the response if audio system is available
                        if self.audio_system and response:
                            await self.audio_system.speak_async(response)
                            
                elif completed_task == input_task:
                    # Text input
                    user_input = completed_task.result()
                    
                    if user_input.lower() in ['exit', 'quit', 'bye']:
                        print("Exiting voice interaction mode...")
                        break
                    
                    if user_input:
                        print(f"\nYou (text): {user_input}")
                        response = await self.process_user_input(user_input, "text")
                        
                        # Speak the response if audio system is available
                        if self.audio_system and response:
                            await self.audio_system.speak_async(response)
                            
            except asyncio.CancelledError:
                # Task was cancelled, exit the loop
                break
            except Exception as e:
                self.logger.error(f"Error in voice interaction loop: {e}", exc_info=True)
                print(f"\nError: {e}")
                # Continue the loop despite errors
                await asyncio.sleep(1)
    
    async def text_interaction_loop(self):
        """Run the text interaction loop."""
        self.logger.info("Starting text interaction loop")
        print("\nGrace AI Text Interaction Mode")
        print("Type your messages or 'exit' to quit\n")
        
        while self.running:
            try:
                # Wait until system is ready
                if not self.system_ready:
                    await self.ready_event.wait()
                
                # Get user input
                user_input = await self._async_input("\nYou: ")
                
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("Exiting...")
                    break
                
                if user_input:
                    response = await self.process_user_input(user_input, "text")
                    
                    if response:
                        print(f"\nGrace: {response}")
            
            except asyncio.CancelledError:
                # Task was cancelled, exit the loop
                break
            except Exception as e:
                self.logger.error(f"Error in text interaction loop: {e}", exc_info=True)
                print(f"\nError: {e}")
                # Continue the loop despite errors
                await asyncio.sleep(1)
    
    async def process_user_input(self, user_input: str, input_mode: str = "text") -> str:
        """
        Process user input and generate a response.
        
        Args:
            user_input: User input text
            input_mode: Input mode ('text' or 'voice')
            
        Returns:
            Response text
        """
        self.logger.info(f"Processing user input: {user_input}")
        
        # Create a new conversation entry
        entry = ConversationEntry(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            user_input=user_input,
            stt_transcript=user_input if input_mode == "voice" else "",
            metadata={"input_mode": input_mode}
        )
        
        try:
            # Search memory for relevant context
            memory_results = await self.memory_system.search_memories(user_input)
            entry.memory_context = memory_results
            self.session_context['last_memory_search'] = memory_results
            
            # Format memory context for prompt
            memory_context = self._format_memory_context(memory_results)
            
            # Build the prompt with memory context
            prompt = self._build_llm_prompt(user_input, memory_context)
            entry.prompt = prompt
            
            # Try OVOS intent handling first if available
            ovos_handled = False
            ovos_response = None
            
            if self.ovos_interface and self.ovos_interface.is_connected():
                # Send to OVOS for intent handling
                self.logger.debug("Sending utterance to OVOS for intent handling")
                self.ovos_interface.send_utterance(user_input)
                
                # We would need a callback mechanism to get the result from OVOS
                # For now, we'll proceed with LLM processing
                
                # Fallback intent handling when OVOS not available
                intent_result = self.ovos_interface.handle_fallback_intent(user_input)
                if intent_result.get("handled", False):
                    ovos_handled = True
                    ovos_response = intent_result.get("response")
            
            # If not handled by OVOS, process with LLM
            if not ovos_handled:
                # Generate response from language model
                self.logger.debug("Generating response from language model")
                llm_response = self.language_model.generate(
                    prompt,
                    max_tokens=None,  # Use default from config
                    temperature=None  # Use default from config
                )
                
                # Parse the model response
                thinking, json_response = self.language_model.parse_response(llm_response)
                
                # Store the results in the conversation entry
                entry.thinking_process = thinking
                entry.json_response = json_response
                entry.model_response = json_response.get("response", "")
                
                # Process commands in the JSON response
                command_result = await self._process_commands(json_response)
                entry.command_result = command_result
                self.session_context['last_command_result'] = command_result
                
                # Final response text
                response_text = entry.model_response
            else:
                # Use OVOS response
                response_text = ovos_response or "I processed your request."
                entry.model_response = response_text
            
            # Log the conversation entry
            await self.memory_system.log_conversation(entry)
            
            # Add to conversation history (limited size)
            self.session_context['conversation_history'].append({
                'user': user_input,
                'assistant': response_text
            })
            
            # Keep history at a reasonable size
            max_history = self.config.get('system', {}).get('max_conversation_history', 10)
            if len(self.session_context['conversation_history']) > max_history:
                self.session_context['conversation_history'] = \
                    self.session_context['conversation_history'][-max_history:]
            
            return response_text
            
        except Exception as e:
            self.logger.error(f"Error processing input: {e}", exc_info=True)
            entry.error = str(e)
            
            # Log the error
            await self.memory_system.log_conversation(entry)
            
            # Return graceful error message
            if self.config.get('system', {}).get('error_fallback', True):
                return "I'm sorry, I encountered an issue processing your request. Please try again."
            else:
                return f"Error: {e}"
    
    def _build_llm_prompt(self, user_input: str, memory_context: str) -> str:
        """
        Build the prompt for the language model.
        
        Args:
            user_input: User input text
            memory_context: Memory context string
            
        Returns:
            Formatted prompt
        """
        # Get current time for context
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Get recent conversation history
        conversation_history = ""
        for item in self.session_context['conversation_history'][-5:]:  # Last 5
            conversation_history += f"USER: {item['user']}\nASSISTANT: {item['assistant']}\n\n"
        
        # Build the prompt with all context
        prompt = f"""You are Grace, a highly intelligent personal AI assistant. The current time is {current_time}.

<context>
{memory_context}

Recent conversation:
{conversation_history}
</context>

<system_instructions>
1. Think carefully about how to respond to the user.
2. When applicable, use the context information to provide personalized responses.
3. If you need to add to memory, command the system using a structured JSON format.
4. Always provide your thinking process enclosed in <think>...</think> tags.
5. Your final response should be structured as a JSON object with a 'response' field.
</system_instructions>

<think>
Analyze the user's request and determine the best way to respond.
Consider what information in memory might be relevant.
Think through what actions or commands might be needed.
</think>

USER: {user_input}

ASSISTANT:"""
        
        return prompt
    
    def _format_memory_context(self, memory_results: Dict) -> str:
        """
        Format memory context for inclusion in the prompt.
        
        Args:
            memory_results: Memory search results
            
        Returns:
            Formatted memory context string
        """
        context_parts = []
        
        # Format critical memories
        if critical_memories := memory_results.get('critical', []):
            context_parts.append("CRITICAL MEMORIES:")
            for mem in critical_memories[:5]:  # Limit to 5 most relevant
                relevance = mem.get('relevance', 0)
                content = mem.get('content', '')
                context_parts.append(f"[{relevance:.2f}] {content}")
        
        # Format contextual memories
        if contextual_memories := memory_results.get('contextual', []):
            context_parts.append("\nCONTEXTUAL MEMORIES:")
            for mem in contextual_memories[:5]:  # Limit to 5 most relevant
                relevance = mem.get('score', 0) if isinstance(mem, dict) else 0
                content = mem.get('content', '') if isinstance(mem, dict) else str(mem)
                context_parts.append(f"[{relevance:.2f}] {content}")
        
        # Format reference materials
        if reference_materials := memory_results.get('reference', []):
            context_parts.append("\nREFERENCE MATERIALS:")
            for mem in reference_materials[:3]:  # Limit to 3 most relevant
                relevance = mem.get('relevance', 0)
                content = mem.get('content', '')
                context_parts.append(f"[{relevance:.2f}] {content}")
        
        # Format conversation history from memory
        if conversations := memory_results.get('conversations', []):
            context_parts.append("\nRELEVANT PAST CONVERSATIONS:")
            for mem in conversations[:3]:  # Limit to 3 most relevant
                relevance = mem.get('relevance', 0)
                content = mem.get('content', '')
                context_parts.append(f"[{relevance:.2f}] {content}")
        
        # Join all parts with newlines
        return "\n".join(context_parts)
    
    async def _process_commands(self, json_response: Dict) -> str:
        """
        Process commands in the JSON response.
        
        Args:
            json_response: Parsed JSON response
            
        Returns:
            Command result message
        """
        results = []
        
        try:
            # Process each command type that might be in the response
            for command_type, handler in self.command_routes.items():
                if command_type in json_response:
                    command_data = json_response[command_type]
                    result = await handler(command_data)
                    if result:
                        results.append(result)
            
            # Join all results with newlines
            return "\n".join(results) if results else ""
            
        except Exception as e:
            self.logger.error(f"Error processing commands: {e}", exc_info=True)
            return f"Error processing commands: {e}"
    
    async def _handle_memory_add(self, command_data) -> str:
        """Handle memory add command."""
        if not isinstance(command_data, dict):
            return "Invalid memory_add format - must be a dictionary"
        
        content = command_data.get('content')
        if not content:
            return "Invalid memory_add format - missing content"
        
        memory_type_str = command_data.get('type', 'CONTEXTUAL')
        user_id = command_data.get('user_id', 'grace_user')
        metadata = command_data.get('metadata', {})
        
        # Convert memory type string to enum
        from grace.memory.types import MemoryType
        try:
            memory_type = getattr(MemoryType, memory_type_str.upper())
        except (AttributeError, TypeError):
            self.logger.warning(f"Invalid memory type: {memory_type_str}, using CONTEXTUAL")
            memory_type = MemoryType.CONTEXTUAL
        
        # Add to memory
        memory_id = await self.memory_system.add_memory(
            content=content,
            memory_type=memory_type,
            user_id=user_id,
            metadata=metadata
        )
        
        if memory_id:
            return f"Memory added successfully with ID: {memory_id}"
        else:
            return "Failed to add memory"
    
    async def _handle_memory_search(self, command_data) -> str:
        """Handle memory search command."""
        if isinstance(command_data, str):
            query = command_data
        elif isinstance(command_data, dict):
            query = command_data.get('query')
        else:
            return "Invalid memory_search format"
        
        if not query:
            return "Invalid memory_search format - missing query"
        
        # Search memory
        results = await self.memory_system.search_memories(query)
        
        # Store in session context
        self.session_context['last_memory_search'] = results
        
        # Summarize results
        total_results = sum(len(results.get(key, [])) for key in results)
        return f"Memory search completed: {total_results} memories found"
    
    async def _handle_speak_command(self, command_data) -> str:
        """Handle speak command."""
        if not self.audio_system:
            return "Audio system not available"
        
        text = command_data if isinstance(command_data, str) else command_data.get('text')
        if not text:
            return "Invalid speak format - missing text"
        
        # Speak the text
        success = await self.audio_system.speak_async(text)
        
        if success:
            return "Speech command executed successfully"
        else:
            return "Failed to execute speech command"
    
    async def _handle_ovos_command(self, command_data) -> str:
        """Handle OVOS command."""
        if not self.ovos_interface or not self.ovos_interface.is_connected():
            return "OVOS interface not available"
        
        if not isinstance(command_data, dict):
            return "Invalid ovos_command format - must be a dictionary"
        
        command_type = command_data.get('type')
        command_params = command_data.get('params', {})
        
        if not command_type:
            return "Invalid ovos_command format - missing type"
        
        # Execute the OVOS command
        success = self.ovos_interface.execute_command({
            'type': command_type,
            'data': command_params
        })
        
        if success:
            return f"OVOS command '{command_type}' executed successfully"
        else:
            return f"Failed to execute OVOS command '{command_type}'"
    
    async def _handle_system_command(self, command_data) -> str:
        """Handle system command."""
        if not isinstance(command_data, dict):
            return "Invalid system_command format - must be a dictionary"
        
        command_type = command_data.get('type')
        
        if not command_type:
            return "Invalid system_command format - missing type"
        
        # Handle different system commands
        if command_type == 'status':
            return self._get_system_status()
        elif command_type == 'restart':
            # Schedule restart
            asyncio.create_task(self._restart_system())
            return "System restart scheduled"
        elif command_type == 'toggle_debug':
            self.debug_mode = not self.debug_mode
            return f"Debug mode {'enabled' if self.debug_mode else 'disabled'}"
        else:
            return f"Unknown system command: {command_type}"
    
    def _get_system_status(self) -> str:
        """Get system status information."""
        status_parts = ["System Status:"]
        
        # Memory system status
        status_parts.append("- Memory System: Active")
        
        # Language model status
        model_info = getattr(self.language_model, 'model', None)
        model_status = "Active" if model_info else "Issues detected"
        status_parts.append(f"- Language Model: {model_status}")
        
        # Audio system status
        if self.audio_system:
            audio_status = "Active"
            if hasattr(self.audio_system, 'get_status'):
                audio_state = self.audio_system.get_status()
                if not audio_state.get('overall_ready', False):
                    audio_status = "Issues detected"
            status_parts.append(f"- Audio System: {audio_status}")
        else:
            status_parts.append("- Audio System: Disabled")
        
        # OVOS status
        if self.ovos_interface:
            ovos_status = "Connected" if self.ovos_interface.is_connected() else "Disconnected"
            status_parts.append(f"- OVOS Integration: {ovos_status}")
        else:
            status_parts.append("- OVOS Integration: Disabled")
        
        # Join all parts with newlines
        return "\n".join(status_parts)
    
    async def _restart_system(self):
        """Restart the system orchestrator."""
        self.logger.info("Restarting system...")
        
        # Stop the orchestrator
        self.running = False
        self.system_ready = False
        self.ready_event.clear()
        
        # Wait a moment
        await asyncio.sleep(1)
        
        # Start again
        self.running = True
        await self.start()
    
    def _on_ovos_ready(self, message):
        """Handle OVOS ready event."""
        self.logger.info("OVOS system is ready")
        self.system_ready = True
        self.ready_event.set()
    
    def _on_intent_failure(self, message):
        """Handle OVOS intent failure event."""
        if not hasattr(message, 'data'):
            return
            
        utterance = message.data.get('utterance', '')
        self.logger.debug(f"OVOS intent failure for: {utterance}")
        
        # Handle fallback through LLM if needed
        if utterance:
            # We could process this utterance through our LLM here
            # but we'll rely on the OVOS system's fallback mechanisms for now
            pass
    
    async def _async_input(self, prompt=""):
        """Asynchronous version of input()."""
        return await asyncio.to_thread(input, prompt)

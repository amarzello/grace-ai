#!/usr/bin/env python3
"""
Grace AI System - Main Module

This is the main entry point for the Grace AI system, an advanced locally-run
personal AI assistant with comprehensive memory and tool integration.
"""

import os
import sys
import json
import time
import yaml
import asyncio
import argparse
import signal
import logging
import re
import tempfile
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

# Import components
from utils import (
    setup_logging, load_config, ConversationEntry, check_required_dependencies,
    GRACE_HOME, MEMORY_DB_PATH, LOGS_PATH, CONFIG_PATH, MODELS_PATH, REFERENCE_PATH,
    MemoryType, get_memory_system_status_dashboard
)
from memory_system import EnhancedMemorySystem
from language_model import LlamaWrapper
from audio_system import AudioSystem
from ovos_integration import OVOSInterface


class GraceSystem:
    """
    Main Grace AI System with integrated components.
    
    Features:
    - Text and voice input modes
    - Advanced 3-tier memory system
    - Local LLM integration
    - OpenVoiceOS integration
    - Comprehensive logging and verification
    - Graceful error handling and shutdown
    """
    
    def __init__(self, config_override: Dict = None):
        """
        Initialize the Grace AI system with the provided configuration.
        
        Args:
            config_override: Configuration overrides
        """
        # First, check required dependencies
        check_required_dependencies()
        
        # Set up logging
        self.logger = setup_logging(debug=config_override.get('debug', False) if config_override else False)
        self.logger.info("Initializing Grace AI System...")
        
        # Load configuration
        config_file = config_override.get('config') if config_override else None
        self.config = load_config(config_file, config_override)
            
        # Check debug mode
        self.debug_mode = self.config.get('debug', False)
            
        # Check amnesia mode
        self.amnesia_mode = self.config.get('amnesia_mode', False)
        if self.amnesia_mode:
            self.logger.info("*** AMNESIA MODE ACTIVE - No new memories will be stored ***")
        
        # Create required directories
        for path in [MEMORY_DB_PATH, LOGS_PATH, REFERENCE_PATH]:
            path.mkdir(parents=True, exist_ok=True)
            
        # System state
        self.running = False
        self.user_id = "grace_user"
        self.input_mode = "both"  # "text", "voice", or "both"
        
        # Initialization flags for better error handling
        self.memory_initialized = False
        self.llm_initialized = False
        self.audio_initialized = False
        self.ovos_initialized = False
        
        # Enable WAL mode for SQLite databases if configured
        if self.config.get('memory', {}).get('sqlite_wal_mode', True):
            self._enable_sqlite_wal_mode()
        
        # Initialize subsystems with proper error handling
        self._init_subsystems()
        
        # Conversation history for proper context management
        self.conversation_history = []
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, sig, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {sig}, shutting down...")
        self.running = False
    
    def _enable_sqlite_wal_mode(self):
        """Enable WAL mode for SQLite databases to improve concurrency."""
        try:
            # Check for existing database files
            db_files = list(MEMORY_DB_PATH.glob("*.db"))
            
            for db_path in db_files:
                try:
                    with sqlite3.connect(db_path) as conn:
                        conn.execute("PRAGMA journal_mode=WAL")
                        conn.execute("PRAGMA synchronous=NORMAL")
                        conn.execute("PRAGMA cache_size=-8000")  # Use ~8MB of memory for cache
                    self.logger.debug(f"Enabled WAL mode for {db_path.name}")
                except sqlite3.Error as e:
                    self.logger.warning(f"Failed to enable WAL mode for {db_path.name}: {e}")
        except Exception as e:
            self.logger.warning(f"Error setting up SQLite WAL mode: {e}")
            
    def _init_subsystems(self):
        """Initialize all subsystems with proper error handling."""
        # Initialize memory system first
        try:
            self.memory_system = EnhancedMemorySystem(self.config)
            self.memory_initialized = True
        except Exception as e:
            error_msg = f"CRITICAL: Failed to initialize memory system: {e}"
            self.logger.critical(error_msg)
            print(f"\n{'!'*80}\n{error_msg}\n{'!'*80}\n")
            sys.exit(1)  # Exit if memory system fails - it's critical
            
        # Set QWQ model parameters according to official recommendations
        if 'QWQ' in self.config.get('llama', {}).get('model_path', ''):
            self.logger.info("QWQ model detected, applying recommended parameters")
            self.config.setdefault('llama', {})['temperature'] = 0.6
            self.config.setdefault('llama', {})['top_p'] = 0.95
            self.config.setdefault('llama', {})['top_k'] = 30
            self.config.setdefault('llama', {})['min_p'] = 0.0
            self.config.setdefault('llama', {})['presence_penalty'] = 1.0
        
        # Initialize language model
        try:
            self.llama_model = LlamaWrapper(self.config)
            self.llm_initialized = self.llama_model.model is not None
            if not self.llm_initialized:
                error_msg = "CRITICAL: Language model failed to initialize properly."
                self.logger.critical(error_msg)
                print(f"\n{'!'*80}\n{error_msg}\n{'!'*80}\n")
                sys.exit(1)  # Exit if language model fails - it's critical
        except Exception as e:
            error_msg = f"CRITICAL: Failed to initialize language model: {e}"
            self.logger.critical(error_msg)
            print(f"\n{'!'*80}\n{error_msg}\n{'!'*80}\n")
            sys.exit(1)  # Exit if language model fails - it's critical
            
        # Initialize audio system
        try:
            self.audio_system = AudioSystem(self.config)
            self.audio_initialized = True
        except Exception as e:
            self.logger.error(f"Failed to initialize audio system: {e}", exc_info=True)
            self.audio_system = None
            
        # Initialize OVOS interface last
        try:
            self.ovos_interface = OVOSInterface(self.config)
            self.ovos_initialized = True
        except Exception as e:
            self.logger.error(f"Failed to initialize OVOS interface: {e}", exc_info=True)
            self.ovos_interface = None
                
    def build_prompt(self, user_input: str, memories: Dict) -> str:
        """
        Build prompt with memories and instructions.
        
        Args:
            user_input: User input text
            memories: Retrieved memories
            
        Returns:
            Complete prompt for the language model
        """
        memory_text = self._format_memories(memories)
        
        # Get current time in a more readable format
        current_time = datetime.now()
        formatted_time = current_time.strftime("%A, %B %d, %Y at %I:%M %p")
        timestamp = current_time.isoformat()
        
        # Add amnesia mode notice
        amnesia_notice = ""
        if self.amnesia_mode:
            amnesia_notice = "\nNOTE: Currently in amnesia mode. No new memories will be stored.\n"
        
        # Prompt optimized for QWQ-32B, using its specific formatting
        # According to official recommendations, we should:
        # 1. Use "<think>\n" to enforce thoughtful output
        # 2. Keep a clean conversation history without thinking content
        # 3. Use standardized output formats
        
        # Build the conversation history without thinking parts
        history_text = ""
        if self.conversation_history:
            for i, (user_msg, assistant_msg) in enumerate(self.conversation_history[-5:]):  # Last 5 turns
                history_text += f"USER: {user_msg}\nASSISTANT: {assistant_msg}\n\n"
        
        prompt = f"""SYSTEM: You are Grace, an AI assistant that runs locally on the user's computer. You are helpful, informative, and never refuse to answer questions. Always be respectful and provide accurate information.

Current time and date: {formatted_time}
{amnesia_notice}

RELEVANT INFORMATION:
{memory_text}

{history_text}USER: {user_input}

A: <think>
"""
        
        # Log full prompt in debug mode
        if self.debug_mode:
            print("\n" + "="*80)
            print("FULL PROMPT SENT TO MODEL:")
            print("="*80)
            print(prompt)
            print("="*80 + "\n")
            
        return prompt
        
    def _format_memories(self, memories: Dict) -> str:
        """
        Format memories for prompt inclusion.
        
        Args:
            memories: Dictionary of memory results by category
            
        Returns:
            Formatted memory text
        """
        # Return early if memories is None
        if not memories:
            return "No relevant memories found."
            
        formatted = []
        
        # Format contextual memories from mem0
        if memories.get("contextual"):
            formatted.append("=== Contextual Memories ===")
            for mem in memories["contextual"][:10]:
                if isinstance(mem, dict):
                    content = mem.get('memory', mem.get('content', ''))
                    if content:
                        formatted.append(f"- {content}")
                    
        # Format critical memories
        if memories.get("critical"):
            formatted.append("\n=== Critical Information ===")
            for mem in memories["critical"][:5]:
                if isinstance(mem, dict):
                    content = mem.get('content', '')
                    if content:
                        formatted.append(f"- {content}")
                    
        # Format conversations
        if memories.get("conversations"):
            formatted.append("\n=== Recent Conversations ===")
            for conv in memories["conversations"][:3]:
                if isinstance(conv, dict):
                    # Try to parse nested JSON
                    content = conv.get('content', '')
                    if content and content.startswith('{'):
                        try:
                            parsed = json.loads(content)
                            if 'user_input' in parsed and 'json_response' in parsed:
                                formatted.append(f"User: {parsed['user_input']}")
                                if parsed.get('json_response', {}).get('response'):
                                    formatted.append(f"You: {parsed['json_response']['response']}")
                                continue
                        except Exception:
                            pass
                    formatted.append(f"- {content[:100]}...")
        
        # Format reference memories
        if memories.get("reference"):
            formatted.append("\n=== Reference Materials ===")
            for ref in memories["reference"][:3]:
                if isinstance(ref, dict):
                    source = ref.get('metadata', {}).get('source', 'Unknown source')
                    filename = ref.get('metadata', {}).get('filename', 
                                                         Path(source).name if isinstance(source, str) else 'Unknown')
                    # Include just a summary or excerpt
                    content = ref.get('content', '')
                    excerpt = content[:100] + "..." if len(content) > 100 else content
                    formatted.append(f"- {filename}: {excerpt}")
                
        return "\n".join(formatted) if formatted else "No relevant memories found."
        
    async def process_input(self, user_input: str, stt_transcript: str = None, audio_data = None) -> ConversationEntry:
        """
        Process user input through the full pipeline.
        
        Args:
            user_input: User input text
            stt_transcript: Original speech-to-text transcript if available
            audio_data: Original audio data if available
            
        Returns:
            Conversation entry with processing results
        """
        entry = ConversationEntry(timestamp=datetime.now().isoformat())
        entry.user_input = user_input
        entry.stt_transcript = stt_transcript or ""
        
        try:
            # Check for required subsystems
            if not self.llm_initialized:
                raise RuntimeError("Language model not initialized. Cannot process input.")
            
            # Search memories
            self.logger.info("Searching memories...")
            memories = {}
            
            # Only search memories if memory system is initialized
            if self.memory_initialized:
                try:
                    memories = await self.memory_system.search_memories_with_verification(
                        user_input, self.user_id
                    )
                except Exception as e:
                    self.logger.error(f"Error searching memories: {e}", exc_info=True)
                    memories = {}  # Use empty memories if search fails
            
            entry.memory_context = memories
            
            # Build prompt
            try:
                prompt = self.build_prompt(user_input, memories)
                entry.prompt = prompt
            except Exception as e:
                self.logger.error(f"Error building prompt: {e}", exc_info=True)
                entry.error = f"Error building prompt: {e}"
                # Use a simplified fallback prompt if the main one fails
                prompt = f"SYSTEM: You are Grace, an AI assistant.\nUSER: {user_input}\nASSISTANT: <think>\n"
            
            # Generate response with parameters optimized for QWQ-32B
            self.logger.info("Generating response...")
            
            # Send any model-specific parameters
            model_params = {}
            if 'QWQ' in self.config.get('llama', {}).get('model_path', ''):
                model_params = {
                    'temperature': 0.6,
                    'top_p': 0.95, 
                    'top_k': 30,
                    'presence_penalty': 1.0
                }
                
            # Store last prompt for debugging purposes
            if hasattr(self.memory_system, '_last_prompt'):
                self.memory_system._last_prompt = prompt
                
            raw_response = self.llama_model.generate(prompt, **model_params)
            entry.model_response = raw_response
            
            # Log raw response in debug mode
            if self.debug_mode:
                print("\n" + "="*80)
                print("RAW MODEL RESPONSE:")
                print("="*80)
                print(raw_response)
                print("="*80 + "\n")
            
            # Parse response
            thinking, response = self._parse_qwq_response(raw_response)
            entry.thinking_process = thinking
            
            # Create json_response structure
            json_response = {"response": response}
            entry.json_response = json_response
            
            self.logger.debug(f"Parsed response: {response[:100]}...")
            
            # Update conversation history (without thinking parts)
            if response and response != "I understand your request.":
                self.conversation_history.append((user_input, response))
                # Keep only the last 10 turns
                if len(self.conversation_history) > 10:
                    self.conversation_history = self.conversation_history[-10:]
            
            # Log full model response for debugging if needed
            if not response or response == "I understand your request.":
                self.logger.warning("Failed to parse meaningful response")
                if self.debug_mode:
                    print("WARNING: Failed to parse a meaningful response from the model")
                
                # Don't use generic fallbacks - provide the raw response if parsing fails
                if raw_response:
                    json_response["response"] = f"Raw model output: {raw_response}"
                else:
                    json_response["response"] = "The model returned an empty response."
            
            # Process any detected commands
            messagebus_command = self._extract_command(raw_response)
            if messagebus_command and self.ovos_initialized and self.ovos_interface.is_connected():
                if isinstance(messagebus_command, dict) and "type" in messagebus_command:
                    success = self.ovos_interface.send_message(
                        messagebus_command["type"],
                        messagebus_command.get("data", {})
                    )
                    entry.command_result = f"Messagebus {'sent' if success else 'failed'}: {messagebus_command}"
                    json_response["messagebus"] = messagebus_command
                    
            utterance = self._extract_utterance(raw_response)
            if utterance and self.ovos_initialized and self.ovos_interface.is_connected():
                success = self.ovos_interface.send_utterance(utterance)
                entry.command_result = f"Utterance {'sent' if success else 'failed'}: {utterance}"
                json_response["utterance"] = utterance
                
            # Handle memory operations (skip in amnesia mode)
            memory_add = self._extract_memory_add(raw_response)
            if not self.amnesia_mode and self.memory_initialized and memory_add:
                if not isinstance(memory_add, dict):
                    # Try to parse if it's a string
                    try:
                        if isinstance(memory_add, str):
                            memory_add = json.loads(memory_add)
                    except json.JSONDecodeError:
                        memory_add = {"content": str(memory_add), "type": "critical"}
                        
                memory_type = MemoryType.CRITICAL
                
                type_map = {
                    "critical": MemoryType.CRITICAL,
                    "reference": MemoryType.REFERENCE,
                    "preference": MemoryType.USER_PREFERENCE,
                    "verification": MemoryType.VERIFICATION
                }
                
                type_str = memory_add.get("type", "critical")
                memory_type = type_map.get(type_str, MemoryType.CRITICAL)
                
                await self.memory_system.add_memory_with_verification(
                    memory_add.get("content", ""),
                    memory_type,
                    self.user_id,
                    memory_add.get("metadata", {})
                )
                json_response["memory_add"] = memory_add
                
            # Handle memory search if requested
            memory_search = self._extract_memory_search(raw_response)
            if memory_search and self.memory_initialized:
                search_results = await self.memory_system.search_memories_with_verification(
                    memory_search, self.user_id
                )
                
                # Add a note about the search to the response
                result_count = len(search_results.get('contextual', [])) + \
                               len(search_results.get('critical', [])) + \
                               len(search_results.get('reference', []))
                               
                json_response["response"] += f"\n(I searched for '{memory_search}' and found {result_count} results.)"
                json_response["memory_search"] = memory_search
                entry.json_response = json_response
                
            # Verify response against memories if verification is enabled
            if self.memory_initialized and self.config.get('memory', {}).get('enable_verification', True):
                try:
                    verification_result = await self.memory_system.verify_response(
                        json_response.get("response", ""),
                        {"user_input": user_input, "memory_context": memories}
                    )
                    # Only add verification note if confidence is low
                    if verification_result.get("score", 1.0) < 0.7:
                        json_response["response"] += f"\n\n(Note: This information has a verification score of {verification_result.get('score', 0):.2f})"
                    entry.verification_result = verification_result
                except Exception as e:
                    self.logger.error(f"Response verification error: {e}", exc_info=True)
                
            # Speak response only if we have a valid response and audio system is initialized
            if json_response.get("response") and self.audio_initialized and not self.config.get('audio', {}).get('mute', False):
                response_text = json_response["response"]
                if response_text and response_text != "I understand your request.":
                    try:
                        tts_success = self.audio_system.speak(response_text)
                        if not tts_success:
                            self.logger.warning("TTS failed, but continuing operation")
                            if self.debug_mode:
                                print("WARNING: Text-to-speech failed")
                    except Exception as tts_error:
                        self.logger.error(f"TTS error: {tts_error}", exc_info=True)
                        if self.debug_mode:
                            print(f"TTS ERROR: {tts_error}")
                entry.tts_output = response_text
                
        except Exception as e:
            self.logger.error(f"Processing error: {e}", exc_info=True)
            entry.error = str(e)
            # Set a specific error message
            entry.json_response = {"response": f"Error processing your request: {str(e)}"}
            if self.debug_mode:
                print(f"ERROR: {str(e)}")
                import traceback
                print(traceback.format_exc())
        
        # Log conversation (respects amnesia mode)
        try:
            if self.memory_initialized:
                self.memory_system.log_conversation(entry)
        except Exception as e:
            self.logger.error(f"Failed to log conversation: {e}")
        
        return entry
    
    def _parse_qwq_response(self, response: str) -> Tuple[str, str]:
        """
        Parse QWQ-32B model response to extract thinking and final response.
        
        Args:
            response: Raw model response
            
        Returns:
            Tuple of (thinking, final_response)
        """
        # Handle empty or None response
        if not response:
            return "", "I didn't receive a proper response. Please try again."
            
        thinking = ""
        final_response = ""
        
        if "</think>" in response:
            # Standard format with thinking tags
            parts = response.split("</think>", 1)
            thinking = parts[0].strip()
            final_response = parts[1].strip() if len(parts) > 1 else ""
        else:
            # No closing tag - might have just thinking or just response
            # Check if response starts with thinking
            if response.strip().startswith("I need to") or \
               response.strip().startswith("The user") or \
               response.strip().startswith("Let me") or \
               response.strip().startswith("This") or \
               response.strip().startswith("Okay"):
                # Likely thinking text
                thinking = response.strip()
                final_response = self._extract_response_from_thinking(thinking)
            else:
                # Likely just a direct response
                final_response = response.strip()
        
        # If we still don't have a valid response, use the raw text
        if not final_response:
            # Just take everything after the thinking
            final_response = response.strip() if not thinking else ""
            
        # If response is still empty, provide a fallback
        if not final_response:
            final_response = "I processed your request but couldn't formulate a response. Please try again."
        
        return thinking, final_response
    
    def _extract_response_from_thinking(self, thinking_text: str) -> str:
        """
        Extract a user-facing response from thinking text.
        
        Args:
            thinking_text: The thinking/reasoning text
            
        Returns:
            A clean response suitable for the user
        """
        # Safety check for None or empty input
        if not thinking_text:
            return ""
            
        # Look for conclusions or final decisions in the thinking
        conclusion_markers = [
            "Therefore", "In conclusion", "To answer", "My response", 
            "I should respond", "I should say", "I will tell", "My answer is",
            "So my answer is", "My final answer is", "The answer is"
        ]
        
        lines = thinking_text.split('\n')
        
        # Look for lines that seem to contain the final answer
        for marker in conclusion_markers:
            for i, line in enumerate(lines):
                if marker in line:
                    # Return the rest of the text from this point
                    remaining = '\n'.join(lines[i:])
                    # Clean up by removing obvious thinking phrases
                    cleaned = re.sub(r"I (will|should|need to|can|must) (say|respond|tell|answer|provide|explain)", "", remaining)
                    cleaned = re.sub(r"(Therefore|In conclusion|To answer)[,:]?", "", cleaned)
                    return cleaned.strip()
        
        # If no conclusion marker, look for text that looks like a direct response
        for line in lines:
            # Skip obvious thinking text
            if any(m in line for m in ["The user", "I need to", "I should", "I will", "need to explain"]):
                continue
                
            # Lines that are plainly formulated responses
            if len(line) > 20 and not line.startswith('"') and '.' in line:
                return line.strip()
        
        # Last resort: take the last 2-3 lines of thinking
        if len(lines) > 2:
            last_lines = [line for line in lines[-3:] if line.strip()]
            if last_lines:
                return '\n'.join(last_lines)
            
        # If all else fails, return a default response
        return "I processed your request but couldn't formulate a proper response."
    
    def _extract_command(self, raw_response: str) -> Optional[Dict]:
        """Extract messagebus command from response text."""
        if not raw_response:
            return None
            
        # Look for messagebus commands in various formats
        patterns = [
            r'messagebus\s*:\s*(\{.*?\})',
            r'<messagebus>(.*?)</messagebus>',
            r'send message\s*:\s*(\{.*?\})',
            r'command\s*:\s*(\{.*?\})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, raw_response, re.DOTALL)
            if match:
                try:
                    command_text = match.group(1).strip()
                    return json.loads(command_text)
                except json.JSONDecodeError:
                    continue
        return None
    
    def _extract_utterance(self, raw_response: str) -> Optional[str]:
        """Extract OVOS utterance from response text."""
        if not raw_response:
            return None
            
        # Look for utterance commands in various formats
        patterns = [
            r'utterance\s*:\s*"([^"]+)"',
            r'utterance\s*:\s*\'([^\']+)\'',
            r'<utterance>(.*?)</utterance>',
            r'send utterance\s*:\s*"([^"]+)"'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, raw_response, re.DOTALL)
            if match:
                return match.group(1).strip()
        return None
    
    def _extract_memory_add(self, raw_response: str) -> Optional[Dict]:
        """Extract memory_add command from response text."""
        if not raw_response:
            return None
            
        # Look for memory_add commands in various formats
        patterns = [
            r'memory_add\s*:\s*(\{.*?\})',
            r'<memory_add>(.*?)</memory_add>',
            r'add memory\s*:\s*(\{.*?\})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, raw_response, re.DOTALL)
            if match:
                try:
                    memory_text = match.group(1).strip()
                    return json.loads(memory_text)
                except json.JSONDecodeError:
                    # If JSON parsing fails, try to extract the content directly
                    return {"content": memory_text, "type": "critical"}
        return None
    
    def _extract_memory_search(self, raw_response: str) -> Optional[str]:
        """Extract memory_search command from response text."""
        if not raw_response:
            return None
            
        # Look for memory_search commands in various formats
        patterns = [
            r'memory_search\s*:\s*"([^"]+)"',
            r'memory_search\s*:\s*\'([^\']+)\'',
            r'<memory_search>(.*?)</memory_search>',
            r'search memory\s*:\s*"([^"]+)"'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, raw_response, re.DOTALL)
            if match:
                return match.group(1).strip()
        return None
        
    async def load_reference_materials(self, directory: Path = None):
        """
        Load reference materials into memory for verification.
        
        Args:
            directory: Directory containing reference materials
            
        Returns:
            Number of files loaded
        """
        if not self.memory_initialized:
            self.logger.warning("Memory system not initialized, cannot load reference materials")
            return 0
            
        directory = directory or REFERENCE_PATH
        try:
            return await self.memory_system.load_reference_directory(directory)
        except Exception as e:
            self.logger.error(f"Error loading reference materials: {e}", exc_info=True)
            return 0
        
    async def run_voice_mode(self):
        """Run in voice input mode with microphone listening."""
        if not self.audio_initialized:
            self.logger.error("Audio system not initialized. Cannot run voice mode.")
            print("\nAudio system not initialized. Please run in text mode.")
            return False
            
        self.logger.info("Starting voice mode...")
        
        # Start audio input
        if not self.audio_system.start_listening():
            self.logger.error("Failed to start audio input - check microphone access")
            print("\nMicrophone access failed. Please check your audio setup or run in text mode.")
            return False
            
        try:
            while self.running:
                print("\nListening... (Press Ctrl+C to exit)")
                
                # Listen for command
                transcript, audio_data = self.audio_system.listen_and_transcribe()
                
                if not transcript:
                    print("No speech detected. Please try again.")
                    continue
                    
                print(f"\nYou said: {transcript}")
                
                # Process input
                entry = await self.process_input(transcript, transcript, audio_data)
                
                if entry.json_response and entry.json_response.get("response"):
                    print(f"\nGrace: {entry.json_response['response']}")
                elif entry.error:
                    print(f"\n[Error: {entry.error}]")
                else:
                    print("\nGrace: I couldn't process that input. Please try again.")
                    
        except KeyboardInterrupt:
            self.logger.info("Voice mode interrupted")
        except Exception as e:
            self.logger.error(f"Voice mode error: {e}", exc_info=True)
            print(f"\nVoice mode encountered an error: {e}")
            print("Falling back to text mode...")
            await self.run_text_mode()
        finally:
            # Stop audio input
            if self.audio_initialized:
                self.audio_system.stop_listening()
            
        return True
        
    async def run_text_mode(self):
        """Run in text input mode with console interface."""
        self.logger.info("Starting text mode...")
        
        try:
            while self.running:
                # Get user input
                try:
                    user_input = input("\nYou: ").strip()
                except EOFError:
                    break
                    
                if user_input.lower() in ["exit", "quit", "bye", "/exit"]:
                    self.logger.info("Shutdown requested by user")
                    break
                    
                if not user_input:
                    continue
                    
                # Process input  
                entry = await self.process_input(user_input)
                
                if entry.json_response and entry.json_response.get("response"):
                    print(f"\nGrace: {entry.json_response['response']}")
                elif entry.error:
                    print(f"\n[Error: {entry.error}]")
                else:
                    print("\nGrace: I couldn't process that input. Please try again.")
                    
        except KeyboardInterrupt:
            self.logger.info("Text mode interrupted")
        except Exception as e:
            self.logger.error(f"Text mode error: {e}", exc_info=True)
            
        return True
        
    async def run_hybrid_mode(self):
        """Run in hybrid mode supporting both voice and text input."""
        self.logger.info("Starting hybrid mode...")
        
        # Start audio input in background if available
        audio_initialized = False
        if self.audio_initialized:
            audio_initialized = self.audio_system.start_listening()
            if not audio_initialized:
                print("\nWARNING: Failed to initialize audio system. Voice commands won't work.")
                print("You can still try initializing it later with 'listen', 'voice', or 'mic'")
        else:
            print("\nWARNING: Audio system not available. Voice commands won't work.")
        
        print("\n=== Grace Hybrid Mode ===")
        print("Type your message or say 'listen', 'voice', or 'mic' to use voice input")
        print("Type 'exit' to quit")
        
        try:
            while self.running:
                # Get user input
                try:
                    user_input = input("\nYou: ").strip()
                except EOFError:
                    break
                    
                if user_input.lower() in ["exit", "quit", "bye", "/exit"]:
                    self.logger.info("Shutdown requested by user")
                    break
                    
                if not user_input:
                    continue
                    
                # Check for voice command keywords
                voice_keywords = ["listen", "voice", "mic", "microphone", 
                                "record", "speak", "audio", "talk"]
                
                if any(kw == user_input.lower() for kw in voice_keywords):
                    if not self.audio_initialized:
                        print("⚠️ Audio system not initialized. Voice commands unavailable.")
                        continue
                        
                    # Attempt to initialize audio if not already initialized
                    if not audio_initialized:
                        print("⚠️ Attempting to initialize audio system...")
                        audio_initialized = self.audio_system.start_listening()
                        if not audio_initialized:
                            print("⚠️ Could not initialize audio. Please check your microphone.")
                            continue
                    
                    # Clear visual indicator for voice mode
                    print("\n" + "-"*40)
                    print("🎤 VOICE MODE ACTIVATED - Speak now...")
                    print("-"*40)
                    
                    transcript, audio_data = self.audio_system.listen_and_transcribe()
                    
                    if transcript:
                        print(f"You said: {transcript}")
                        entry = await self.process_input(transcript, transcript, audio_data)
                    else:
                        print("❌ No speech detected. Returning to text mode.")
                        continue
                else:
                    # Process text input
                    entry = await self.process_input(user_input)
                
                if entry.json_response and entry.json_response.get("response"):
                    print(f"\nGrace: {entry.json_response['response']}")
                elif entry.error:
                    print(f"\n[Error: {entry.error}]")
                else:
                    # Show raw response instead of generic message
                    if hasattr(entry, 'model_response') and entry.model_response:
                        print(f"\nGrace (raw output): {entry.model_response}")
                    else:
                        print("\nGrace: I couldn't process that input. Please try again.")
                    
        except KeyboardInterrupt:
            self.logger.info("Hybrid mode interrupted")
        except Exception as e:
            self.logger.error(f"Hybrid mode error: {e}", exc_info=True)
            print(f"\nHybrid mode encountered an error: {e}")
            print("Falling back to text mode...")
            await self.run_text_mode()
        finally:
            # Stop audio input
            if self.audio_initialized:
                self.audio_system.stop_listening()
            
        return True
        
    async def check_system_health(self) -> Dict:
        """
        Check the health of all subsystems and return a status report.
        
        Returns:
            Dictionary of system health status
        """
        health = {
            "llm": {"status": "unknown", "error": None},
            "audio": {"status": "unknown", "error": None},
            "memory": {"status": "unknown", "error": None},
            "ovos": {"status": "unknown", "error": None}
        }
        
        # Check LLM
        if self.llm_initialized and self.llama_model and self.llama_model.model:
            health["llm"]["status"] = "ok"
        else:
            health["llm"]["status"] = "failed"
            health["llm"]["error"] = "Model not loaded"
            
        # Check audio
        if self.audio_initialized:
            if self.audio_system.whisper_model:
                health["audio"]["status"] = "ok"
            else:
                health["audio"]["status"] = "partial"
                health["audio"]["error"] = "Whisper model not loaded"
        else:
            health["audio"]["status"] = "not_initialized"
            health["audio"]["error"] = "Audio system initialization failed"
                
        # Check memory
        if self.memory_initialized:
            try:
                mem_stats = await self.memory_system.get_memory_stats()
                health["memory"]["status"] = "ok"
                health["memory"]["stats"] = mem_stats
            except Exception as e:
                health["memory"]["status"] = "error"
                health["memory"]["error"] = str(e)
        else:
            health["memory"]["status"] = "not_initialized"
            health["memory"]["error"] = "Memory system initialization failed"
            
        # Check OVOS
        if self.ovos_initialized:
            if self.ovos_interface.is_connected():
                health["ovos"]["status"] = "connected"
            else:
                # Get detailed connection status
                if hasattr(self.ovos_interface, 'get_connection_status'):
                    status = self.ovos_interface.get_connection_status()
                    health["ovos"]["status"] = status.get("status", "not_connected")
                    if status.get("last_error"):
                        health["ovos"]["error"] = status.get("last_error")
                    if status.get("disabled", False):
                        health["ovos"]["status"] = "disabled"
                else:
                    health["ovos"]["status"] = "not_connected"
        else:
            health["ovos"]["status"] = "not_initialized"
            health["ovos"]["error"] = "OVOS interface initialization failed"
                    
        return health
    
    def print_memory_status_dashboard(self):
        """Print the memory system status dashboard."""
        try:
            if hasattr(self, 'memory_system') and self.memory_initialized:
                status_text = self.memory_system.print_memory_status_dashboard()
                print(status_text)
            else:
                print("\nMemory system not initialized. Status dashboard unavailable.")
        except Exception as e:
            print(f"\nError displaying memory status dashboard: {e}")

    async def maintenance_task(self):
        """Run periodic maintenance tasks for memory systems."""
        if not self.memory_initialized:
            return
            
        try:
            # Run memory system maintenance
            await self.memory_system._check_and_run_maintenance()
            self.logger.info("Completed scheduled maintenance tasks")
        except Exception as e:
            self.logger.error(f"Error during maintenance: {e}")
        
    async def run(self):
        """Main run loop with mode selection."""
        self.running = True
        self.logger.info("Grace AI System is ready!")
        
        # Load reference materials (unless in amnesia mode)
        file_count = 0
        if self.memory_initialized and not self.amnesia_mode:
            file_count = await self.load_reference_materials()

        # Set up periodic maintenance tasks
        maintenance_interval = self.config.get('system', {}).get('maintenance_interval_hours', 24)
        maintenance_task = asyncio.create_task(self._periodic_maintenance(maintenance_interval))
        
        # Check system health
        health = await self.check_system_health()
        
        # Print system status
        print("\n=== Grace AI System Status ===")
        print(f"LLaMA Model: {'Loaded' if health['llm']['status'] == 'ok' else 'Not loaded'}")
        print(f"Whisper Model: {'Loaded' if health['audio']['status'] == 'ok' else 'Not loaded'}")
        print(f"OVOS Connection: {'Connected' if health['ovos']['status'] == 'connected' else 'Not connected'}")
        if health['ovos']['status'] != 'connected' and health['ovos'].get('error'):
            print(f"  OVOS Error: {health['ovos'].get('error')}")
        print(f"Memory System: {'Active' if health['memory']['status'] == 'ok' else 'Not active'}", end='')
        if health['memory']['status'] == 'ok':
            print(f" with {file_count} reference files")
        else:
            print()
        print(f"Amnesia Mode: {'ACTIVE - No new memories will be stored' if self.amnesia_mode else 'Inactive'}")
        print(f"Debug Mode: {'Enabled' if self.debug_mode else 'Disabled'}")
        print("============================\n")
        
        # Display memory system dashboard in debug mode
        if self.debug_mode and self.memory_initialized:
            self.print_memory_status_dashboard()
        
        # Determine input mode
        input_mode = self.config.get('system', {}).get('input_mode', self.input_mode).lower()
        
        # Validate LLM is working - if not, we can't proceed
        if health['llm']['status'] != 'ok':
            print("ERROR: Language model not loaded. Grace cannot function properly.")
            print("Please check your configuration and model file path.")
            return False
        
        success = False
        try:
            if input_mode == 'voice':
                if health['audio']['status'] == 'ok':
                    success = await self.run_voice_mode()
                else:
                    print("WARNING: Audio system not initialized properly.")
                    print("Falling back to text mode.")
                    success = await self.run_text_mode()
            elif input_mode == 'both' or input_mode == 'hybrid':
                success = await self.run_hybrid_mode()
            else:
                # Default to text mode
                success = await self.run_text_mode()
                
        except Exception as e:
            self.logger.error(f"Runtime error: {e}", exc_info=True)
        finally:
            # Cancel maintenance task
            maintenance_task.cancel()
            try:
                await maintenance_task
            except asyncio.CancelledError:
                pass
                
            await self.shutdown()
            
        return success
    
    async def _periodic_maintenance(self, hours=24):
        """Run periodic maintenance at the specified interval."""
        while self.running:
            try:
                # Wait for the specified interval
                await asyncio.sleep(hours * 3600)  # Convert hours to seconds
                
                # Run maintenance if system is still running
                if self.running:
                    await self.maintenance_task()
            except asyncio.CancelledError:
                # Task was cancelled, exit loop
                break
            except Exception as e:
                self.logger.error(f"Error in periodic maintenance: {e}")
                # Wait a bit before trying again
                await asyncio.sleep(300)  # 5 minutes
            
    async def shutdown(self):
        """Clean shutdown with proper resource cleanup."""
        if not self.running:
            return
            
        self.running = False
        self.logger.info("Shutting down Grace AI System...")
        
        # Shutdown components in order with error handling
        # Audio system
        if self.audio_initialized:
            try:
                self.audio_system.stop()
            except Exception as e:
                self.logger.error(f"Error shutting down audio system: {e}")
            
        # OVOS interface
        if self.ovos_initialized:    
            try:
                self.ovos_interface.shutdown()
            except Exception as e:
                self.logger.error(f"Error shutting down OVOS interface: {e}")
            
        # Language model
        if self.llm_initialized:
            try:
                self.llama_model.shutdown()
            except Exception as e:
                self.logger.error(f"Error shutting down language model: {e}")
            
        # Memory system
        if self.memory_initialized:
            try:
                self.memory_system.shutdown()
            except Exception as e:
                self.logger.error(f"Error shutting down memory system: {e}")
            
        self.logger.info("Grace AI System shut down successfully")


async def main():
    """Entry point with command-line argument handling."""
    parser = argparse.ArgumentParser(description="Grace AI System")
    parser.add_argument("--config", type=str, help="Path to custom config file")
    parser.add_argument("--model", type=str, help="Override model path")
    parser.add_argument("--reference-dir", type=str, help="Directory with reference materials")
    parser.add_argument("--mode", type=str, choices=["text", "voice", "both"], default="text",
                       help="Input mode (text, voice, or both)")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    parser.add_argument("--no-verification", action="store_true", help="Disable response verification")
    parser.add_argument("--amnesia", action="store_true", help="Enable amnesia mode (no new memories)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging and output")
    parser.add_argument("--no-ovos", action="store_true", help="Disable OVOS integration")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio system")
    parser.add_argument("--no-tts", action="store_true", help="Disable text-to-speech")
    parser.add_argument("--memory-status", action="store_true", help="Display memory system status and exit")
    parser.add_argument("--hot-memory", type=int, default=4, help="Hot memory size in GB (default: 4)")
    parser.add_argument("--warm-memory", type=int, default=16, help="Warm memory size in GB (default: 16)")
    
    args = parser.parse_args()
    
    # Build configuration override
    config_override = {}
    
    # Override model path if provided
    if args.model:
        config_override.setdefault('llama', {})['model_path'] = args.model
        
    # Override reference directory if provided
    if args.reference_dir:
        config_override['reference_dir'] = args.reference_dir
        
    # Set input mode
    if args.mode:
        config_override['system'] = {'input_mode': args.mode}
        
    # Disable GPU if requested
    if args.no_gpu:
        config_override.setdefault('llama', {})['n_gpu_layers'] = 0
        config_override.setdefault('whisper', {})['device'] = 'cpu'
        config_override.setdefault('whisper', {})['compute_type'] = 'int8'
        
    # Disable verification if requested
    if args.no_verification:
        config_override.setdefault('memory', {})['enable_verification'] = False
        
    # Enable amnesia mode if requested
    if args.amnesia:
        config_override['amnesia_mode'] = True
        
    # Set debug mode
    if args.debug:
        config_override['debug'] = True
        
    # Disable OVOS if requested
    if args.no_ovos:
        config_override.setdefault('ovos', {})['disable_ovos'] = True
        
    # Disable audio system if requested
    if args.no_audio:
        config_override.setdefault('audio', {})['use_microphone'] = False
        
    # Disable TTS if requested
    if args.no_tts:
        config_override.setdefault('audio', {})['mute'] = True
        
    # Set memory limits
    if args.hot_memory:
        config_override.setdefault('memory', {})['hot_memory_gb'] = args.hot_memory
        
    if args.warm_memory:
        config_override.setdefault('memory', {})['warm_memory_gb'] = args.warm_memory
        
    # Add config file path if provided
    if args.config:
        config_override['config'] = args.config
    
    try:
        # Check for required dependencies first
        check_required_dependencies()
        
        # Handle memory status display mode
        if args.memory_status:
            print(get_memory_system_status_dashboard())
            return
        
        # Initialize and run Grace
        grace = GraceSystem(config_override)
        await grace.run()
    except ImportError as e:
        print(f"CRITICAL ERROR: Missing required dependencies: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Failed to start Grace: {e}", exc_info=True)
        print(f"CRITICAL ERROR: Failed to start Grace: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
Grace AI System - Language Model Integration
This module provides a wrapper for large language models with optimized
settings for locally running models like QWQ-32B.
"""
import logging
import json
import re
import time
import asyncio
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path

# Import utilities
from grace.utils.token_utils import estimate_tokens

# Try to import llama-cpp-python
try:
    from llama_cpp import Llama, LlamaGrammar
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None
    LlamaGrammar = None

# Add a sanitize function for markdown safety
def _sanitize_markdown(text):
    """Remove or escape markdown formatting that could lead to prompt injection."""
    if not text:
        return ""
    # Escape common markdown control characters
    sanitized = text.replace("`", "\\`").replace("*", "\\*").replace("_", "\\_")
    sanitized = sanitized.replace("#", "\\#").replace(">", "\\>")
    return sanitized

class LlamaWrapper:
    """
    Wrapper for llama-cpp-python model with enhanced features.
    
    Features:
    - Optimized settings for high-performance local models
    - Robust JSON response parsing
    - Context window management
    - Dynamic temperature adjustment
    - Graceful degradation if model fails to load
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the language model with the provided configuration.
        
        Args:
            config: Language model configuration
        """
        self.logger = logging.getLogger('grace.llama')
        self.config = config
        self.llama_config = config.get('llama', {})
        self.model = None
        self.json_grammar = None
        self._load_model()
        self._create_json_grammar()
        
        # Track generation requests for monitoring
        self.generation_count = 0
        self.total_tokens_generated = 0
        self.last_generation_time = 0
        
    def _load_model(self):
        """Load the language model with proper error handling and fallbacks."""
        if not LLAMA_CPP_AVAILABLE:
            self.logger.error("llama-cpp-python not available. Install with: CMAKE_ARGS=\"-DLLAMA_CUDA=on\" pip install llama-cpp-python")
            return
            
        try:
            model_path = self.llama_config.get('model_path')
            self.logger.info(f"Loading model from {model_path}")
            
            # Check if model file exists
            model_path_obj = Path(model_path)
            if not model_path_obj.exists():
                self.logger.error(f"Model file not found: {model_path}")
                return
                
            # Load model with optimal settings for the specified hardware
            self.model = Llama(
                model_path=model_path,
                n_ctx=self.llama_config.get('n_ctx', 32768),
                n_gpu_layers=self.llama_config.get('n_gpu_layers', 31),
                n_threads=self.llama_config.get('n_threads', 6),
                n_batch=self.llama_config.get('n_batch', 512),
                rope_freq_base=10000.0,
                rope_freq_scale=1.0,
                use_mlock=self.llama_config.get('use_mlock', True),
                use_mmap=self.llama_config.get('use_mmap', False),
                verbose=self.llama_config.get('verbose', False)
            )
            
            # Check if this is a QWQ model and log special parameters
            if 'QWQ' in model_path:
                self.logger.info("QWQ model detected, using recommended parameters: temp=0.6, top_p=0.95, top_k=30, min_p=0.0, presence_penalty=1.0")
                
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            
    def _create_json_grammar(self):
        """Create a JSON grammar for more reliable response parsing."""
        if not LLAMA_CPP_AVAILABLE or LlamaGrammar is None:
            return
            
        try:
            # Define a simplified JSON grammar to ensure valid responses
            json_grammar_text = r"""
            root ::= object
            value ::= object | array | string | number | ("true" | "false" | "null")
            object ::= "{" (pair ("," pair)*)? "}"
            pair ::= string ":" value
            array ::= "[" (value ("," value)*)? "]"
            string ::= "\"" ([^"\\] | ("\\" (["\\/bfnrt] | ("u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))))* "\""
            number ::= [0-9]+ ("." [0-9]+)? (("e" | "E") ([+-]? [0-9]+))?
            """
            
            self.json_grammar = LlamaGrammar.from_string(json_grammar_text)
            self.logger.debug("JSON grammar created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create JSON grammar: {e}")
            self.json_grammar = None
    
    async def generate_async(self, prompt: str, max_tokens: int = None,
                          temperature: float = None, use_grammar: bool = True,
                          top_p: float = None, top_k: int = None, 
                          presence_penalty: float = None,
                          min_p: float = None) -> str:
        """
        Generate a response from the language model asynchronously.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            use_grammar: Use JSON grammar for response
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            presence_penalty: Presence penalty for repetition reduction
            min_p: Minimum probability for nucleus sampling
            
        Returns:
            Generated text response
        """
        # Run generation in a thread to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.generate(prompt, max_tokens, temperature, use_grammar, 
                                  top_p, top_k, presence_penalty, min_p)
        )
        
    def generate(self, prompt: str, max_tokens: int = None,
                temperature: float = None, use_grammar: bool = True,
                top_p: float = None, top_k: int = None, 
                presence_penalty: float = None,
                min_p: float = None) -> str:
        """
        Generate a response from the language model.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            use_grammar: Use JSON grammar for response
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            presence_penalty: Presence penalty for repetition reduction
            min_p: Minimum probability for nucleus sampling
            
        Returns:
            Generated text response
        """
        if not self.model:
            self.logger.error("Model not loaded")
            return "Error: Model not loaded properly. Please check the logs for details."
            
        # Set default parameters
        max_tokens = max_tokens or self.llama_config.get('max_tokens', 2048)
        
        # Get parameters with QWQ model optimizations if needed
        is_qwq = 'QWQ' in self.llama_config.get('model_path', '')
        
        # Apply QWQ recommended parameters if this is a QWQ model
        if is_qwq:
            temperature = temperature or self.llama_config.get('temperature', 0.6)
            top_p = top_p or self.llama_config.get('top_p', 0.95)
            top_k = top_k or self.llama_config.get('top_k', 30)
            min_p = min_p or self.llama_config.get('min_p', 0.0)
            presence_penalty = presence_penalty or self.llama_config.get('presence_penalty', 1.0)
        else:
            temperature = temperature or self.llama_config.get('temperature', 0.7)
            top_p = top_p or self.llama_config.get('top_p', 0.9)
            top_k = top_k or self.llama_config.get('top_k', 40)
            min_p = min_p or self.llama_config.get('min_p', 0.05)
            presence_penalty = presence_penalty or self.llama_config.get('presence_penalty', 0.0)
            
        try:
            # Track generation start time
            start_time = time.time()
            
            # Adjust parameters dynamically based on prompt
            adjusted_params = self._adjust_parameters(prompt, max_tokens, temperature)
            max_tokens = adjusted_params['max_tokens']
            temperature = adjusted_params['temperature']
            
            # Validate parameters are within acceptable ranges
            temperature = max(0.01, min(2.0, temperature))
            top_p = max(0.01, min(1.0, top_p))
            top_k = max(1, min(100, top_k))
            min_p = max(0.0, min(1.0, min_p))
            presence_penalty = max(-2.0, min(2.0, presence_penalty))
            
            # Define stop sequences - do not include </think> as we want the model to generate it
            # Fixed: Added "USER:" to stop sequences for QWQ-32B which often emits all-caps USER
            self.stop_sequences = ["<|end_user|>"]
            
            # Log prompt for debugging
            self.logger.debug(f"Last 100 chars of prompt: {prompt[-100:]}")
            
            # For QWQ models, avoid using grammar as it seems to confuse it
            use_grammar_for_this_request = use_grammar and not self.llama_config.get('disable_grammar', False)
            if is_qwq:
                use_grammar_for_this_request = False
                
            grammar = self.json_grammar if use_grammar_for_this_request else None
            
            # Log generation parameters for debugging
            self.logger.debug(f"Generation parameters: temp={temperature}, top_p={top_p}, top_k={top_k}, min_p={min_p}, presence_penalty={presence_penalty}")
            
            response = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                presence_penalty=presence_penalty,
                stop=self.stop_sequences,
                grammar=grammar,
                echo=False
            )
            
            # Calculate generation time
            generation_time = time.time() - start_time
            self.last_generation_time = generation_time
            
            # Update tracking metrics
            self.generation_count += 1
            
            # Log a sample of the response for debugging
            if response and 'choices' in response and len(response['choices']) > 0:
                sample = response['choices'][0].get('text', '')[:100]
                tokens_generated = len(response.get('usage', {}).get('completion_tokens', 0))
                self.total_tokens_generated += tokens_generated
                self.logger.debug(f"Model response sample: {sample}")
                self.logger.debug(f"Generated {tokens_generated} tokens in {generation_time:.2f} seconds")
                
            if response and 'choices' in response and len(response['choices']) > 0:
                return response['choices'][0].get('text', '')
                
            return ""
        except Exception as e:
            self.logger.error(f"Generation error: {e}")
            
            # Provide graceful fallback
            return f"I apologize, but I encountered a technical issue while processing your request. Please try again."
            
    def _adjust_parameters(self, prompt: str, max_tokens: int, temperature: float) -> Dict:
        """
        Dynamically adjust generation parameters based on the prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Requested max tokens
            temperature: Requested temperature
            
        Returns:
            Dictionary of adjusted parameters
        """
        # Estimate tokens used in prompt
        prompt_tokens = estimate_tokens(prompt)
        
        # Adjust max_tokens based on context length
        context_length = self.llama_config.get('n_ctx', 32768)
        available_tokens = max(context_length - prompt_tokens - 50, 100)  # 50 token buffer
        max_tokens = min(max_tokens, available_tokens)
        
        # Adjust temperature based on task and model
        adjusted_temp = temperature
        is_qwq = 'QWQ' in self.llama_config.get('model_path', '')
        
        # Optimization for QWQ models: Lower temperature for structured outputs
        if is_qwq:
            if "<think>" in prompt:
                adjusted_temp = min(temperature, 0.6)  # Lower temperature for more precise responses
                
            # Lower temperature for critical tasks
            if "CRITICAL" in prompt or "IMPORTANT" in prompt:
                adjusted_temp = min(temperature, 0.5)  # More conservative
                
            # Slightly higher temperature for creative tasks
            if any(term in prompt.lower() for term in ["creative", "story", "imagine", "generate ideas"]):
                adjusted_temp = max(temperature, 0.65)  # More creative but still controlled
        else:
            # Standard temperature adjustments for non-QWQ models
            if "<think>" in prompt:
                adjusted_temp = min(temperature, 0.7)
                
            # Lower temperature for critical tasks
            if "CRITICAL" in prompt or "IMPORTANT" in prompt:
                adjusted_temp = min(temperature, 0.4)  # More conservative
                
            # Higher temperature for creative tasks
            if any(term in prompt.lower() for term in ["creative", "story", "imagine", "generate ideas"]):
                adjusted_temp = max(temperature, 0.7)  # More creative
                
        return {
            'max_tokens': max_tokens,
            'temperature': adjusted_temp
        }
        
    def parse_response(self, response: str) -> Tuple[str, Dict]:
        """
        Parse model response to extract thinking process and JSON.
        
        Args:
            response: Raw model response
            
        Returns:
            Tuple of (thinking, json_response)
        """
        thinking = ""
        json_response = {"response": "I understand your request."}
        
        if not response:
            self.logger.debug("Empty response from model")
            return thinking, json_response
            
        # Debug output for response parsing
        self.logger.debug(f"Parsing response of length {len(response)} chars")
        
        # Check if this is a QWQ model - they have specific patterns
        is_qwq = 'QWQ' in self.llama_config.get('model_path', '')
        
        # QWQ models often output raw thinking text, no JSON, and no </think> tags
        # Let's detect this and handle it properly
        is_thinking_output = False
        
        # Check if this looks like thinking text rather than JSON
        if len(response) > 20 and not response.strip().startswith('{"'):
            # Look for typical thinking patterns
            thinking_patterns = [
                "The user is asking",
                "The user wants to know",
                "The user has requested",
                "I need to provide",
                "I should explain",
                "Let me analyze",
                "I'll respond with"
            ]
            
            if any(pattern in response for pattern in thinking_patterns):
                is_thinking_output = True
                self.logger.debug("Detected thinking text output instead of JSON")
                
        # Special case: QWQ model is outputting thinking text directly
        if is_thinking_output:
            thinking = response.strip()
            
            # Extract a sensible response from the thinking
            extracted_response = self._extract_response_from_thinking(thinking)
            if extracted_response:
                json_response["response"] = extracted_response
                return thinking, json_response
        
        # Extract thinking section if present
        if "</think>" in response:
            parts = response.split("</think>", 1)
            thinking = parts[0].strip()
            if "<think>" in thinking:
                thinking = thinking.split("<think>", 1)[1].strip()
            json_text = parts[1].strip() if len(parts) > 1 else ""
            
            self.logger.debug(f"Extracted thinking section. JSON text: {json_text[:100]}...")
            
            # Try to extract JSON
            if json_text:
                # First attempt: Find the most complete JSON object using a proper state machine
                try:
                    parsed_json = self._extract_json_object(json_text)
                    if parsed_json and "response" in parsed_json:
                        json_response = parsed_json
                        self.logger.debug("JSON extracted successfully using state machine")
                except Exception as e:
                    self.logger.debug(f"State machine JSON extraction failed: {e}")
                
                # Second attempt: Try with regex for the whole JSON object
                if not json_response.get("response") or json_response["response"] == "I understand your request.":
                    try:
                        json_match = re.search(r'(\{.*\})', json_text, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(1)
                            # Clean up common issues
                            json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                            
                            # Try to parse the JSON
                            parsed_json = json.loads(json_str)
                            if "response" in parsed_json:
                                json_response = parsed_json
                                self.logger.debug("JSON extracted successfully using regex")
                    except Exception as e:
                        self.logger.debug(f"Regex JSON extraction failed: {e}")
                
                # Third method: Extract field by field with regex if needed
                if not json_response.get("response") or json_response["response"] == "I understand your request.":
                    try:
                        response_match = re.search(r'"response"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', json_text)
                        if response_match:
                            response_value = response_match.group(1).replace('\\"', '"')
                            json_response["response"] = response_value
                            self.logger.debug(f"Extracted response field: {response_value[:30]}...")
                            
                        # Extract other fields if available
                        for field in ["messagebus", "utterance", "memory_add", "memory_search", "command_result", "speak", "ovos_command", "system_command"]:
                            # Properly escaped regex
                            field_match = re.search(f'"{field}"\\s*:\\s*("([^"]*(?:\\\\.[^"]*)*)"|(\\{{\\s*[^{{}}]*\\s*\\}})|(\\[\\s*[^\\[\\]]*\\s*\\]))', json_text)
                            
                            if field_match:
                                try:
                                    value = field_match.group(1)
                                    
                                    if value.startswith('"') and value.endswith('"'):
                                        # String value - remove quotes and handle escapes
                                        json_response[field] = value[1:-1].replace('\\"', '"')
                                    elif (value.startswith('{') and value.endswith('}')) or (value.startswith('[') and value.endswith(']')):
                                        # Object or array value - parse as JSON
                                        try:
                                            json_response[field] = json.loads(value)
                                        except json.JSONDecodeError:
                                            # If JSON parsing fails, store as string
                                            json_response[field] = value
                                    else:
                                        json_response[field] = value
                                        
                                    self.logger.debug(f"Extracted {field} field")
                                except Exception as e:
                                    self.logger.debug(f"Failed to extract {field}: {e}")
                    except Exception as e:
                        self.logger.debug(f"Field extraction method failed: {e}")
            else:
                self.logger.warning("No JSON text found after </think> tag")
                
                # If no JSON but we have text after thinking, use that as the response
                if len(parts) > 1 and parts[1].strip():
                    json_response["response"] = parts[1].strip()
        else:
            self.logger.warning("No </think> tag found in response")
            
            # If we're getting JSON-like content, try to parse it
            if response.strip().startswith('{"') and '}' in response:
                try:
                    # Try with the state machine extractor
                    parsed_json = self._extract_json_object(response)
                    if parsed_json and "response" in parsed_json:
                        return "", parsed_json
                except Exception:
                    # Fall back to simple regex
                    try:
                        json_match = re.search(r'(\{.*\})', response, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(1)
                            parsed_json = json.loads(json_str)
                            
                            if "response" in parsed_json:
                                return "", parsed_json
                    except json.JSONDecodeError:
                        pass
                    
            # If not thinking output, treat as regular content
            if not is_thinking_output:
                # Use the raw response as-is
                json_response["response"] = response.strip()
        
        # For QWQ models, if we still don't have a meaningful response,
        # attempt a more aggressive extraction from any available text
        if is_qwq and (not json_response.get("response") or json_response["response"] == "I understand your request."):
            extracted_text = self._extract_meaningful_content(response)
            if extracted_text:
                json_response["response"] = extracted_text
                
        return thinking, json_response
    
    def _extract_json_object(self, text: str) -> Optional[Dict]:
        """
        Extract a JSON object from text using a proper state machine.
        
        Args:
            text: Text containing a JSON object
            
        Returns:
            Parsed JSON object or None if extraction fails
        """
        # Find the start of the JSON object
        start_idx = text.find('{')
        if start_idx == -1:
            return None
            
        # State machine variables
        stack = []  # Track nested braces, brackets
        in_string = False
        escaped = False
        
        # Scan through the text
        for i in range(start_idx, len(text)):
            char = text[i]
            
            # Handle escape sequences in strings
            if in_string:
                if escaped:
                    escaped = False
                    continue
                    
                if char == '\\':
                    escaped = True
                    continue
                    
                if char == '"':
                    in_string = False
                continue
                
            # Process non-string characters
            if char == '"':
                in_string = True
            elif char == '{' or char == '[':
                stack.append(char)
            elif char == '}':
                if not stack or stack[-1] != '{':
                    # Mismatched braces
                    return None
                stack.pop()
                # If stack is empty, we've found the complete object
                if not stack and text[start_idx] == '{':
                    json_str = text[start_idx:i+1]
                    try:
                        # Clean up common issues
                        json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        return None
            elif char == ']':
                if not stack or stack[-1] != '[':
                    # Mismatched brackets
                    return None
                stack.pop()
        
        # If we reached the end without completing the object
        return None
        
    def _extract_meaningful_content(self, text: str) -> str:
        """
        Extract any meaningful content from text, even if not well-formatted.
        This is a last resort for QWQ models that sometimes don't follow the format.
        
        Args:
            text: Raw text to process
            
        Returns:
            Extracted meaningful content
        """
        if not text or len(text) < 10:
            return ""
            
        # Split text into lines and remove empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if not lines:
            return ""
            
        # Try to identify response-like lines
        # Skip lines that look like thinking or system content
        response_lines = []
        
        for line in lines:
            # Skip lines that look like system or thinking content
            if any(marker in line.lower() for marker in [
                "user is", "i need to", "i should", "let me", "thinking:",
                "system:", "assistant:", "<think", "</think", "user:", "human:"
            ]):
                continue
                
            # Prefer complete sentences
            if len(line) > 15 and any(c in line for c in ['.', '!', '?']):
                response_lines.append(line)
                
        # If we found response-like lines, return them
        if response_lines:
            return " ".join(response_lines)
            
        # If no good response lines, use the last non-empty line as fallback
        return lines[-1]
        
    def _extract_response_from_thinking(self, thinking_text: str) -> str:
        """
        Extract a user-facing response from thinking text.
        
        Args:
            thinking_text: The thinking/reasoning text
            
        Returns:
            A clean response suitable for the user
        """
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
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        model_info = {
            "loaded": self.model is not None,
            "model_path": self.llama_config.get('model_path', 'unknown'),
            "n_ctx": self.llama_config.get('n_ctx', 32768),
            "generation_count": self.generation_count,
            "total_tokens_generated": self.total_tokens_generated,
            "last_generation_time": self.last_generation_time,
            "is_qwq": 'QWQ' in self.llama_config.get('model_path', '')
        }
        
        return model_info
        
    def shutdown(self):
        """Clean shutdown of the language model."""
        self.logger.info("Shutting down language model")
        
        # Free memory
        try:
            del self.model
            self.model = None
        except Exception as e:
            self.logger.debug(f"Error closing language model: {e}")

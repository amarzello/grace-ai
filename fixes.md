# Grace AI Codebase Overhaul: Fixing Today's Bugs for Tomorrow's Features

The Grace AI project requires several critical fixes to address compatibility issues, resource leaks, and integration patterns across its memory, LLM, audio, and OVOS subsystems. This comprehensive guide provides specific solutions with modern implementation patterns, ensuring your codebase is both stable and future-ready.

## Memory system: modern patterns for hybrid storage

Neo4j, Qdrant, and SQLite integration issues represent the most critical memory system challenges. Here's how to fix them using current best practices.

### Neo4j integration modernization

The Neo4j Python driver (now at v5.28.1) requires significant updates to connection and transaction patterns:

```python
from neo4j import GraphDatabase

# Create a single driver instance per application (outdated pattern fixed)
driver = GraphDatabase.driver(
    "neo4j+s://your-instance.neo4j.io",  # Use neo4j+s:// for secure connections
    auth=("username", "password"),
    max_connection_lifetime=30 * 60,
    max_connection_pool_size=50, 
    connection_acquisition_timeout=2.0  # Fail fast if connection can't be acquired
)

# Verify connectivity during startup
driver.verify_connectivity()

# Modern transaction pattern (replacing deprecated session.read_transaction)
def get_person_data(tx, name):
    result = tx.run("MATCH (p:Person {name: $name}) RETURN p", name=name)
    return result.single()

with driver.session(database="neo4j") as session:
    try:
        # For read operations
        person = session.execute_read(get_person_data, "Alice")
        
        # For write operations (automatically retried on transient errors)
        session.execute_write(create_person_tx, "Bob")
    except Neo4jError as error:
        if error.code == "Neo.ClientError.Schema.ConstraintValidationFailed":
            # Handle specific error types
            pass
        else:
            # Log and handle other errors
            raise
```

**Key fixes:**
- Replace `bolt://` with `neo4j+s://` for secure connections
- Replace deprecated `session.read_transaction()` with `session.execute_read()`
- Add proper error handling with specific Neo4j error types
- Use proper resource management with context managers

### Qdrant vector database update

The Qdrant vector database client requires modernization for proper vector storage, retrieval, and resource management:

```python
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# Initialize client with proper connection and timeout parameters
client = QdrantClient(
    url="https://your-instance.qdrant.io",
    api_key="your-api-key",
    timeout=10.0
)

# For high-volume insertions, use batch operations
points = [
    PointStruct(
        id=idx,
        vector=embedding_vector,  # Already converted to list
        payload={"text": f"Memory {idx}", "metadata": {"source": "conversation"}}
    )
    for idx in range(100)
]

# Use wait=True for synchronous operations when needed
client.upsert(
    collection_name="memories",
    points=points,
    wait=True  # Ensure data is committed before continuing
)

# For async operations, use AsyncQdrantClient
async def vector_search():
    async_client = AsyncQdrantClient(url="https://your-instance.qdrant.io")
    results = await async_client.search(
        collection_name="memories",
        query_vector=query_embedding,
        limit=5
    )
    return results
```

**Key fixes:**
- Implement proper connection parameters with timeouts
- Use modern batch operations for vector insertion
- Implement proper async patterns with AsyncQdrantClient
- Add proper error handling and resource cleanup

### SQLite memory system with async

The SQLite memory system requires proper async patterns and connection pooling:

```python
import aiosqlite
import asyncio
from contextlib import asynccontextmanager

@asynccontextmanager
async def sqlite_connection(db_path):
    """Context manager for SQLite connections with optimized settings."""
    conn = await aiosqlite.connect(db_path)
    try:
        # Configure for optimal performance
        await conn.execute("PRAGMA journal_mode = WAL")  # Write-Ahead Logging
        await conn.execute("PRAGMA synchronous = NORMAL")  # Balance durability and speed
        await conn.execute("PRAGMA foreign_keys = ON")
        yield conn
    finally:
        await conn.close()

class SQLitePool:
    """Connection pool for thread-safe SQLite access."""
    def __init__(self, database_path, max_connections=5):
        self.database_path = database_path
        self.max_connections = max_connections
        self._pool = asyncio.Queue(maxsize=max_connections)
        self._connections = 0

    async def initialize(self):
        for _ in range(self.max_connections):
            await self._add_connection()

    async def _add_connection(self):
        if self._connections >= self.max_connections:
            return
        
        conn = await aiosqlite.connect(self.database_path)
        await conn.execute("PRAGMA journal_mode = WAL")
        await conn.execute("PRAGMA synchronous = NORMAL")
        await conn.execute("PRAGMA foreign_keys = ON")
        self._connections += 1
        await self._pool.put(conn)

    @asynccontextmanager
    async def acquire(self):
        conn = await self._pool.get()
        try:
            yield conn
        finally:
            await self._pool.put(conn)

    async def close(self):
        while not self._pool.empty():
            conn = await self._pool.get()
            await conn.close()
            self._connections -= 1
```

**Key fixes:**
- Implement connection pooling for thread safety
- Use Write-Ahead Logging (WAL) mode for better concurrency
- Implement proper async context managers for resource management
- Add transaction handling with proper error recovery

### Hybrid memory architecture

Grace AI needs a hybrid memory system combining Neo4j, Qdrant, and SQLite:

```python
class HybridMemorySystem:
    """A hybrid memory system combining vector, graph, and relational databases."""
    
    def __init__(self, vector_db_url, graph_db_url, sqlite_path):
        # Initialize vector database (Qdrant)
        self.vector_db = QdrantClient(url=vector_db_url)
        
        # Initialize graph database (Neo4j)
        self.graph_db = GraphDatabase.driver(graph_db_url, auth=("neo4j", "password"))
        
        # Initialize relational database (SQLite)
        self.sqlite_path = sqlite_path
        self.sqlite_pool = None
    
    async def initialize(self):
        # Initialize SQLite pool
        self.sqlite_pool = SQLitePool(self.sqlite_path)
        await self.sqlite_pool.initialize()
        
        # Set up schema
        async with self.sqlite_pool.acquire() as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS memory_metadata (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP,
                    access_count INTEGER DEFAULT 0
                )
            """)
            await db.commit()
    
    async def store_memory(self, text, embedding, metadata=None):
        """Store a memory in the hybrid system."""
        # Generate a unique ID
        memory_id = str(uuid.uuid4())
        
        # 1. Store in vector database for semantic search
        try:
            self.vector_db.upsert(
                collection_name="memories",
                points=[PointStruct(
                    id=memory_id,
                    vector=embedding,
                    payload={"text": text, "metadata": metadata or {}}
                )]
            )
        except Exception as e:
            # Log error and continue with other storage
            logging.error(f"Vector DB storage failed: {e}")
        
        # 2. Store in graph database for relational queries
        try:
            with self.graph_db.session() as session:
                session.execute_write(
                    lambda tx: tx.run(
                        """
                        CREATE (m:Memory {id: $id, text: $text})
                        RETURN m
                        """,
                        id=memory_id,
                        text=text
                    )
                )
        except Exception as e:
            # Log error and continue with other storage
            logging.error(f"Graph DB storage failed: {e}")
        
        # 3. Store metadata in relational database
        try:
            async with self.sqlite_pool.acquire() as db:
                await db.execute(
                    """
                    INSERT INTO memory_metadata (id, type, created_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    """,
                    (memory_id, metadata.get("type", "general") if metadata else "general")
                )
                await db.commit()
        except Exception as e:
            # Log error
            logging.error(f"SQLite storage failed: {e}")
        
        return memory_id
    
    async def close(self):
        """Close all database connections."""
        # Vector DB doesn't need explicit closing
        
        # Close Neo4j driver
        self.graph_db.close()
        
        # Close SQLite pool
        if self.sqlite_pool:
            await self.sqlite_pool.close()
```

**Key fixes:**
- Implement proper error handling for each storage component
- Use connection pooling for SQLite
- Ensure proper resource cleanup
- Handle partial failures gracefully

## LLM integration: enhancing llama-cpp-python

The LLM integration requires updates to llama-cpp-python usage, JSON parsing, and prompt formatting.

### llama-cpp-python modern implementation

```python
from llama_cpp import Llama
import logging
import torch
import gc

class LLMManager:
    """Manager for LLM operations with proper resource handling."""
    
    def __init__(self, model_path=None, model_kwargs=None):
        self.model_path = model_path
        self.model_kwargs = model_kwargs or {}
        self.llm = None
    
    def initialize(self):
        """Initialize the LLM with proper error handling."""
        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=2048,           # Adjust context window based on needs
                n_batch=512,          # Optimize batch size for your hardware
                n_gpu_layers=-1,      # Use all possible GPU layers
                verbose=False,        # Disable in production
                offload_kqv=True,     # Offload KQV matrices to GPU
                use_mlock=True,       # Pin memory to prevent swapping
                chat_format="chatml"  # Use modern chat format
            )
            return True
        except RuntimeError as e:
            if "CUDA" in str(e):
                # Fall back to CPU if CUDA fails
                logging.warning(f"CUDA error: {e}. Falling back to CPU.")
                self.model_kwargs["n_gpu_layers"] = 0
                self.llm = Llama(
                    model_path=self.model_path,
                    **self.model_kwargs
                )
                return True
            else:
                logging.error(f"Failed to initialize LLM: {e}")
                return False
    
    def generate_text(self, prompt, max_tokens=512, temperature=0.7):
        """Generate text with proper error handling."""
        if not self.llm:
            raise RuntimeError("LLM not initialized")
            
        try:
            output = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                echo=False
            )
            return output["choices"][0]["text"]
        except Exception as e:
            logging.error(f"Text generation failed: {e}")
            raise
    
    def generate_chat_completion(self, messages, **kwargs):
        """Generate chat completion with modern chat format."""
        if not self.llm:
            raise RuntimeError("LLM not initialized")
            
        try:
            response = self.llm.create_chat_completion(
                messages=messages,
                **kwargs
            )
            return response
        except Exception as e:
            logging.error(f"Chat completion failed: {e}")
            raise
    
    def close(self):
        """Properly close and clean up resources."""
        if hasattr(self, 'llm') and self.llm:
            del self.llm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
```

**Key fixes:**
- Implement proper resource management and cleanup
- Add error handling for CUDA/GPU issues with fallback
- Use modern chat format APIs
- Implement proper memory cleanup for GPU resources

### JSON parsing from LLM outputs

```python
import json
import re
from pydantic import BaseModel, Field, ValidationError

def safe_json_parse(json_string, default_value=None):
    """
    Safely parse a JSON string with multiple fallback strategies.
    Returns default_value if all parsing attempts fail.
    """
    if not json_string:
        return default_value
    
    # First attempt: direct parsing
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        pass
    
    # Second attempt: find JSON-like content between curly braces
    try:
        json_match = re.search(r'(\{.*\})', json_string, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
    except (json.JSONDecodeError, AttributeError):
        pass
    
    # Third attempt: repair common JSON issues
    try:
        # Replace single quotes with double quotes
        fixed = json_string.replace("'", '"')
        # Ensure proper quotes around keys
        fixed = re.sub(r'(\w+):', r'"\1":', fixed)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # All parsing attempts failed
    return default_value

class UserProfile(BaseModel):
    """Example Pydantic model for validation."""
    name: str = Field(..., description="The user's full name")
    age: int = Field(..., description="The user's age in years", ge=0, le=120)
    
    @classmethod
    def from_llm_output(cls, llm_output):
        """Create from LLM output with validation."""
        data = safe_json_parse(llm_output)
        if not data:
            raise ValueError("Failed to parse JSON from LLM output")
        
        try:
            return cls(**data)
        except ValidationError as e:
            # Log validation errors
            logging.error(f"Validation error: {e}")
            raise
```

**Key fixes:**
- Implement robust JSON parsing with multiple fallback strategies
- Use Pydantic for validation and type checking
- Add proper error handling and logging

### Optimized prompt formatting

```python
def create_structured_prompt(system_instruction, user_query, context=None):
    """Create a structured prompt with clear sections for better LLM performance."""
    # Start with system instruction
    components = [f"<|system|>\n{system_instruction}</|system|>"]
    
    # Add context if provided
    if context:
        components.append(f"<|context|>\n{context}</|context|>")
    
    # Add user query
    components.append(f"<|user|>\n{user_query}</|user|>")
    
    # Add assistant start token
    components.append("<|assistant|>")
    
    # Combine with newlines for clarity
    return "\n".join(components)

def create_json_extraction_prompt(text, schema):
    """Create a prompt optimized for JSON extraction."""
    # Format schema as readable JSON
    schema_desc = json.dumps(schema, indent=2)
    
    prompt = f"""
    <|system|>
    You are a precise data extraction assistant that ONLY outputs valid JSON.
    Your entire response must be parseable by the Python json.loads() function.
    </|system|>
    
    <|user|>
    Extract information from this text according to the specified JSON schema:
    
    TEXT TO PROCESS:
    {text}
    
    TARGET JSON SCHEMA:
    {schema_desc}
    
    Return ONLY valid JSON that follows the schema.
    </|user|>
    
    <|assistant|>
    """
    
    return prompt
```

**Key fixes:**
- Use modern prompt formats compatible with llama-cpp-python
- Structure prompts clearly with system/user/assistant roles
- Optimize JSON extraction prompts

## Audio system resource management

The audio system requires updates to resource management for faster-whisper, webrtcvad, and sounddevice.

### faster-whisper implementation

```python
from faster_whisper import WhisperModel
import torch
from contextlib import contextmanager
import gc

@contextmanager
def whisper_model_context(model_size="large-v3", device="auto", compute_type=None):
    """Context manager for proper whisper model resource management."""
    model = None
    try:
        # Auto-detect device if not specified
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Set default compute_type based on device
        if compute_type is None:
            compute_type = "float16" if device == "cuda" else "int8"
        
        # Create model
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        yield model
    finally:
        # Ensure cleanup
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

def transcribe_with_error_handling(audio_path, model_size="large-v3"):
    """Transcribe audio with comprehensive error handling."""
    try:
        # Determine optimal device and compute type
        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16"
        else:
            device = "cpu"
            compute_type = "int8"
        
        with whisper_model_context(model_size, device, compute_type) as model:
            # Use VAD filtering for better results
            segments, info = model.transcribe(
                audio_path,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            # Force evaluation to capture errors early
            segments_list = list(segments)
            return segments_list, info
            
    except Exception as e:
        logging.error(f"Transcription failed: {e}")
        # Attempt fallback with smaller model if OOM error
        if "out of memory" in str(e).lower() and device == "cuda":
            logging.warning("Attempting fallback to smaller model due to OOM")
            try:
                with whisper_model_context("base", device, "int8") as model:
                    segments, info = model.transcribe(audio_path)
                    return list(segments), info
            except Exception as fallback_error:
                logging.error(f"Fallback transcription failed: {fallback_error}")
        
        return None, None
```

**Key fixes:**
- Implement proper resource management with context managers
- Add CUDA memory cleanup
- Implement fallback strategies for OOM errors
- Use VAD filtering for better results

### Audio device resource management

```python
import sounddevice as sd
import threading
import queue
import atexit

class AudioStreamManager:
    """Audio stream manager with proper resource management."""
    
    _active_streams = set()
    _lock = threading.RLock()
    
    @classmethod
    def _register_stream(cls, stream):
        """Register a stream for global cleanup."""
        with cls._lock:
            cls._active_streams.add(stream)
    
    @classmethod
    def _unregister_stream(cls, stream):
        """Unregister a stream from global cleanup."""
        with cls._lock:
            if stream in cls._active_streams:
                cls._active_streams.remove(stream)
    
    @classmethod
    def cleanup_all_streams(cls):
        """Clean up all active streams."""
        with cls._lock:
            for stream in list(cls._active_streams):
                try:
                    if stream.active:
                        stream.stop()
                    stream.close()
                except Exception:
                    pass
            cls._active_streams.clear()
    
    def __init__(self, samplerate=44100, channels=2, dtype='float32'):
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.input_stream = None
        self.output_stream = None
    
    def start_input(self, callback=None, **kwargs):
        """Start input stream with proper error handling."""
        try:
            self.cleanup_input()  # Clean up any existing stream first
            self.input_stream = sd.InputStream(
                samplerate=self.samplerate,
                channels=self.channels,
                dtype=self.dtype,
                callback=callback,
                **kwargs
            )
            self.input_stream.start()
            self._register_stream(self.input_stream)
            return True
        except sd.PortAudioError as e:
            logging.error(f"Failed to start input stream: {e}")
            self.cleanup_input()
            return False
    
    def cleanup_input(self):
        """Clean up input stream resources."""
        if self.input_stream is not None:
            try:
                if self.input_stream.active:
                    self.input_stream.stop()
                self.input_stream.close()
                self._unregister_stream(self.input_stream)
            except Exception as e:
                logging.error(f"Error cleaning up input stream: {e}")
            self.input_stream = None
    
    def cleanup(self):
        """Clean up all resources."""
        self.cleanup_input()
        self.cleanup_output()
    
    def __del__(self):
        """Ensure cleanup on object destruction."""
        self.cleanup()

# Register cleanup at process exit
atexit.register(AudioStreamManager.cleanup_all_streams)
```

**Key fixes:**
- Implement proper resource cleanup with atexit handlers
- Use thread-safe stream management
- Add error handling for audio device operations
- Ensure streams are properly closed

### VAD Implementation

```python
import webrtcvad
import collections
import wave
import struct
import numpy as np
from contextlib import contextmanager

class VADProcessor:
    """Process audio with Voice Activity Detection."""
    
    def __init__(self, aggressiveness=3, sample_rate=16000, frame_duration_ms=30):
        """Initialize VAD with parameters."""
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        
        # Validate parameters
        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError(f"Sample rate must be 8000, 16000, 32000, or 48000 Hz")
        if frame_duration_ms not in (10, 20, 30):
            raise ValueError(f"Frame duration must be 10, 20, or 30 ms")
        
        # Calculate frame size in bytes
        self.frame_size = int(sample_rate * frame_duration_ms / 1000) * 2  # 2 bytes per sample
    
    def process_audio_file(self, audio_path):
        """Process an audio file and extract voiced segments."""
        try:
            # Read audio file
            with wave.open(audio_path, 'rb') as wf:
                # Validate audio format
                if wf.getnchannels() != 1:
                    raise ValueError("Audio must be mono")
                if wf.getsampwidth() != 2:
                    raise ValueError("Audio must be 16-bit")
                if wf.getframerate() != self.sample_rate:
                    raise ValueError(f"Audio must have {self.sample_rate}Hz sample rate")
                
                # Read all audio data
                pcm_data = wf.readframes(wf.getnframes())
            
            # Process frames
            frames = self._frame_generator(pcm_data)
            voiced_segments = self._vad_collector(frames)
            
            return voiced_segments
        except Exception as e:
            logging.error(f"VAD processing failed: {e}")
            return []
    
    def _frame_generator(self, audio_data):
        """Generate audio frames from PCM data."""
        offset = 0
        while offset + self.frame_size <= len(audio_data):
            yield audio_data[offset:offset + self.frame_size]
            offset += self.frame_size
    
    def _vad_collector(self, frames, padding_duration_ms=300):
        """Collect voiced segments from a stream of audio frames."""
        num_padding_frames = int(padding_duration_ms / self.frame_duration_ms)
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False
        voiced_segments = []
        
        for frame in frames:
            is_speech = self.vad.is_speech(frame, self.sample_rate)
            
            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                
                # Start collecting if enough voiced frames in buffer
                if num_voiced > 0.9 * ring_buffer.maxlen:
                    triggered = True
                    voiced_segments.append([])
                    for f, s in ring_buffer:
                        voiced_segments[-1].append(f)
                    ring_buffer.clear()
            else:
                # Continue collecting if still in voice segment
                voiced_segments[-1].append(frame)
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                
                # Stop collecting if enough unvoiced frames
                if num_unvoiced > 0.9 * ring_buffer.maxlen:
                    triggered = False
                    ring_buffer.clear()
        
        # Convert segments to byte strings
        return [''.join(segment) for segment in voiced_segments]
```

**Key fixes:**
- Implement proper parameter validation
- Add error handling for audio format issues
- Use collections.deque for efficient buffer implementation
- Implement proper frame handling with accurate sizing

## OVOS integration thread safety

The OVOS integration requires updates to ensure thread safety in the messagebus integration.

### Thread-safe messagebus integration

```python
import time
from threading import Lock, RLock
from ovos_bus_client import MessageBusClient, Message
import logging

class ThreadSafeOVOSComponent:
    """Thread-safe OVOS integration component."""
    
    def __init__(self, bus=None, name="GraceAI"):
        self.name = name
        self.bus = bus or MessageBusClient()
        self.running = False
        
        # Locks for thread safety
        self.sessions_lock = RLock()
        self.processing_lock = RLock()
        
        # Component state
        self.active_sessions = {}
        self.handlers = {}
    
    def start(self):
        """Start the component and connect to the messagebus."""
        # Register message handlers thread-safely
        with self.processing_lock:
            # Define and register handlers
            self.handlers = {
                'grace.ai.request': self.handle_request,
                'grace.ai.cancel': self.handle_cancel,
                'recognition.result': self.handle_recognition
            }
            
            # Register all handlers
            for message_type, handler in self.handlers.items():
                self.bus.on(message_type, handler)
        
        # Start the message bus client
        self.running = True
        self.bus.run_in_thread()
        logging.info(f"{self.name} component started")
    
    def stop(self):
        """Stop the component and disconnect from the messagebus."""
        # Set running flag first to prevent new operations
        self.running = False
        
        # Remove all handlers
        with self.processing_lock:
            for message_type in self.handlers:
                try:
                    self.bus.remove_all_listeners(message_type)
                except Exception as e:
                    logging.error(f"Error removing listener for {message_type}: {e}")
        
        # Close connection
        try:
            self.bus.close()
        except Exception as e:
            logging.error(f"Error closing message bus: {e}")
        
        logging.info(f"{self.name} component stopped")
    
    def handle_request(self, message):
        """Thread-safe handler for incoming requests."""
        if not self.running:
            return
            
        session_id = message.data.get('session_id') or str(time.time())
        
        try:
            # Thread-safely register the session
            with self.sessions_lock:
                if session_id in self.active_sessions:
                    # Session already exists, send error
                    response = message.response({
                        'success': False, 
                        'error': 'Session already exists'
                    })
                    self.bus.emit(response)
                    return
                
                # Create new session
                self.active_sessions[session_id] = {
                    'status': 'processing',
                    'created_at': time.time(),
                    'context': message.context
                }
            
            # Process the request protected by lock to prevent race conditions
            with self.processing_lock:
                result = self._process_request(message.data)
            
            # Send response with results
            response = message.response({
                'success': True,
                'session_id': session_id,
                'result': result
            })
            self.bus.emit(response)
            
        except Exception as e:
            logging.exception(f"Error handling request: {e}")
            # Send error response
            response = message.response({
                'success': False,
                'session_id': session_id,
                'error': str(e)
            })
            self.bus.emit(response)
            
        finally:
            # Clean up session
            with self.sessions_lock:
                if session_id in self.active_sessions:
                    self.active_sessions[session_id]['status'] = 'completed'
    
    def handle_cancel(self, message):
        """Cancel an active session with thread safety."""
        if not self.running:
            return
            
        session_id = message.data.get('session_id')
        if not session_id:
            self.bus.emit(message.response({
                'success': False,
                'error': 'No session_id provided'
            }))
            return
            
        with self.sessions_lock:
            if session_id in self.active_sessions:
                self.active_sessions[session_id]['status'] = 'cancelled'
                self.bus.emit(message.response({'success': True}))
            else:
                self.bus.emit(message.response({
                    'success': False,
                    'error': 'Session not found'
                }))
    
    def handle_recognition(self, message):
        """Process recognition results with thread safety."""
        if not self.running:
            return
            
        with self.processing_lock:
            # Thread-safe processing of recognition results
            self._process_recognition(message.data)
    
    def _process_request(self, data):
        """Process request data safely."""
        # Implementation here
        return {"processed": True}
    
    def _process_recognition(self, data):
        """Process recognition results safely."""
        # Implementation here
        pass
```

**Key fixes:**
- Use RLock for thread safety
- Implement proper handler registration/removal
- Add thread-safe session management
- Add checks for component running state

## System-wide error handling and resource management

Implement consistent error handling and resource management across the codebase.

### Error handling patterns

```python
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

# Comprehensive error logging system
import logging
import traceback
import json
import datetime

class ErrorLogger:
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
```

### Resource management patterns

```python
from contextlib import contextmanager, asynccontextmanager
import tempfile
import shutil

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
```

### Async/await patterns

```python
import asyncio
from contextlib import asynccontextmanager

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
```

### Dependencies and requirements updates

```
# Updated requirements.txt with all dependencies

# Core dependencies
neo4j==5.28.1  # Updated to latest Neo4j Python driver
qdrant-client==1.6.4  # Updated vector database client
llama-cpp-python==0.3.9  # Updated LLM interface

# Memory system
aiosqlite==0.19.0  # Async SQLite support
pydantic==2.5.3  # Data validation

# Audio system
faster-whisper==0.10.0  # Latest version
webrtcvad==2.0.10  # Voice activity detection
sounddevice==0.4.6  # Audio device management
numpy==1.26.4  # Required for audio processing

# OVOS integration
ovos-bus-client==0.0.7  # OVOS messagebus client
ovos-utils==0.0.36  # OVOS utilities

# General utilities
aiohttp==3.9.1  # Async HTTP client
requests==2.31.0  # HTTP client
python-dotenv==1.0.0  # Environment variable management
loguru==0.7.2  # Advanced logging
```

## Conclusion and integration advice

To successfully fix the Grace AI codebase, implement these changes in the following order:

1. First, update all dependencies in requirements.txt to the latest compatible versions
2. Implement the consistent error handling and resource management patterns across the codebase
3. Fix the memory system integration (Neo4j, Qdrant, SQLite) with proper resource management
4. Update the LLM integration with improved prompt formatting and JSON parsing
5. Implement proper audio system resource management
6. Fix OVOS integration with thread safety improvements

By following these implementation patterns, you'll address the specific issues identified in the Grace AI project while ensuring proper error handling, resource cleanup, and compatibility with the latest APIs. The hybrid memory system design provides a robust foundation for AI assistants, combining the strengths of graph, vector, and relational databases.

Remember to thoroughly test each component after updates, especially focusing on resource cleanup and error recovery. The provided code patterns are designed to work together as an integrated system, with consistent error handling and resource management approaches throughout the codebase.
# Grace AI System

A locally-run personal AI assistant with memory capabilities.

## Overview

Grace is designed to be a fully local, privacy-respecting AI assistant that provides a memory system for maintaining context and storing information over time.

This project is currently focused on building a robust memory system that can be integrated with language models, audio processing, and other components to create a complete AI assistant experience.

## Features

- **Memory System**: Hybrid memory system using vector databases and SQLite for persistent storage
- **Critical Memory**: Special handling for important information that must be preserved
- **Memory Search**: Intelligent search capabilities with relevance ranking
- **Privacy Mode**: Optional amnesia mode to prevent storing any memories

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd grace
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Basic usage:

```
python run.py
```

Options:
- `--config PATH`: Specify a custom configuration file
- `--amnesia`: Enable amnesia mode (no memory storage)
- `--debug`: Enable debug logging
- `--test-memory`: Run tests on the memory system
- `--wipe-memory`: Wipe all memory storage (will prompt for confirmation)

## Project Structure

```
grace/
├── memory/               # Memory system modules
│   ├── core.py           # Main memory system interface
│   ├── sqlite.py         # SQLite storage backend
│   ├── vector.py         # Vector storage using mem0
│   ├── critical.py       # Critical memory management
│   └── types.py          # Memory type definitions
├── utils/                # Utility modules
│   └── common.py         # Shared utilities and helpers
└── ...                   # (Other modules to be added)

run.py                    # Main entry point
requirements.txt          # Dependencies
```

## Configuration

The system uses a YAML configuration file located at `~/.grace/config/config.yaml`. A default configuration will be created on first run, which you can then modify.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

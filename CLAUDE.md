# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Course Materials RAG System** - a full-stack Retrieval-Augmented Generation application that allows users to query course materials and receive AI-powered responses with proper source attribution.

## Architecture

The system uses a **modular, three-tier architecture**:

### Backend (`/backend/`)
- **FastAPI** web framework with CORS and proxy middleware
- **RAG System Core**: Main orchestrator (`rag_system.py`)
- **Vector Storage**: ChromaDB with SentenceTransformers embeddings (`vector_store.py`)
- **AI Generation**: Anthropic Claude integration with tool calling (`ai_generator.py`)
- **Document Processing**: Handles PDF/DOCX/TXT files (`document_processor.py`)
- **Tool-Based Search**: Semantic search with course/lesson filtering (`search_tools.py`)
- **Session Management**: Conversation history tracking (`session_manager.py`)

### Frontend (`/frontend/`)
- **Vanilla JavaScript** SPA with marked.js for markdown rendering
- **Real-time chat interface** with loading states and source attribution
- **Course statistics sidebar** with collapsible sections
- **Suggested questions** for user guidance

### Data Models (`/backend/models.py`)
- **Course**: Title, description, lessons, instructor, URL
- **Lesson**: Number, title, content, URL
- **CourseChunk**: Processed text chunks for vector storage

## Development Commands

### Quick Start
```bash
chmod +x run.sh
./run.sh
```

### Manual Development
```bash
# Install dependencies (first time)
uv sync

# Start backend server
cd backend && uv run uvicorn app:app --reload --port 8000

# Application runs at:
# - Web Interface: http://localhost:8000
# - API Docs: http://localhost:8000/docs
```

### Environment Setup
Create `.env` file in root:
```
ANTHROPIC_API_KEY=your_key_here
```

## Key Technical Patterns

### RAG Query Flow
1. User query → FastAPI endpoint (`/api/query`)
2. RAG system creates AI prompt with tool definitions
3. Claude uses `search_course_content` tool with semantic matching
4. Vector store searches ChromaDB with course/lesson filtering
5. Search results formatted with source attribution
6. Claude synthesizes response using retrieved content
7. Response returned with clickable source links

### Tool-Based Search Architecture
- **CourseSearchTool**: Handles semantic search with course name fuzzy matching
- **ToolManager**: Registers and executes tools for AI agent
- **Source Tracking**: Last search sources stored for UI display
- **Flexible Filtering**: Supports course title and lesson number filters

### Vector Storage Strategy
- **SentenceTransformers**: `all-MiniLM-L6-v2` for embeddings
- **ChromaDB Collections**: Separate storage for course metadata vs content chunks
- **Smart Deduplication**: Avoids re-processing existing courses
- **Metadata Enrichment**: Course titles, lesson numbers, URLs stored as metadata

### Session Management
- **Conversation History**: Tracks user-assistant exchanges per session
- **Context Limits**: Configurable max history (default: 2 messages)
- **Session Creation**: Auto-generated UUIDs for frontend sessions

## Configuration (`/backend/config.py`)

Key settings:
- **ANTHROPIC_MODEL**: `claude-sonnet-4-20250514`
- **EMBEDDING_MODEL**: `all-MiniLM-L6-v2`
- **CHUNK_SIZE**: 800 characters
- **CHUNK_OVERLAP**: 100 characters
- **MAX_RESULTS**: 5 search results
- **MAX_HISTORY**: 2 conversation turns

## Document Processing

Supports: **PDF, DOCX, TXT** files
- Course documents placed in `/docs/` folder
- Auto-loaded on server startup
- Structured parsing extracts course metadata and lessons
- Text chunking with overlap for semantic search
- Duplicate detection prevents re-processing

## API Endpoints

- **POST** `/api/query` - Process user questions
- **GET** `/api/courses` - Get course statistics
- **Static files** served at `/` (frontend)

## Testing and Development

Since this is a RAG system with AI components:
- Test with sample course documents in `/docs/`
- Verify ChromaDB storage at `./backend/chroma_db/`
- Monitor API logs for tool usage and search results
- Test different question types (general vs course-specific)
- Validate source attribution and clickable links
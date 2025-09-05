# RAG System Query Flow Diagram

```
┌─────────────────┐
│    Frontend     │
│   (script.js)   │
└─────────────────┘
         │
         │ 1. User submits query
         │ POST /api/query
         │ { "query": "...", "session_id": "..." }
         ▼
┌─────────────────┐
│   FastAPI App   │
│    (app.py)     │
│                 │
│ query_documents()│
└─────────────────┘
         │
         │ 2. Create/get session
         │ 3. Call rag_system.query()
         ▼
┌─────────────────┐
│   RAG System    │
│ (rag_system.py) │
│                 │
│ query() method  │
└─────────────────┘
         │
         │ 4. Get conversation history
         │ 5. Prepare prompt + tools
         ▼
┌─────────────────┐
│  AI Generator   │
│(ai_generator.py)│
│                 │
│ Claude API +    │
│ Tool Manager    │
└─────────────────┘
         │
         │ 6. Claude decides to use tools
         │ 7. Execute CourseSearchTool
         ▼
┌─────────────────┐
│  Vector Store   │
│(vector_store.py)│
│                 │
│   ChromaDB      │
│   Semantic      │
│   Search        │
└─────────────────┘
         │
         │ 8. Return relevant chunks
         ▲
         │
┌─────────────────┐
│ Course Chunks   │
│                 │
│ "Course X       │
│ Lesson Y        │
│ content: ..."   │
└─────────────────┘
         │
         │ 9. Generate response using context
         ▼
┌─────────────────┐
│ Claude Response │
│                 │
│ + Sources List  │
└─────────────────┘
         │
         │ 10. Update session history
         │ 11. Return answer + sources
         ▼
┌─────────────────┐
│   Frontend      │
│                 │
│ Display answer  │
│ + collapsible   │
│ sources         │
└─────────────────┘
```

## Data Flow Details

### Request Flow (Frontend → Backend)
1. **User Input**: Types question in chat interface
2. **API Call**: POST to `/api/query` with JSON payload
3. **Session Management**: Create new session or use existing
4. **RAG Orchestration**: Main query processing logic
5. **AI Processing**: Claude with tool access for search

### Search & Retrieval Flow
6. **Tool Decision**: Claude decides when to search for information
7. **Vector Search**: Semantic similarity search in ChromaDB
8. **Context Retrieval**: Get relevant course content chunks
9. **Response Generation**: Claude creates answer using retrieved context

### Response Flow (Backend → Frontend)
10. **History Update**: Save conversation for context
11. **JSON Response**: Return structured answer + metadata
12. **UI Update**: Display formatted response with sources

## Key Components

- **Session Manager**: Maintains conversation context
- **Document Processor**: Chunks course content with context prefixes
- **Vector Store**: ChromaDB for semantic search
- **Tool Manager**: Enables Claude to search when needed
- **Course Search Tool**: Retrieves relevant content chunks
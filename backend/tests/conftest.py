"""Pytest configuration and shared fixtures for RAG system tests"""
import os
import sys
import tempfile
import shutil
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, MagicMock
import pytest

# Add backend directory to Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import Course, Lesson, CourseChunk
from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from ai_generator import AIGenerator
from rag_system import RAGSystem
from config import Config


@pytest.fixture
def temp_chroma_path():
    """Create temporary ChromaDB path for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_config(temp_chroma_path):
    """Test configuration with temporary paths"""
    config = Config()
    config.CHROMA_PATH = temp_chroma_path
    config.ANTHROPIC_API_KEY = "test-key"
    config.MAX_RESULTS = 3  # Smaller for testing
    return config


@pytest.fixture
def sample_course():
    """Sample course for testing"""
    return Course(
        title="Introduction to Machine Learning",
        course_link="https://example.com/ml-course",
        instructor="Dr. Smith",
        lessons=[
            Lesson(lesson_number=1, title="What is ML?", lesson_link="https://example.com/lesson1"),
            Lesson(lesson_number=2, title="Types of ML", lesson_link="https://example.com/lesson2"),
            Lesson(lesson_number=3, title="ML Algorithms", lesson_link="https://example.com/lesson3")
        ]
    )


@pytest.fixture
def sample_course_chunks(sample_course):
    """Sample course chunks for testing"""
    return [
        CourseChunk(
            content="Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="There are three main types of machine learning: supervised, unsupervised, and reinforcement learning.",
            course_title=sample_course.title,
            lesson_number=2,
            chunk_index=1
        ),
        CourseChunk(
            content="Popular ML algorithms include linear regression, decision trees, and neural networks.",
            course_title=sample_course.title,
            lesson_number=3,
            chunk_index=2
        )
    ]


@pytest.fixture
def mock_vector_store():
    """Mock VectorStore for testing search tools"""
    mock_store = Mock(spec=VectorStore)
    
    # Configure default successful search response
    mock_store.search.return_value = SearchResults(
        documents=["Machine learning is a subset of artificial intelligence."],
        metadata=[{
            'course_title': 'Introduction to Machine Learning', 
            'lesson_number': 1
        }],
        distances=[0.2],
        error=None
    )
    
    # Configure course outline response
    mock_store.get_course_outline.return_value = {
        'course_title': 'Introduction to Machine Learning',
        'course_link': 'https://example.com/ml-course',
        'lessons': [
            {'lesson_number': 1, 'lesson_title': 'What is ML?'},
            {'lesson_number': 2, 'lesson_title': 'Types of ML?'},
            {'lesson_number': 3, 'lesson_title': 'ML Algorithms'}
        ]
    }
    
    # Configure link methods
    mock_store.get_lesson_link.return_value = "https://example.com/lesson1"
    mock_store.get_course_link.return_value = "https://example.com/ml-course"
    
    return mock_store


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing AI generator"""
    mock_client = Mock()
    
    # Mock response without tool use
    mock_response = Mock()
    mock_response.content = [Mock(text="This is a test response")]
    mock_response.stop_reason = "end_turn"
    
    mock_client.messages.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_anthropic_tool_response():
    """Mock Anthropic response with tool usage"""
    mock_response = Mock()
    mock_response.stop_reason = "tool_use"
    
    # Mock tool use content block
    tool_block = Mock()
    tool_block.type = "tool_use"
    tool_block.name = "search_course_content"
    tool_block.id = "tool_123"
    tool_block.input = {"query": "machine learning", "course_name": "ML"}
    
    mock_response.content = [tool_block]
    return mock_response


@pytest.fixture
def mock_final_anthropic_response():
    """Mock final Anthropic response after tool execution"""
    mock_response = Mock()
    mock_response.content = [Mock(text="Based on the search results, machine learning is...")]
    return mock_response


@pytest.fixture 
def real_vector_store(test_config):
    """Real VectorStore instance for integration tests"""
    return VectorStore(
        chroma_path=test_config.CHROMA_PATH,
        embedding_model=test_config.EMBEDDING_MODEL,
        max_results=test_config.MAX_RESULTS
    )


@pytest.fixture
def populated_vector_store(real_vector_store, sample_course, sample_course_chunks):
    """Vector store populated with test data"""
    real_vector_store.add_course_metadata(sample_course)
    real_vector_store.add_course_content(sample_course_chunks)
    return real_vector_store


@pytest.fixture
def course_search_tool(mock_vector_store):
    """CourseSearchTool with mocked dependencies"""
    return CourseSearchTool(mock_vector_store)


@pytest.fixture
def course_outline_tool(mock_vector_store):
    """CourseOutlineTool with mocked dependencies"""
    return CourseOutlineTool(mock_vector_store)


@pytest.fixture
def tool_manager(course_search_tool, course_outline_tool):
    """ToolManager with registered tools"""
    manager = ToolManager()
    manager.register_tool(course_search_tool)
    manager.register_tool(course_outline_tool)
    return manager
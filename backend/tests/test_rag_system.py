"""Integration tests for RAGSystem"""

from unittest.mock import Mock, patch

import pytest
from models import Course, CourseChunk, Lesson
from rag_system import RAGSystem


class TestRAGSystemIntegration:
    """Test RAG system integration and complete query flow"""

    @pytest.fixture
    def mock_rag_system(self, test_config):
        """Create RAG system with mocked dependencies"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("rag_system.AIGenerator") as mock_ai_generator,
            patch("rag_system.SessionManager"),
        ):
            rag = RAGSystem(test_config)

            # Configure mocks
            rag.vector_store = mock_vector_store.return_value
            rag.ai_generator = mock_ai_generator.return_value

            # Mock AI generator to return simple response
            rag.ai_generator.generate_response.return_value = "Mocked AI response"

            return rag

    def test_rag_system_initialization(self, test_config):
        """Test that RAG system initializes all components correctly"""
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator"),
            patch("rag_system.SessionManager"),
        ):
            rag = RAGSystem(test_config)

            # Verify tools are registered
            assert "search_course_content" in rag.tool_manager.tools
            assert "get_course_outline" in rag.tool_manager.tools
            assert len(rag.tool_manager.tools) == 2

    def test_query_without_session(self, mock_rag_system):
        """Test query processing without session ID"""
        response, sources = mock_rag_system.query("What is machine learning?")

        assert response == "Mocked AI response"
        assert isinstance(sources, list)

        # Verify AI generator was called with correct parameters
        mock_rag_system.ai_generator.generate_response.assert_called_once()
        call_args = mock_rag_system.ai_generator.generate_response.call_args

        # Check query format
        assert (
            "Answer this question about course materials: What is machine learning?"
            in call_args[0][0]
        )

        # Check tools are provided
        assert call_args[1]["tools"] is not None
        assert call_args[1]["tool_manager"] is not None

    def test_query_with_session(self, mock_rag_system):
        """Test query processing with session ID"""
        session_id = "test-session-123"

        # Mock session manager
        mock_rag_system.session_manager.get_conversation_history.return_value = (
            "Previous context"
        )

        response, sources = mock_rag_system.query("Follow up question", session_id)

        assert response == "Mocked AI response"

        # Verify session history was retrieved
        mock_rag_system.session_manager.get_conversation_history.assert_called_once_with(
            session_id
        )

        # Verify conversation history was passed to AI generator
        call_args = mock_rag_system.ai_generator.generate_response.call_args
        assert call_args[1]["conversation_history"] == "Previous context"

        # Verify session was updated
        mock_rag_system.session_manager.add_exchange.assert_called_once_with(
            session_id, "Follow up question", "Mocked AI response"
        )

    def test_query_with_tool_execution(self, mock_rag_system):
        """Test query that triggers tool execution"""
        # Configure search tool to return results
        mock_rag_system.search_tool.last_sources = [
            {"text": "ML Course - Lesson 1", "url": "https://example.com/lesson1"}
        ]

        # Configure tool manager to return sources
        mock_rag_system.tool_manager.get_last_sources.return_value = (
            mock_rag_system.search_tool.last_sources
        )

        response, sources = mock_rag_system.query("What is machine learning?")

        assert response == "Mocked AI response"
        assert len(sources) == 1
        assert sources[0]["text"] == "ML Course - Lesson 1"
        assert sources[0]["url"] == "https://example.com/lesson1"

        # Verify sources were reset after retrieval
        mock_rag_system.tool_manager.reset_sources.assert_called_once()

    def test_query_ai_generator_exception(self, mock_rag_system):
        """Test handling when AI generator raises exception"""
        mock_rag_system.ai_generator.generate_response.side_effect = Exception(
            "API error"
        )

        with pytest.raises(Exception) as exc_info:
            mock_rag_system.query("test query")

        assert "API error" in str(exc_info.value)


class TestRAGSystemDocumentProcessing:
    """Test document processing functionality"""

    @pytest.fixture
    def mock_rag_with_docs(self, test_config):
        """RAG system with document processing mocks"""
        with (
            patch("rag_system.DocumentProcessor") as mock_doc_processor,
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("rag_system.AIGenerator"),
            patch("rag_system.SessionManager"),
        ):
            rag = RAGSystem(test_config)

            # Configure document processor mock
            sample_course = Course(
                title="Test Course",
                course_link="https://example.com/course",
                lessons=[Lesson(lesson_number=1, title="Test Lesson")],
            )
            sample_chunks = [
                CourseChunk(
                    content="Test content",
                    course_title="Test Course",
                    lesson_number=1,
                    chunk_index=0,
                )
            ]

            mock_doc_processor.return_value.process_course_document.return_value = (
                sample_course,
                sample_chunks,
            )
            rag.document_processor = mock_doc_processor.return_value
            rag.vector_store = mock_vector_store.return_value

            return rag

    def test_add_course_document_success(self, mock_rag_with_docs):
        """Test successful course document addition"""
        course, chunk_count = mock_rag_with_docs.add_course_document(
            "/path/to/course.pdf"
        )

        assert course.title == "Test Course"
        assert chunk_count == 1

        # Verify document was processed
        mock_rag_with_docs.document_processor.process_course_document.assert_called_once_with(
            "/path/to/course.pdf"
        )

        # Verify data was added to vector store
        mock_rag_with_docs.vector_store.add_course_metadata.assert_called_once()
        mock_rag_with_docs.vector_store.add_course_content.assert_called_once()

    def test_add_course_document_processing_error(self, mock_rag_with_docs):
        """Test handling of document processing errors"""
        mock_rag_with_docs.document_processor.process_course_document.side_effect = (
            Exception("Processing failed")
        )

        course, chunk_count = mock_rag_with_docs.add_course_document(
            "/path/to/invalid.pdf"
        )

        assert course is None
        assert chunk_count == 0

        # Vector store should not be called
        mock_rag_with_docs.vector_store.add_course_metadata.assert_not_called()
        mock_rag_with_docs.vector_store.add_course_content.assert_not_called()

    @patch("os.path.exists")
    @patch("os.listdir")
    @patch("os.path.isfile")
    def test_add_course_folder_success(
        self, mock_isfile, mock_listdir, mock_exists, mock_rag_with_docs
    ):
        """Test successful course folder processing"""
        # Mock file system
        mock_exists.return_value = True
        mock_listdir.return_value = ["course1.pdf", "course2.docx", "readme.txt"]
        mock_isfile.return_value = True

        # Mock existing courses (empty)
        mock_rag_with_docs.vector_store.get_existing_course_titles.return_value = []

        courses, chunks = mock_rag_with_docs.add_course_folder("/docs")

        assert courses == 3  # All three files processed
        assert chunks == 3  # One chunk per file

        # Verify all files were processed
        assert (
            mock_rag_with_docs.document_processor.process_course_document.call_count
            == 3
        )

    @patch("os.path.exists")
    def test_add_course_folder_missing_folder(self, mock_exists, mock_rag_with_docs):
        """Test handling of missing course folder"""
        mock_exists.return_value = False

        courses, chunks = mock_rag_with_docs.add_course_folder("/nonexistent")

        assert courses == 0
        assert chunks == 0

    @patch("os.path.exists")
    @patch("os.listdir")
    @patch("os.path.isfile")
    def test_add_course_folder_skip_existing(
        self, mock_isfile, mock_listdir, mock_exists, mock_rag_with_docs
    ):
        """Test skipping existing courses when adding folder"""
        mock_exists.return_value = True
        mock_listdir.return_value = ["course1.pdf"]
        mock_isfile.return_value = True

        # Mock existing courses to include the course we're trying to add
        mock_rag_with_docs.vector_store.get_existing_course_titles.return_value = [
            "Test Course"
        ]

        courses, chunks = mock_rag_with_docs.add_course_folder("/docs")

        assert courses == 0  # Should skip existing course
        assert chunks == 0

        # Document should still be processed to check if it's duplicate
        mock_rag_with_docs.document_processor.process_course_document.assert_called_once()

        # But vector store should not be updated
        mock_rag_with_docs.vector_store.add_course_metadata.assert_not_called()
        mock_rag_with_docs.vector_store.add_course_content.assert_not_called()

    @patch("os.path.exists")
    @patch("os.listdir")
    @patch("os.path.isfile")
    def test_add_course_folder_clear_existing(
        self, mock_isfile, mock_listdir, mock_exists, mock_rag_with_docs
    ):
        """Test clearing existing data before adding folder"""
        mock_exists.return_value = True
        mock_listdir.return_value = ["course1.pdf"]
        mock_isfile.return_value = True

        mock_rag_with_docs.vector_store.get_existing_course_titles.return_value = []

        courses, chunks = mock_rag_with_docs.add_course_folder(
            "/docs", clear_existing=True
        )

        # Verify data was cleared
        mock_rag_with_docs.vector_store.clear_all_data.assert_called_once()

        assert courses == 1
        assert chunks == 1


class TestRAGSystemAnalytics:
    """Test RAG system analytics functionality"""

    def test_get_course_analytics(self, mock_rag_system):
        """Test course analytics retrieval"""
        # Configure vector store mock
        mock_rag_system.vector_store.get_course_count.return_value = 5
        mock_rag_system.vector_store.get_existing_course_titles.return_value = [
            "Course 1",
            "Course 2",
            "Course 3",
            "Course 4",
            "Course 5",
        ]

        analytics = mock_rag_system.get_course_analytics()

        assert analytics["total_courses"] == 5
        assert len(analytics["course_titles"]) == 5
        assert "Course 1" in analytics["course_titles"]


class TestRAGSystemRealIntegration:
    """Test RAG system with real components (integration test)"""

    @pytest.fixture
    def real_rag_system(self, test_config):
        """RAG system with real components"""
        # Only mock the Anthropic client to avoid real API calls
        with patch("ai_generator.anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.content = [Mock(text="Real integration test response")]
            mock_response.stop_reason = "end_turn"
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            return RAGSystem(test_config)

    def test_real_integration_query_flow(
        self, real_rag_system, sample_course, sample_course_chunks
    ):
        """Test complete query flow with real components"""
        # Add test data
        real_rag_system.vector_store.add_course_metadata(sample_course)
        real_rag_system.vector_store.add_course_content(sample_course_chunks)

        # Execute query
        response, sources = real_rag_system.query("What is machine learning?")

        assert "Real integration test response" in response
        assert isinstance(sources, list)

        # Verify vector store has the data
        assert real_rag_system.vector_store.get_course_count() == 1
        assert (
            "Introduction to Machine Learning"
            in real_rag_system.vector_store.get_existing_course_titles()
        )

    def test_real_tool_registration(self, real_rag_system):
        """Test that tools are properly registered in real system"""
        tool_definitions = real_rag_system.tool_manager.get_tool_definitions()

        assert len(tool_definitions) == 2

        # Check search tool
        search_tool = next(
            (
                tool
                for tool in tool_definitions
                if tool["name"] == "search_course_content"
            ),
            None,
        )
        assert search_tool is not None
        assert "course materials" in search_tool["description"].lower()

        # Check outline tool
        outline_tool = next(
            (tool for tool in tool_definitions if tool["name"] == "get_course_outline"),
            None,
        )
        assert outline_tool is not None
        assert "outline" in outline_tool["description"].lower()

    def test_real_search_tool_execution(
        self, real_rag_system, sample_course, sample_course_chunks
    ):
        """Test that search tool actually works with real vector store"""
        # Add test data
        real_rag_system.vector_store.add_course_metadata(sample_course)
        real_rag_system.vector_store.add_course_content(sample_course_chunks)

        # Execute search tool directly
        result = real_rag_system.tool_manager.execute_tool(
            "search_course_content",
            query="machine learning",
            course_name="Introduction to Machine Learning",
        )

        assert isinstance(result, str)
        assert len(result) > 0
        assert "Introduction to Machine Learning" in result

        # Check that sources are tracked
        sources = real_rag_system.tool_manager.get_last_sources()
        assert len(sources) > 0

    def test_real_outline_tool_execution(
        self, real_rag_system, sample_course, sample_course_chunks
    ):
        """Test that outline tool actually works with real vector store"""
        # Add test data
        real_rag_system.vector_store.add_course_metadata(sample_course)
        real_rag_system.vector_store.add_course_content(sample_course_chunks)

        # Execute outline tool directly
        result = real_rag_system.tool_manager.execute_tool(
            "get_course_outline",
            course_name="Machine Learning",  # Partial name to test fuzzy matching
        )

        assert isinstance(result, str)
        assert "Introduction to Machine Learning" in result
        assert "Lessons (3 total):" in result
        assert "1. What is ML?" in result
        assert "2. Types of ML" in result
        assert "3. ML Algorithms" in result

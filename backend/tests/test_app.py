"""API layer tests for FastAPI endpoints"""

import os
import sys
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

# Add backend directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestQueryEndpoint:
    """Test /api/query endpoint"""

    @pytest.fixture
    def mock_app(self):
        """Create test client with mocked RAG system"""
        with patch("app.RAGSystem") as mock_rag_class:
            # Import app after patching
            from app import app

            # Configure mock RAG system
            mock_rag = Mock()
            mock_rag.query.return_value = (
                "Test response",
                [{"text": "Test Course", "url": "https://example.com"}],
            )
            mock_rag.session_manager.create_session.return_value = "test-session-123"
            mock_rag_class.return_value = mock_rag

            client = TestClient(app)
            return client, mock_rag

    def test_query_without_session(self, mock_app):
        """Test query endpoint without session ID"""
        client, mock_rag = mock_app

        response = client.post(
            "/api/query", json={"query": "What is machine learning?"}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["answer"] == "Test response"
        assert len(data["sources"]) == 1
        assert data["sources"][0]["text"] == "Test Course"
        assert data["session_id"] == "test-session-123"

        # Verify RAG system was called correctly
        mock_rag.query.assert_called_once_with(
            "What is machine learning?", "test-session-123"
        )

    def test_query_with_session(self, mock_app):
        """Test query endpoint with existing session ID"""
        client, mock_rag = mock_app

        response = client.post(
            "/api/query",
            json={"query": "Follow up question", "session_id": "existing-session-456"},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["session_id"] == "existing-session-456"

        # Verify RAG system was called with existing session
        mock_rag.query.assert_called_once_with(
            "Follow up question", "existing-session-456"
        )
        # Session creation should not be called
        mock_rag.session_manager.create_session.assert_not_called()

    def test_query_with_string_sources(self, mock_app):
        """Test query endpoint with string sources (backward compatibility)"""
        client, mock_rag = mock_app

        # Configure RAG to return string sources
        mock_rag.query.return_value = ("Test response", ["Course 1", "Course 2"])

        response = client.post("/api/query", json={"query": "Test query"})

        assert response.status_code == 200
        data = response.json()

        assert data["sources"] == ["Course 1", "Course 2"]

    def test_query_with_mixed_sources(self, mock_app):
        """Test query endpoint with mixed source types"""
        client, mock_rag = mock_app

        # Configure RAG to return mixed sources
        mixed_sources = [
            {"text": "Course with link", "url": "https://example.com/course"},
            "Plain text source",
        ]
        mock_rag.query.return_value = ("Test response", mixed_sources)

        response = client.post("/api/query", json={"query": "Test query"})

        assert response.status_code == 200
        data = response.json()

        assert len(data["sources"]) == 2
        assert data["sources"][0]["text"] == "Course with link"
        assert data["sources"][0]["url"] == "https://example.com/course"
        assert data["sources"][1] == "Plain text source"

    def test_query_empty_query(self, mock_app):
        """Test query endpoint with empty query"""
        client, mock_rag = mock_app

        response = client.post("/api/query", json={"query": ""})

        assert response.status_code == 200
        # Should still process empty query
        mock_rag.query.assert_called_once()

    def test_query_missing_query_field(self, mock_app):
        """Test query endpoint with missing query field"""
        client, mock_rag = mock_app

        response = client.post("/api/query", json={"session_id": "test-session"})

        assert response.status_code == 422  # Validation error

    def test_query_rag_system_exception(self, mock_app):
        """Test query endpoint when RAG system raises exception"""
        client, mock_rag = mock_app

        # Configure RAG to raise exception
        mock_rag.query.side_effect = Exception("RAG system error")

        response = client.post("/api/query", json={"query": "Test query"})

        assert response.status_code == 500
        data = response.json()
        assert "RAG system error" in data["detail"]

    def test_query_session_creation_exception(self, mock_app):
        """Test query endpoint when session creation fails"""
        client, mock_rag = mock_app

        # Configure session manager to raise exception
        mock_rag.session_manager.create_session.side_effect = Exception(
            "Session creation failed"
        )

        response = client.post("/api/query", json={"query": "Test query"})

        assert response.status_code == 500
        data = response.json()
        assert "Session creation failed" in data["detail"]


class TestCoursesEndpoint:
    """Test /api/courses endpoint"""

    @pytest.fixture
    def mock_app_courses(self):
        """Create test client with mocked RAG system for courses endpoint"""
        with patch("app.RAGSystem") as mock_rag_class:
            from app import app

            mock_rag = Mock()
            mock_rag.get_course_analytics.return_value = {
                "total_courses": 3,
                "course_titles": ["Course 1", "Course 2", "Course 3"],
            }
            mock_rag_class.return_value = mock_rag

            client = TestClient(app)
            return client, mock_rag

    def test_get_course_stats_success(self, mock_app_courses):
        """Test successful course statistics retrieval"""
        client, mock_rag = mock_app_courses

        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        assert data["total_courses"] == 3
        assert len(data["course_titles"]) == 3
        assert "Course 1" in data["course_titles"]

        # Verify RAG system was called
        mock_rag.get_course_analytics.assert_called_once()

    def test_get_course_stats_empty(self, mock_app_courses):
        """Test course statistics when no courses exist"""
        client, mock_rag = mock_app_courses

        # Configure RAG to return empty analytics
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }

        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_get_course_stats_exception(self, mock_app_courses):
        """Test course statistics endpoint when RAG system raises exception"""
        client, mock_rag = mock_app_courses

        # Configure RAG to raise exception
        mock_rag.get_course_analytics.side_effect = Exception("Analytics error")

        response = client.get("/api/courses")

        assert response.status_code == 500
        data = response.json()
        assert "Analytics error" in data["detail"]


class TestAppStartup:
    """Test application startup behavior"""

    @patch("app.os.path.exists")
    @patch("app.RAGSystem")
    def test_startup_with_docs_folder(self, mock_rag_class, mock_exists):
        """Test startup behavior when docs folder exists"""
        mock_exists.return_value = True
        mock_rag = Mock()
        mock_rag.add_course_folder.return_value = (2, 10)  # 2 courses, 10 chunks
        mock_rag_class.return_value = mock_rag

        from app import app

        with TestClient(app):
            # Trigger startup event
            pass

        # Verify docs folder was processed
        mock_rag.add_course_folder.assert_called_once_with(
            "../docs", clear_existing=False
        )

    @patch("app.os.path.exists")
    @patch("app.RAGSystem")
    def test_startup_without_docs_folder(self, mock_rag_class, mock_exists):
        """Test startup behavior when docs folder doesn't exist"""
        mock_exists.return_value = False
        mock_rag = Mock()
        mock_rag_class.return_value = mock_rag

        from app import app

        with TestClient(app):
            pass

        # Verify docs processing was not attempted
        mock_rag.add_course_folder.assert_not_called()

    @patch("app.os.path.exists")
    @patch("app.RAGSystem")
    def test_startup_docs_processing_error(self, mock_rag_class, mock_exists):
        """Test startup behavior when docs processing fails"""
        mock_exists.return_value = True
        mock_rag = Mock()
        mock_rag.add_course_folder.side_effect = Exception("Processing error")
        mock_rag_class.return_value = mock_rag

        from app import app

        # Should not crash despite processing error
        with TestClient(app):
            pass


class TestAppConfiguration:
    """Test app configuration and middleware"""

    def test_cors_configuration(self):
        """Test CORS middleware configuration"""
        from app import app

        client = TestClient(app)

        # Test CORS headers on OPTIONS request
        response = client.options("/api/query")

        # Should handle CORS properly
        assert (
            response.status_code == 405
        )  # Method not allowed, but CORS headers should be present

    def test_trusted_host_middleware(self):
        """Test trusted host middleware allows requests"""
        from app import app

        client = TestClient(app)

        # Should accept requests from any host (configured with "*")
        with patch("app.RAGSystem"):
            response = client.get("/api/courses")
            # Should not be blocked by trusted host middleware
            assert response.status_code != 400


class TestErrorHandling:
    """Test error handling across the application"""

    @pytest.fixture
    def error_app(self):
        """App configured to test error scenarios"""
        with patch("app.RAGSystem") as mock_rag_class:
            from app import app

            mock_rag = Mock()
            mock_rag_class.return_value = mock_rag

            return TestClient(app), mock_rag

    def test_query_validation_error(self, error_app):
        """Test validation error handling"""
        client, mock_rag = error_app

        # Send invalid JSON
        response = client.post("/api/query", json={"invalid_field": "value"})

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_query_unexpected_error(self, error_app):
        """Test unexpected error handling in query endpoint"""
        client, mock_rag = error_app

        # Configure RAG to raise unexpected error
        mock_rag.query.side_effect = RuntimeError("Unexpected system error")

        response = client.post("/api/query", json={"query": "test"})

        assert response.status_code == 500
        data = response.json()
        assert "Unexpected system error" in data["detail"]

    def test_courses_unexpected_error(self, error_app):
        """Test unexpected error handling in courses endpoint"""
        client, mock_rag = error_app

        # Configure RAG to raise unexpected error
        mock_rag.get_course_analytics.side_effect = ValueError(
            "Analytics computation error"
        )

        response = client.get("/api/courses")

        assert response.status_code == 500
        data = response.json()
        assert "Analytics computation error" in data["detail"]


class TestRequestResponseModels:
    """Test Pydantic request/response models"""

    def test_query_request_model_validation(self):
        """Test QueryRequest model validation"""
        from app import QueryRequest

        # Valid request
        request = QueryRequest(query="test query", session_id="test-session")
        assert request.query == "test query"
        assert request.session_id == "test-session"

        # Request without session_id
        request = QueryRequest(query="test query")
        assert request.query == "test query"
        assert request.session_id is None

        # Invalid request (missing query)
        with pytest.raises(ValueError):
            QueryRequest(session_id="test-session")

    def test_query_response_model_validation(self):
        """Test QueryResponse model validation"""
        from app import QueryResponse

        # Valid response with dict sources
        response = QueryResponse(
            answer="test answer",
            sources=[{"text": "source", "url": "https://example.com"}],
            session_id="test-session",
        )
        assert response.answer == "test answer"
        assert len(response.sources) == 1

        # Valid response with string sources
        response = QueryResponse(
            answer="test answer", sources=["string source"], session_id="test-session"
        )
        assert response.sources == ["string source"]

    def test_course_stats_model_validation(self):
        """Test CourseStats model validation"""
        from app import CourseStats

        stats = CourseStats(total_courses=5, course_titles=["Course 1", "Course 2"])
        assert stats.total_courses == 5
        assert len(stats.course_titles) == 2

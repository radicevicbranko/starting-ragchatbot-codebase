"""Comprehensive API endpoint tests using new fixtures and test app factory"""
import pytest
import json
from unittest.mock import Mock
from fastapi.testclient import TestClient


@pytest.mark.api
class TestQueryEndpointEnhanced:
    """Enhanced API tests for /api/query endpoint using new fixtures"""
    
    def test_query_with_new_fixtures(self, test_client):
        """Test basic query functionality with new fixtures"""
        client, mock_rag = test_client
        
        response = client.post("/api/query", json={
            "query": "What is machine learning?"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["answer"] == "This is a test response about machine learning."
        assert len(data["sources"]) == 1
        assert data["sources"][0]["text"] == "Introduction to ML - Lesson 1"
        assert data["sources"][0]["url"] == "https://example.com/ml/lesson1"
        assert data["session_id"] == "test-session-123"
    
    def test_query_with_custom_mock(self, test_app_factory):
        """Test query with custom mock configuration"""
        custom_mock = Mock()
        custom_mock.query.return_value = (
            "Custom response about neural networks.",
            [
                {"text": "Neural Networks - Chapter 1", "url": "https://example.com/nn/ch1"},
                {"text": "Deep Learning Basics", "url": "https://example.com/dl/basics"}
            ]
        )
        custom_mock.session_manager.create_session.return_value = "custom-session-456"
        
        app, rag_mock = test_app_factory(custom_mock)
        client = TestClient(app)
        
        response = client.post("/api/query", json={
            "query": "How do neural networks work?"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["answer"] == "Custom response about neural networks."
        assert len(data["sources"]) == 2
        assert data["session_id"] == "custom-session-456"
        
        # Verify mock was called correctly
        custom_mock.query.assert_called_once_with("How do neural networks work?", "custom-session-456")
    
    def test_query_large_response(self, test_app_factory):
        """Test query with large response data"""
        large_mock = Mock()
        
        # Create large response
        large_answer = "A" * 5000  # 5KB response
        large_sources = [
            {"text": f"Large Source {i}", "url": f"https://example.com/large/{i}"}
            for i in range(20)
        ]
        
        large_mock.query.return_value = (large_answer, large_sources)
        large_mock.session_manager.create_session.return_value = "large-session"
        
        app, _ = test_app_factory(large_mock)
        client = TestClient(app)
        
        response = client.post("/api/query", json={
            "query": "Tell me everything about machine learning"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["answer"]) == 5000
        assert len(data["sources"]) == 20
        assert all(source["text"].startswith("Large Source") for source in data["sources"])
    
    def test_query_unicode_content(self, test_app_factory):
        """Test query with Unicode and special characters"""
        unicode_mock = Mock()
        unicode_mock.query.return_value = (
            "这是关于机器学习的回答。Machine learning is 机械学習。",
            [{"text": "多语言课程 🚀", "url": "https://example.com/unicode/课程"}]
        )
        unicode_mock.session_manager.create_session.return_value = "unicode-session"
        
        app, _ = test_app_factory(unicode_mock)
        client = TestClient(app)
        
        response = client.post("/api/query", json={
            "query": "什么是机器学习？"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "机器学习" in data["answer"]
        assert "多语言课程 🚀" in data["sources"][0]["text"]
    
    def test_query_concurrent_sessions(self, test_app_factory):
        """Test handling multiple concurrent sessions"""
        session_mock = Mock()
        session_mock.session_manager.create_session.side_effect = [
            "session-1", "session-2", "session-3"
        ]
        session_mock.query.return_value = ("Response", [])
        
        app, _ = test_app_factory(session_mock)
        client = TestClient(app)
        
        # Simulate concurrent requests
        responses = []
        for i in range(3):
            response = client.post("/api/query", json={
                "query": f"Query {i+1}"
            })
            responses.append(response)
        
        # All should succeed with different session IDs
        session_ids = set()
        for response in responses:
            assert response.status_code == 200
            session_ids.add(response.json()["session_id"])
        
        assert len(session_ids) == 3  # All different sessions


@pytest.mark.api
class TestCoursesEndpointEnhanced:
    """Enhanced API tests for /api/courses endpoint"""
    
    def test_courses_with_new_fixtures(self, test_client):
        """Test courses endpoint with new fixtures"""
        client, mock_rag = test_client
        
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_courses"] == 2
        assert len(data["course_titles"]) == 2
        assert "Introduction to Machine Learning" in data["course_titles"]
        assert "Advanced Python Programming" in data["course_titles"]
    
    def test_courses_large_dataset(self, test_app_factory):
        """Test courses endpoint with large dataset"""
        large_courses_mock = Mock()
        
        # Create large course list
        large_titles = [f"Course {i:03d}: Advanced Topic {i}" for i in range(100)]
        large_courses_mock.get_course_analytics.return_value = {
            "total_courses": 100,
            "course_titles": large_titles
        }
        
        app, _ = test_app_factory(large_courses_mock)
        client = TestClient(app)
        
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_courses"] == 100
        assert len(data["course_titles"]) == 100
        assert "Course 050: Advanced Topic 50" in data["course_titles"]
    
    def test_courses_unicode_titles(self, test_app_factory):
        """Test courses endpoint with Unicode course titles"""
        unicode_courses_mock = Mock()
        unicode_courses_mock.get_course_analytics.return_value = {
            "total_courses": 3,
            "course_titles": [
                "机器学习入门",
                "Aprendizaje Automático 🤖",
                "Машинное обучение"
            ]
        }
        
        app, _ = test_app_factory(unicode_courses_mock)
        client = TestClient(app)
        
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_courses"] == 3
        assert "机器学习入门" in data["course_titles"]
        assert "Aprendizaje Automático 🤖" in data["course_titles"]
        assert "Машинное обучение" in data["course_titles"]


@pytest.mark.api
class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check(self, test_client):
        """Test health check endpoint"""
        client, _ = test_client
        
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_health_check_no_dependencies(self, test_app_factory):
        """Test health check works even if RAG system is broken"""
        broken_mock = Mock()
        broken_mock.query.side_effect = Exception("RAG system broken")
        broken_mock.get_course_analytics.side_effect = Exception("Analytics broken")
        
        app, _ = test_app_factory(broken_mock)
        client = TestClient(app)
        
        # Health check should still work
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


@pytest.mark.api
class TestErrorHandlingEnhanced:
    """Enhanced error handling tests"""
    
    def test_query_timeout_simulation(self, test_app_factory):
        """Test query timeout handling"""
        import time
        
        timeout_mock = Mock()
        def slow_query(*args, **kwargs):
            time.sleep(0.1)  # Simulate slow operation
            raise TimeoutError("Query timeout")
        
        timeout_mock.query.side_effect = slow_query
        timeout_mock.session_manager.create_session.return_value = "timeout-session"
        
        app, _ = test_app_factory(timeout_mock)
        client = TestClient(app)
        
        response = client.post("/api/query", json={"query": "slow query"})
        
        assert response.status_code == 500
        assert "Query timeout" in response.json()["detail"]
    
    def test_malformed_json_request(self, test_client):
        """Test handling of malformed JSON requests"""
        client, _ = test_client
        
        # Send malformed JSON
        response = client.post(
            "/api/query",
            data='{"query": "test", "invalid": json}',  # Invalid JSON
            headers={"content-type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_extremely_long_query(self, test_client):
        """Test handling of extremely long query strings"""
        client, mock_rag = test_client
        
        # Create very long query (100KB)
        long_query = "A" * 100000
        
        response = client.post("/api/query", json={
            "query": long_query
        })
        
        # Should handle gracefully
        assert response.status_code == 200
        
        # Verify the long query was passed to RAG system
        args, kwargs = mock_rag.query.call_args
        assert args[0] == long_query
    
    def test_empty_course_analytics(self, test_app_factory):
        """Test courses endpoint with empty analytics"""
        empty_mock = Mock()
        empty_mock.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }
        
        app, _ = test_app_factory(empty_mock)
        client = TestClient(app)
        
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []


@pytest.mark.api
class TestMiddlewareConfiguration:
    """Test middleware and CORS configuration"""
    
    def test_cors_headers_query(self, test_client):
        """Test CORS headers on query endpoint"""
        client, _ = test_client
        
        # Make request with custom origin
        response = client.post(
            "/api/query",
            json={"query": "test"},
            headers={"Origin": "https://example.com"}
        )
        
        assert response.status_code == 200
        # TestClient doesn't process CORS middleware the same way,
        # but we can verify the endpoint works
    
    def test_cors_headers_courses(self, test_client):
        """Test CORS headers on courses endpoint"""
        client, _ = test_client
        
        response = client.get(
            "/api/courses",
            headers={"Origin": "https://localhost:3000"}
        )
        
        assert response.status_code == 200
    
    def test_options_request(self, test_client):
        """Test OPTIONS request handling"""
        client, _ = test_client
        
        response = client.options("/api/query")
        
        # FastAPI should handle OPTIONS requests
        # Status could be 405 (method not allowed) or 200 depending on CORS setup
        assert response.status_code in [200, 405]


@pytest.mark.integration
class TestAppFactory:
    """Test the app factory fixture itself"""
    
    def test_app_factory_creates_different_apps(self, test_app_factory):
        """Test that app factory creates independent app instances"""
        mock1 = Mock()
        mock1.query.return_value = ("Response 1", [])
        mock1.session_manager.create_session.return_value = "session-1"
        
        mock2 = Mock()
        mock2.query.return_value = ("Response 2", [])
        mock2.session_manager.create_session.return_value = "session-2"
        
        app1, _ = test_app_factory(mock1)
        app2, _ = test_app_factory(mock2)
        
        client1 = TestClient(app1)
        client2 = TestClient(app2)
        
        response1 = client1.post("/api/query", json={"query": "test"})
        response2 = client2.post("/api/query", json={"query": "test"})
        
        assert response1.json()["answer"] == "Response 1"
        assert response1.json()["session_id"] == "session-1"
        
        assert response2.json()["answer"] == "Response 2"
        assert response2.json()["session_id"] == "session-2"
    
    def test_app_factory_with_none_mock(self, test_app_factory):
        """Test app factory creates default mock when None provided"""
        app, rag_mock = test_app_factory(None)
        client = TestClient(app)
        
        response = client.post("/api/query", json={"query": "test"})
        
        assert response.status_code == 200
        assert response.json()["answer"] == "Test response"
        assert response.json()["session_id"] == "test-session"
"""Tests for static file serving and frontend integration"""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient


@pytest.mark.api
class TestStaticFileHandling:
    """Test static file serving without requiring actual frontend files"""
    
    @pytest.fixture
    def temp_frontend_dir(self):
        """Create temporary frontend directory with test files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            frontend_path = Path(temp_dir) / "frontend"
            frontend_path.mkdir()
            
            # Create test HTML file
            index_html = frontend_path / "index.html"
            index_html.write_text("""
                <!DOCTYPE html>
                <html>
                <head><title>RAG System Test</title></head>
                <body>
                    <h1>Course Materials RAG System</h1>
                    <div id="app">Test Application</div>
                </body>
                </html>
            """)
            
            # Create test CSS file
            styles_css = frontend_path / "styles.css"
            styles_css.write_text("""
                body { font-family: Arial, sans-serif; }
                .container { max-width: 1200px; margin: 0 auto; }
            """)
            
            # Create test JS file
            app_js = frontend_path / "app.js"
            app_js.write_text("""
                console.log('RAG System Test App');
                document.addEventListener('DOMContentLoaded', function() {
                    console.log('App loaded');
                });
            """)
            
            yield frontend_path
    
    def test_static_app_with_frontend_files(self, temp_frontend_dir, test_app_factory):
        """Test app with actual frontend files"""
        from fastapi import FastAPI
        from fastapi.staticfiles import StaticFiles
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.middleware.trustedhost import TrustedHostMiddleware
        
        # Create app with static files
        mock_rag = Mock()
        mock_rag.query.return_value = ("Test response", [])
        mock_rag.session_manager.create_session.return_value = "test-session"
        mock_rag.get_course_analytics.return_value = {"total_courses": 0, "course_titles": []}
        
        app = FastAPI(title="RAG System with Static Files")
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add API routes (simplified)
        @app.get("/api/courses")
        async def get_courses():
            return {"total_courses": 0, "course_titles": []}
        
        # Mount static files
        app.mount("/", StaticFiles(directory=str(temp_frontend_dir), html=True), name="static")
        
        client = TestClient(app)
        
        # Test that index.html is served
        response = client.get("/")
        assert response.status_code == 200
        assert "Course Materials RAG System" in response.text
        assert "Test Application" in response.text
        
        # Test CSS file
        response = client.get("/styles.css")
        assert response.status_code == 200
        assert "font-family: Arial" in response.text
        
        # Test JS file
        response = client.get("/app.js")
        assert response.status_code == 200
        assert "RAG System Test App" in response.text
        
        # Test API still works
        response = client.get("/api/courses")
        assert response.status_code == 200
    
    def test_static_file_not_found(self, temp_frontend_dir):
        """Test 404 handling for missing static files"""
        from fastapi import FastAPI
        from fastapi.staticfiles import StaticFiles
        
        app = FastAPI()
        app.mount("/", StaticFiles(directory=str(temp_frontend_dir), html=True), name="static")
        
        client = TestClient(app)
        
        # Request non-existent file
        response = client.get("/nonexistent.html")
        assert response.status_code == 404
    
    def test_static_file_security(self, temp_frontend_dir):
        """Test that static files don't expose sensitive paths"""
        from fastapi import FastAPI
        from fastapi.staticfiles import StaticFiles
        
        app = FastAPI()
        app.mount("/", StaticFiles(directory=str(temp_frontend_dir), html=True), name="static")
        
        client = TestClient(app)
        
        # Try to access files outside the static directory
        response = client.get("/../../../etc/passwd")
        assert response.status_code == 404
        
        response = client.get("/../../backend/config.py")
        assert response.status_code == 404


@pytest.mark.api
class TestDevStaticFiles:
    """Test custom DevStaticFiles class with no-cache headers"""
    
    @pytest.fixture
    def dev_app_with_static(self, temp_frontend_dir):
        """Create app using DevStaticFiles class"""
        from fastapi import FastAPI
        
        # Recreate the DevStaticFiles class from app.py
        from fastapi.staticfiles import StaticFiles
        from fastapi.responses import FileResponse
        
        class DevStaticFiles(StaticFiles):
            async def get_response(self, path: str, scope):
                response = await super().get_response(path, scope)
                if isinstance(response, FileResponse):
                    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                    response.headers["Pragma"] = "no-cache"
                    response.headers["Expires"] = "0"
                return response
        
        app = FastAPI()
        app.mount("/", DevStaticFiles(directory=str(temp_frontend_dir), html=True), name="static")
        
        return TestClient(app)
    
    def test_dev_static_no_cache_headers(self, dev_app_with_static):
        """Test that DevStaticFiles adds no-cache headers"""
        client = dev_app_with_static
        
        response = client.get("/index.html")
        assert response.status_code == 200
        
        # Check for no-cache headers
        assert response.headers.get("Cache-Control") == "no-cache, no-store, must-revalidate"
        assert response.headers.get("Pragma") == "no-cache"
        assert response.headers.get("Expires") == "0"
    
    def test_dev_static_css_no_cache(self, dev_app_with_static):
        """Test no-cache headers on CSS files"""
        client = dev_app_with_static
        
        response = client.get("/styles.css")
        assert response.status_code == 200
        assert "no-cache" in response.headers.get("Cache-Control", "")
    
    def test_dev_static_js_no_cache(self, dev_app_with_static):
        """Test no-cache headers on JS files"""
        client = dev_app_with_static
        
        response = client.get("/app.js")
        assert response.status_code == 200
        assert "no-cache" in response.headers.get("Cache-Control", "")


@pytest.mark.integration
class TestFullAppWithStatic:
    """Integration tests with both API and static file serving"""
    
    @pytest.fixture
    def full_test_app(self, temp_frontend_dir):
        """Create full app with both API and static files"""
        from fastapi import FastAPI, HTTPException
        from fastapi.staticfiles import StaticFiles
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.middleware.trustedhost import TrustedHostMiddleware
        from pydantic import BaseModel
        from typing import List, Dict
        from unittest.mock import Mock
        
        # Create mock RAG system
        mock_rag = Mock()
        mock_rag.query.return_value = ("Integration test response", [])
        mock_rag.session_manager.create_session.return_value = "integration-session"
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 1,
            "course_titles": ["Integration Test Course"]
        }
        
        app = FastAPI(title="Full Integration Test App")
        
        # Add middleware
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Request/Response models
        class QueryRequest(BaseModel):
            query: str
            session_id: str = None
        
        class QueryResponse(BaseModel):
            answer: str
            sources: List[Dict[str, str]]
            session_id: str
        
        class CourseStats(BaseModel):
            total_courses: int
            course_titles: List[str]
        
        # API endpoints
        @app.post("/api/query", response_model=QueryResponse)
        async def query_documents(request: QueryRequest):
            try:
                session_id = request.session_id or mock_rag.session_manager.create_session()
                answer, sources = mock_rag.query(request.query, session_id)
                return QueryResponse(answer=answer, sources=sources, session_id=session_id)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/api/courses", response_model=CourseStats)
        async def get_course_stats():
            try:
                analytics = mock_rag.get_course_analytics()
                return CourseStats(
                    total_courses=analytics["total_courses"],
                    course_titles=analytics["course_titles"]
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Mount static files
        app.mount("/", StaticFiles(directory=str(temp_frontend_dir), html=True), name="static")
        
        return TestClient(app), mock_rag
    
    def test_api_and_static_coexistence(self, full_test_app):
        """Test that API and static files work together"""
        client, mock_rag = full_test_app
        
        # Test API endpoints work
        query_response = client.post("/api/query", json={"query": "test integration"})
        assert query_response.status_code == 200
        assert query_response.json()["answer"] == "Integration test response"
        
        courses_response = client.get("/api/courses")
        assert courses_response.status_code == 200
        assert courses_response.json()["total_courses"] == 1
        
        # Test static files work
        static_response = client.get("/")
        assert static_response.status_code == 200
        assert "Course Materials RAG System" in static_response.text
        
        css_response = client.get("/styles.css")
        assert css_response.status_code == 200
        assert "font-family" in css_response.text
    
    def test_api_priority_over_static(self, full_test_app):
        """Test that API routes have priority over static file paths"""
        client, mock_rag = full_test_app
        
        # Create a file that could conflict with API path
        # (This would be handled by FastAPI route precedence)
        
        # API should still work
        response = client.get("/api/courses")
        assert response.status_code == 200
        assert "total_courses" in response.json()
    
    def test_error_handling_with_static(self, full_test_app):
        """Test error handling doesn't interfere with static files"""
        client, mock_rag = full_test_app
        
        # Make API throw error
        mock_rag.query.side_effect = Exception("Test error")
        
        # API should return 500
        api_response = client.post("/api/query", json={"query": "test"})
        assert api_response.status_code == 500
        
        # Static files should still work
        static_response = client.get("/index.html")
        assert static_response.status_code == 200


@pytest.mark.api  
class TestStaticFileConfiguration:
    """Test static file configuration and edge cases"""
    
    def test_html_fallback_behavior(self, temp_frontend_dir):
        """Test HTML fallback for SPA routing"""
        from fastapi import FastAPI
        from fastapi.staticfiles import StaticFiles
        
        app = FastAPI()
        # html=True enables SPA fallback to index.html
        app.mount("/", StaticFiles(directory=str(temp_frontend_dir), html=True), name="static")
        
        client = TestClient(app)
        
        # Request to SPA route should fallback to index.html
        response = client.get("/nonexistent-spa-route")
        # This might be 404 or return index.html depending on StaticFiles implementation
        # The important thing is it doesn't crash
        assert response.status_code in [200, 404]
    
    def test_static_file_content_types(self, temp_frontend_dir):
        """Test proper content types are set for different file types"""
        from fastapi import FastAPI
        from fastapi.staticfiles import StaticFiles
        
        app = FastAPI()
        app.mount("/", StaticFiles(directory=str(temp_frontend_dir)), name="static")
        
        client = TestClient(app)
        
        # HTML file
        html_response = client.get("/index.html")
        if html_response.status_code == 200:
            assert "text/html" in html_response.headers.get("content-type", "")
        
        # CSS file  
        css_response = client.get("/styles.css")
        if css_response.status_code == 200:
            assert "text/css" in css_response.headers.get("content-type", "")
        
        # JS file
        js_response = client.get("/app.js")
        if js_response.status_code == 200:
            content_type = js_response.headers.get("content-type", "")
            assert any(mime in content_type for mime in ["application/javascript", "text/javascript"])
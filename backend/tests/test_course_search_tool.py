"""Unit tests for CourseSearchTool"""

from unittest.mock import Mock

from search_tools import CourseSearchTool
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test cases for CourseSearchTool functionality"""

    def test_get_tool_definition(self, course_search_tool):
        """Test that tool definition is correctly structured"""
        definition = course_search_tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["required"] == ["query"]

        # Check properties structure
        properties = definition["input_schema"]["properties"]
        assert "query" in properties
        assert "course_name" in properties
        assert "lesson_number" in properties

    def test_execute_successful_search_basic_query(self, course_search_tool):
        """Test successful search with basic query only"""
        result = course_search_tool.execute("machine learning")

        # Should call vector store search with correct parameters
        course_search_tool.store.search.assert_called_once_with(
            query="machine learning", course_name=None, lesson_number=None
        )

        # Should return formatted results
        assert "[Introduction to Machine Learning" in result
        assert "Machine learning is a subset" in result

        # Should track sources
        assert len(course_search_tool.last_sources) > 0

    def test_execute_successful_search_with_course_filter(self, course_search_tool):
        """Test successful search with course name filter"""
        result = course_search_tool.execute(
            "machine learning", course_name="Introduction to Machine Learning"
        )

        course_search_tool.store.search.assert_called_once_with(
            query="machine learning",
            course_name="Introduction to Machine Learning",
            lesson_number=None,
        )

        assert "[Introduction to Machine Learning" in result

    def test_execute_successful_search_with_lesson_filter(self, course_search_tool):
        """Test successful search with lesson number filter"""
        result = course_search_tool.execute(
            "machine learning",
            course_name="Introduction to Machine Learning",
            lesson_number=1,
        )

        course_search_tool.store.search.assert_called_once_with(
            query="machine learning",
            course_name="Introduction to Machine Learning",
            lesson_number=1,
        )

        assert "[Introduction to Machine Learning - Lesson 1]" in result

    def test_execute_vector_store_error(self, course_search_tool):
        """Test handling of vector store errors"""
        # Mock vector store to return error
        course_search_tool.store.search.return_value = SearchResults.empty(
            "Database connection failed"
        )

        result = course_search_tool.execute("machine learning")

        assert result == "Database connection failed"
        assert course_search_tool.last_sources == []

    def test_execute_no_results_found_basic_query(self, course_search_tool):
        """Test handling when no results are found"""
        # Mock vector store to return empty results
        course_search_tool.store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        result = course_search_tool.execute("nonexistent topic")

        assert result == "No relevant content found."
        assert course_search_tool.last_sources == []

    def test_execute_no_results_found_with_filters(self, course_search_tool):
        """Test handling when no results are found with filters"""
        course_search_tool.store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        result = course_search_tool.execute(
            "nonexistent topic", course_name="ML Course", lesson_number=5
        )

        assert result == "No relevant content found in course 'ML Course' in lesson 5."
        assert course_search_tool.last_sources == []

    def test_execute_partial_course_filter_message(self, course_search_tool):
        """Test error message construction with partial filters"""
        course_search_tool.store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        # Test with only course name
        result = course_search_tool.execute("test query", course_name="Some Course")
        assert result == "No relevant content found in course 'Some Course'."

        # Test with only lesson number
        result = course_search_tool.execute("test query", lesson_number=3)
        assert result == "No relevant content found in lesson 3."

    def test_format_results_with_links(self, course_search_tool):
        """Test result formatting with lesson and course links"""
        # Configure mock to return links
        course_search_tool.store.get_lesson_link.return_value = (
            "https://example.com/lesson1"
        )
        course_search_tool.store.get_course_link.return_value = (
            "https://example.com/course"
        )

        # Create search results with multiple documents
        search_results = SearchResults(
            documents=[
                "Content about machine learning algorithms",
                "More content about neural networks",
            ],
            metadata=[
                {"course_title": "ML Course", "lesson_number": 1},
                {"course_title": "ML Course", "lesson_number": 2},
            ],
            distances=[0.1, 0.2],
        )

        course_search_tool.store.search.return_value = search_results

        result = course_search_tool.execute("algorithms")

        # Should format with lesson headers
        assert "[ML Course - Lesson 1]" in result
        assert "[ML Course - Lesson 2]" in result
        assert "Content about machine learning algorithms" in result
        assert "More content about neural networks" in result

        # Should track sources with links
        expected_sources = [
            {"text": "ML Course - Lesson 1", "url": "https://example.com/lesson1"},
            {
                "text": "ML Course - Lesson 2",
                "url": "https://example.com/lesson1",
            },  # Mock returns same link
        ]
        assert len(course_search_tool.last_sources) == 2

    def test_format_results_without_links(self, course_search_tool):
        """Test result formatting when links are not available"""
        # Configure mock to return no links
        course_search_tool.store.get_lesson_link.return_value = None
        course_search_tool.store.get_course_link.return_value = None

        search_results = SearchResults(
            documents=["Content without links"],
            metadata=[{"course_title": "No Link Course", "lesson_number": 1}],
            distances=[0.1],
        )

        course_search_tool.store.search.return_value = search_results

        result = course_search_tool.execute("test")

        # Should still format properly but sources should be plain text
        assert "[No Link Course - Lesson 1]" in result
        assert course_search_tool.last_sources == ["No Link Course - Lesson 1"]

    def test_format_results_course_level_content(self, course_search_tool):
        """Test result formatting for course-level content (no lesson number)"""
        course_search_tool.store.get_course_link.return_value = (
            "https://example.com/course"
        )

        search_results = SearchResults(
            documents=["Course overview content"],
            metadata=[{"course_title": "Overview Course"}],  # No lesson_number
            distances=[0.1],
        )

        course_search_tool.store.search.return_value = search_results

        result = course_search_tool.execute("overview")

        # Should format without lesson number
        assert "[Overview Course]" in result
        assert "Course overview content" in result

        # Should track course-level source with link
        expected_source = {
            "text": "Overview Course",
            "url": "https://example.com/course",
        }
        assert course_search_tool.last_sources == [expected_source]

    def test_format_results_malformed_metadata(self, course_search_tool):
        """Test handling of malformed metadata"""
        search_results = SearchResults(
            documents=["Some content"],
            metadata=[{}],
            distances=[0.1],  # Empty metadata
        )

        course_search_tool.store.search.return_value = search_results

        result = course_search_tool.execute("test")

        # Should handle gracefully with unknown course
        assert "[unknown]" in result
        assert "Some content" in result

    def test_sources_reset_between_searches(self, course_search_tool):
        """Test that sources are properly managed between searches"""
        # First search
        course_search_tool.execute("first query")
        first_sources = course_search_tool.last_sources.copy()
        assert len(first_sources) > 0

        # Second search with different results
        course_search_tool.store.search.return_value = SearchResults(
            documents=["Different content"],
            metadata=[{"course_title": "Different Course", "lesson_number": 2}],
            distances=[0.3],
        )

        course_search_tool.execute("second query")
        second_sources = course_search_tool.last_sources

        # Sources should be different and reflect the new search
        assert second_sources != first_sources
        assert any("Different Course" in str(source) for source in second_sources)


class TestCourseSearchToolEdgeCases:
    """Test edge cases and error conditions"""

    def test_execute_with_none_query(self):
        """Test behavior with None query - should return error message"""
        mock_store = Mock()
        tool = CourseSearchTool(mock_store)

        result = tool.execute(None)
        assert result == "Error: Search query cannot be None."

    def test_execute_with_empty_string_query(self, course_search_tool):
        """Test behavior with empty string query"""
        result = course_search_tool.execute("")

        course_search_tool.store.search.assert_called_once_with(
            query="", course_name=None, lesson_number=None
        )

    def test_execute_with_invalid_lesson_number(self, course_search_tool):
        """Test behavior with invalid lesson number types"""
        # Should handle string lesson numbers
        result = course_search_tool.execute("test", lesson_number="not_a_number")

        course_search_tool.store.search.assert_called_once_with(
            query="test",
            course_name=None,
            lesson_number="not_a_number",  # Vector store should handle validation
        )

    def test_vector_store_exception_handling(self):
        """Test handling when vector store raises unexpected exceptions"""
        mock_store = Mock()
        mock_store.search.side_effect = Exception("Unexpected database error")

        tool = CourseSearchTool(mock_store)

        result = tool.execute("test query")

        # Should handle gracefully and return error message
        assert isinstance(result, str)
        assert "Search failed due to an internal error" in result
        assert "Unexpected database error" in result

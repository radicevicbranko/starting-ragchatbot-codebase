"""Unit tests for VectorStore"""

from unittest.mock import patch

import pytest
from vector_store import SearchResults, VectorStore


class TestVectorStoreSearch:
    """Test VectorStore search functionality"""

    def test_search_basic_query_success(self, populated_vector_store):
        """Test basic search without filters"""
        results = populated_vector_store.search("machine learning")

        assert not results.is_empty()
        assert results.error is None
        assert len(results.documents) > 0
        assert len(results.metadata) == len(results.documents)
        assert all("course_title" in meta for meta in results.metadata)

    def test_search_with_course_name_filter(self, populated_vector_store):
        """Test search with course name filter"""
        results = populated_vector_store.search(
            "algorithms", course_name="Introduction to Machine Learning"
        )

        assert not results.is_empty()
        assert results.error is None
        # All results should be from the specified course
        for meta in results.metadata:
            assert meta["course_title"] == "Introduction to Machine Learning"

    def test_search_with_lesson_number_filter(self, populated_vector_store):
        """Test search with lesson number filter"""
        results = populated_vector_store.search("machine learning", lesson_number=1)

        assert not results.is_empty()
        assert results.error is None
        # All results should be from lesson 1
        for meta in results.metadata:
            assert meta.get("lesson_number") == 1

    def test_search_with_both_filters(self, populated_vector_store):
        """Test search with both course name and lesson number filters"""
        results = populated_vector_store.search(
            "machine learning",
            course_name="Introduction to Machine Learning",
            lesson_number=1,
        )

        assert not results.is_empty()
        assert results.error is None
        # Results should match both filters
        for meta in results.metadata:
            assert meta["course_title"] == "Introduction to Machine Learning"
            assert meta.get("lesson_number") == 1

    def test_search_nonexistent_course(self, populated_vector_store):
        """Test search with non-existent course name"""
        results = populated_vector_store.search(
            "machine learning", course_name="Nonexistent Course"
        )

        assert results.error is not None
        assert "No course found matching 'Nonexistent Course'" in results.error
        assert results.is_empty()

    def test_search_no_results_found(self, populated_vector_store):
        """Test search that yields no results - semantic search may still return distant matches"""
        results = populated_vector_store.search("completely unrelated topic xyz123")

        # Should return results without error (semantic search finds best matches even if distant)
        assert results.error is None
        # Note: Semantic search typically returns results even for unrelated queries,
        # finding the "closest" match. This is expected behavior.

    def test_search_with_limit(self, populated_vector_store):
        """Test search with custom result limit"""
        results = populated_vector_store.search("machine learning", limit=1)

        assert not results.is_empty()
        assert len(results.documents) == 1
        assert len(results.metadata) == 1
        assert len(results.distances) == 1


class TestVectorStoreCourseNameResolution:
    """Test course name resolution functionality"""

    def test_resolve_exact_course_name(self, populated_vector_store):
        """Test resolving exact course name"""
        resolved = populated_vector_store._resolve_course_name(
            "Introduction to Machine Learning"
        )
        assert resolved == "Introduction to Machine Learning"

    def test_resolve_partial_course_name(self, populated_vector_store):
        """Test resolving partial course name"""
        resolved = populated_vector_store._resolve_course_name("Machine Learning")
        assert resolved == "Introduction to Machine Learning"

    def test_resolve_case_insensitive_course_name(self, populated_vector_store):
        """Test case-insensitive course name resolution"""
        resolved = populated_vector_store._resolve_course_name("machine learning")
        assert resolved == "Introduction to Machine Learning"

    def test_resolve_nonexistent_course_name(self, populated_vector_store):
        """Test resolving non-existent course name"""
        resolved = populated_vector_store._resolve_course_name("Nonexistent Course")
        assert resolved is None


class TestVectorStoreFilterBuilding:
    """Test filter building for ChromaDB queries"""

    def test_build_filter_no_parameters(self, real_vector_store):
        """Test filter building with no parameters"""
        filter_dict = real_vector_store._build_filter(None, None)
        assert filter_dict is None

    def test_build_filter_course_only(self, real_vector_store):
        """Test filter building with course title only"""
        filter_dict = real_vector_store._build_filter("Test Course", None)
        expected = {"course_title": "Test Course"}
        assert filter_dict == expected

    def test_build_filter_lesson_only(self, real_vector_store):
        """Test filter building with lesson number only"""
        filter_dict = real_vector_store._build_filter(None, 5)
        expected = {"lesson_number": 5}
        assert filter_dict == expected

    def test_build_filter_both_parameters(self, real_vector_store):
        """Test filter building with both parameters"""
        filter_dict = real_vector_store._build_filter("Test Course", 3)
        expected = {"$and": [{"course_title": "Test Course"}, {"lesson_number": 3}]}
        assert filter_dict == expected


class TestVectorStoreDataManagement:
    """Test data addition and management"""

    def test_add_course_metadata(self, real_vector_store, sample_course):
        """Test adding course metadata"""
        real_vector_store.add_course_metadata(sample_course)

        # Verify course was added
        existing_titles = real_vector_store.get_existing_course_titles()
        assert sample_course.title in existing_titles

    def test_add_course_content(self, real_vector_store, sample_course_chunks):
        """Test adding course content chunks"""
        real_vector_store.add_course_content(sample_course_chunks)

        # Search should find the added content
        results = real_vector_store.search("machine learning")
        assert not results.is_empty()

    def test_get_course_count(self, populated_vector_store):
        """Test getting course count"""
        count = populated_vector_store.get_course_count()
        assert count == 1  # One course from fixture

    def test_get_existing_course_titles(self, populated_vector_store):
        """Test getting existing course titles"""
        titles = populated_vector_store.get_existing_course_titles()
        assert "Introduction to Machine Learning" in titles

    def test_clear_all_data(self, populated_vector_store):
        """Test clearing all data"""
        # Verify data exists
        assert populated_vector_store.get_course_count() > 0

        # Clear data
        populated_vector_store.clear_all_data()

        # Verify data is cleared
        assert populated_vector_store.get_course_count() == 0
        assert len(populated_vector_store.get_existing_course_titles()) == 0


class TestVectorStoreLinkMethods:
    """Test course and lesson link retrieval"""

    def test_get_course_link_existing(self, populated_vector_store):
        """Test getting link for existing course"""
        link = populated_vector_store.get_course_link(
            "Introduction to Machine Learning"
        )
        assert link == "https://example.com/ml-course"

    def test_get_course_link_nonexistent(self, populated_vector_store):
        """Test getting link for non-existent course"""
        link = populated_vector_store.get_course_link("Nonexistent Course")
        assert link is None

    def test_get_lesson_link_existing(self, populated_vector_store):
        """Test getting link for existing lesson"""
        link = populated_vector_store.get_lesson_link(
            "Introduction to Machine Learning", 1
        )
        assert link == "https://example.com/lesson1"

    def test_get_lesson_link_nonexistent_course(self, populated_vector_store):
        """Test getting lesson link for non-existent course"""
        link = populated_vector_store.get_lesson_link("Nonexistent Course", 1)
        assert link is None

    def test_get_lesson_link_nonexistent_lesson(self, populated_vector_store):
        """Test getting link for non-existent lesson"""
        link = populated_vector_store.get_lesson_link(
            "Introduction to Machine Learning", 999
        )
        assert link is None


class TestVectorStoreCourseOutline:
    """Test course outline functionality"""

    def test_get_course_outline_existing(self, populated_vector_store):
        """Test getting outline for existing course"""
        outline = populated_vector_store.get_course_outline(
            "Introduction to Machine Learning"
        )

        assert outline is not None
        assert outline["course_title"] == "Introduction to Machine Learning"
        assert outline["course_link"] == "https://example.com/ml-course"
        assert len(outline["lessons"]) == 3

        # Verify lesson structure
        lesson1 = outline["lessons"][0]
        assert lesson1["lesson_number"] == 1
        assert lesson1["lesson_title"] == "What is ML?"

    def test_get_course_outline_with_fuzzy_matching(self, populated_vector_store):
        """Test getting outline with partial course name"""
        outline = populated_vector_store.get_course_outline("Machine Learning")

        assert outline is not None
        assert outline["course_title"] == "Introduction to Machine Learning"

    def test_get_course_outline_nonexistent(self, populated_vector_store):
        """Test getting outline for non-existent course"""
        outline = populated_vector_store.get_course_outline("Nonexistent Course")
        assert outline is None


class TestVectorStoreErrorHandling:
    """Test error handling in VectorStore"""

    @patch("chromadb.PersistentClient")
    def test_chromadb_connection_error(self, mock_client_class, test_config):
        """Test handling of ChromaDB connection errors"""
        # Mock client to raise exception on creation
        mock_client_class.side_effect = Exception("Cannot connect to ChromaDB")

        with pytest.raises(Exception):
            VectorStore(test_config.CHROMA_PATH, test_config.EMBEDDING_MODEL)

    def test_search_with_chromadb_exception(self, real_vector_store):
        """Test search behavior when ChromaDB raises exception"""
        # Mock the collection to raise exception
        with patch.object(real_vector_store.course_content, "query") as mock_query:
            mock_query.side_effect = Exception("ChromaDB query failed")

            results = real_vector_store.search("test query")

            assert results.error is not None
            assert "Search error" in results.error
            assert results.is_empty()

    def test_course_name_resolution_exception(self, real_vector_store):
        """Test course name resolution when ChromaDB raises exception"""
        with patch.object(real_vector_store.course_catalog, "query") as mock_query:
            mock_query.side_effect = Exception("ChromaDB query failed")

            resolved = real_vector_store._resolve_course_name("Test Course")
            assert resolved is None

    def test_add_empty_course_content(self, real_vector_store):
        """Test adding empty course content list"""
        real_vector_store.add_course_content([])
        # Should not raise exception - verify by checking no crash occurs


class TestSearchResults:
    """Test SearchResults data class"""

    def test_from_chroma_normal_results(self):
        """Test creating SearchResults from normal ChromaDB results"""
        chroma_results = {
            "documents": [["doc1", "doc2"]],
            "metadatas": [["meta1", "meta2"]],
            "distances": [[0.1, 0.2]],
        }

        results = SearchResults.from_chroma(chroma_results)

        assert results.documents == ["doc1", "doc2"]
        assert results.metadata == ["meta1", "meta2"]
        assert results.distances == [0.1, 0.2]
        assert results.error is None

    def test_from_chroma_empty_results(self):
        """Test creating SearchResults from empty ChromaDB results"""
        chroma_results = {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        results = SearchResults.from_chroma(chroma_results)

        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error is None

    def test_empty_with_error(self):
        """Test creating empty SearchResults with error"""
        results = SearchResults.empty("Test error message")

        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error == "Test error message"
        assert results.is_empty()

    def test_is_empty_detection(self):
        """Test is_empty method"""
        empty_results = SearchResults([], [], [])
        assert empty_results.is_empty()

        non_empty_results = SearchResults(["doc1"], [{"meta": "data"}], [0.1])
        assert not non_empty_results.is_empty()

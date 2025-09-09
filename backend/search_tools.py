from abc import ABC, abstractmethod
from typing import Any

from vector_store import SearchResults, VectorStore


class Tool(ABC):
    """Abstract base class for all tools"""

    @abstractmethod
    def get_tool_definition(self) -> dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters"""
        pass


class CourseSearchTool(Tool):
    """Tool for searching course content with semantic course name matching"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last search

    def get_tool_definition(self) -> dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "search_course_content",
            "description": "Search course materials with smart course name matching and lesson filtering",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in the course content",
                    },
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')",
                    },
                    "lesson_number": {
                        "type": "integer",
                        "description": "Specific lesson number to search within (e.g. 1, 2, 3)",
                    },
                },
                "required": ["query"],
            },
        }

    def execute(
        self,
        query: str,
        course_name: str | None = None,
        lesson_number: int | None = None,
    ) -> str:
        """
        Execute the search tool with given parameters.

        Args:
            query: What to search for
            course_name: Optional course filter
            lesson_number: Optional lesson filter

        Returns:
            Formatted search results or error message
        """

        # Input validation
        if query is None:
            return "Error: Search query cannot be None."

        try:
            # Use the vector store's unified search interface
            results = self.store.search(
                query=query, course_name=course_name, lesson_number=lesson_number
            )

            # Handle errors from vector store
            if results.error:
                return results.error

            # Handle empty results
            if results.is_empty():
                filter_info = ""
                if course_name:
                    filter_info += f" in course '{course_name}'"
                if lesson_number:
                    filter_info += f" in lesson {lesson_number}"
                return f"No relevant content found{filter_info}."

            # Format and return results
            return self._format_results(results)

        except Exception as e:
            # Handle any unexpected errors gracefully
            error_msg = f"Search failed due to an internal error: {str(e)}"
            print(f"CourseSearchTool error: {e}")  # Log for debugging
            return error_msg

    def _format_results(self, results: SearchResults) -> str:
        """Format search results with course and lesson context"""
        formatted = []
        sources = []  # Track sources for the UI

        for doc, meta in zip(results.documents, results.metadata, strict=False):
            course_title = meta.get("course_title", "unknown")
            lesson_num = meta.get("lesson_number")

            # Build context header
            header = f"[{course_title}"
            if lesson_num is not None:
                header += f" - Lesson {lesson_num}"
            header += "]"

            # Track source for the UI with link if available
            source_text = course_title
            if lesson_num is not None:
                source_text += f" - Lesson {lesson_num}"
                # Try to get lesson link
                lesson_link = self.store.get_lesson_link(course_title, lesson_num)
                if lesson_link:
                    # Create source object with link
                    sources.append({"text": source_text, "url": lesson_link})
                else:
                    # Fallback to plain text
                    sources.append(source_text)
            else:
                # For course-level content, try to get course link
                course_link = self.store.get_course_link(course_title)
                if course_link:
                    sources.append({"text": source_text, "url": course_link})
                else:
                    sources.append(source_text)

            formatted.append(f"{header}\n{doc}")

        # Store sources for retrieval
        self.last_sources = sources

        return "\n\n".join(formatted)


class CourseOutlineTool(Tool):
    """Tool for getting complete course outlines with lesson structure"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last search

    def get_tool_definition(self) -> dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "get_course_outline",
            "description": "Get the complete outline/structure of a course including all lessons with numbers and titles",
            "input_schema": {
                "type": "object",
                "properties": {
                    "course_name": {
                        "type": "string",
                        "description": "Course title or partial course name (e.g. 'MCP', 'Introduction', 'RAG')",
                    }
                },
                "required": ["course_name"],
            },
        }

    def execute(self, course_name: str) -> str:
        """
        Execute the course outline tool with given course name.

        Args:
            course_name: Course title to get outline for

        Returns:
            Formatted course outline with lessons or error message
        """

        # Get course outline from vector store
        outline = self.store.get_course_outline(course_name)

        # Handle course not found
        if not outline:
            return f"No course found matching '{course_name}'. Please check the course name or try a partial match."

        # Format the outline response
        return self._format_outline(outline)

    def _format_outline(self, outline: dict[str, Any]) -> str:
        """Format course outline for AI response"""
        course_title = outline.get("course_title", "Unknown Course")
        course_link = outline.get("course_link")
        lessons = outline.get("lessons", [])

        # Build formatted response
        formatted = [f"Course: {course_title}"]

        if lessons:
            formatted.append(f"\nLessons ({len(lessons)} total):")
            for lesson in lessons:
                lesson_num = lesson.get("lesson_number", "?")
                lesson_title = lesson.get("lesson_title", "Untitled Lesson")
                formatted.append(f"  {lesson_num}. {lesson_title}")
        else:
            formatted.append("\nNo lesson structure available for this course.")

        # Track sources for the UI
        sources = []
        if course_link:
            sources.append({"text": course_title, "url": course_link})
        else:
            sources.append(course_title)

        self.last_sources = sources

        return "\n".join(formatted)


class ToolManager:
    """Manages available tools for the AI"""

    def __init__(self):
        self.tools = {}

    def register_tool(self, tool: Tool):
        """Register any tool that implements the Tool interface"""
        tool_def = tool.get_tool_definition()
        tool_name = tool_def.get("name")
        if not tool_name:
            raise ValueError("Tool must have a 'name' in its definition")
        self.tools[tool_name] = tool

    def get_tool_definitions(self) -> list:
        """Get all tool definitions for Anthropic tool calling"""
        return [tool.get_tool_definition() for tool in self.tools.values()]

    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given parameters"""
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found"

        return self.tools[tool_name].execute(**kwargs)

    def get_last_sources(self) -> list:
        """Get sources from the last search operation"""
        # Check all tools for last_sources attribute
        for tool in self.tools.values():
            if hasattr(tool, "last_sources") and tool.last_sources:
                return tool.last_sources
        return []

    def reset_sources(self):
        """Reset sources from all tools that track sources"""
        for tool in self.tools.values():
            if hasattr(tool, "last_sources"):
                tool.last_sources = []

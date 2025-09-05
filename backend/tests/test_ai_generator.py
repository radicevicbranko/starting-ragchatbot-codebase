"""Unit tests for AIGenerator"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from ai_generator import AIGenerator
from search_tools import ToolManager


class TestAIGeneratorBasic:
    """Test basic AIGenerator functionality"""
    
    def test_init(self):
        """Test AIGenerator initialization"""
        generator = AIGenerator("test-api-key", "claude-sonnet-4")
        
        assert generator.model == "claude-sonnet-4"
        assert generator.base_params["model"] == "claude-sonnet-4"
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800
    
    @patch('anthropic.Anthropic')
    def test_generate_response_simple_query(self, mock_anthropic):
        """Test simple response generation without tools"""
        # Mock client and response
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        mock_response = Mock()
        mock_response.content = [Mock(text="This is a simple response")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        
        generator = AIGenerator("test-key", "claude-sonnet-4")
        result = generator.generate_response("What is machine learning?")
        
        assert result == "This is a simple response"
        mock_client.messages.create.assert_called_once()
    
    @patch('anthropic.Anthropic')
    def test_generate_response_with_conversation_history(self, mock_anthropic):
        """Test response generation with conversation history"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        mock_response = Mock()
        mock_response.content = [Mock(text="Response with context")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        
        generator = AIGenerator("test-key", "claude-sonnet-4")
        result = generator.generate_response(
            "Follow up question",
            conversation_history="Previous conversation context"
        )
        
        assert result == "Response with context"
        
        # Verify system prompt includes history
        call_args = mock_client.messages.create.call_args
        system_content = call_args[1]["system"]
        assert "Previous conversation context" in system_content


class TestAIGeneratorToolIntegration:
    """Test AIGenerator tool calling functionality"""
    
    @patch('anthropic.Anthropic')
    def test_generate_response_with_tools_no_tool_use(self, mock_anthropic):
        """Test response generation with tools available but not used"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        mock_response = Mock()
        mock_response.content = [Mock(text="Direct response without tools")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        
        # Create mock tools and tool manager
        mock_tools = [{"name": "search_tool", "description": "Search content"}]
        mock_tool_manager = Mock()
        
        generator = AIGenerator("test-key", "claude-sonnet-4")
        result = generator.generate_response(
            "General question",
            tools=mock_tools,
            tool_manager=mock_tool_manager
        )
        
        assert result == "Direct response without tools"
        
        # Verify tools were included in API call
        call_args = mock_client.messages.create.call_args
        assert call_args[1]["tools"] == mock_tools
        assert call_args[1]["tool_choice"] == {"type": "auto"}
    
    @patch('anthropic.Anthropic')
    def test_generate_response_with_tool_use_single_tool(self, mock_anthropic):
        """Test response generation with single tool call"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock initial response with tool use
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_123"
        tool_block.input = {"query": "machine learning", "course_name": "ML"}
        mock_tool_response.content = [tool_block]
        
        # Mock final response after tool execution
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Based on search results: ML is...")]
        
        # Configure client to return different responses for each call
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results: Machine learning is AI subset"
        
        mock_tools = [{"name": "search_course_content", "description": "Search content"}]
        
        generator = AIGenerator("test-key", "claude-sonnet-4")
        result = generator.generate_response(
            "What is machine learning?",
            tools=mock_tools,
            tool_manager=mock_tool_manager
        )
        
        assert result == "Based on search results: ML is..."
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="machine learning",
            course_name="ML"
        )
        
        # Verify two API calls were made
        assert mock_client.messages.create.call_count == 2
    
    @patch('anthropic.Anthropic')
    def test_generate_response_with_multiple_tool_calls(self, mock_anthropic):
        """Test response generation with multiple tool calls"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock initial response with multiple tool uses
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        
        tool_block1 = Mock()
        tool_block1.type = "tool_use"
        tool_block1.name = "search_course_content"
        tool_block1.id = "tool_1"
        tool_block1.input = {"query": "machine learning"}
        
        tool_block2 = Mock()
        tool_block2.type = "tool_use"
        tool_block2.name = "get_course_outline"
        tool_block2.id = "tool_2"
        tool_block2.input = {"course_name": "ML Course"}
        
        mock_tool_response.content = [tool_block1, tool_block2]
        
        # Mock final response
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Combined results from both tools")]
        
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        
        # Mock tool manager with multiple tool results
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Search result 1",
            "Course outline result"
        ]
        
        mock_tools = [
            {"name": "search_course_content", "description": "Search content"},
            {"name": "get_course_outline", "description": "Get outline"}
        ]
        
        generator = AIGenerator("test-key", "claude-sonnet-4")
        result = generator.generate_response(
            "Tell me about ML course",
            tools=mock_tools,
            tool_manager=mock_tool_manager
        )
        
        assert result == "Combined results from both tools"
        
        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="machine learning")
        mock_tool_manager.execute_tool.assert_any_call("get_course_outline", course_name="ML Course")


class TestAIGeneratorToolExecutionHandling:
    """Test _handle_tool_execution method specifically"""
    
    def test_handle_tool_execution_message_construction(self):
        """Test correct message construction during tool execution"""
        # Create a real AIGenerator (without mocking Anthropic for this test)
        with patch('anthropic.Anthropic') as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client
            
            # Mock final response
            mock_final_response = Mock()
            mock_final_response.content = [Mock(text="Final response after tool use")]
            mock_client.messages.create.return_value = mock_final_response
            
            generator = AIGenerator("test-key", "claude-sonnet-4")
            
            # Create mock initial response
            initial_response = Mock()
            tool_block = Mock()
            tool_block.type = "tool_use"
            tool_block.name = "search_tool"
            tool_block.id = "tool_123"
            tool_block.input = {"query": "test"}
            initial_response.content = [tool_block]
            
            # Create base params
            base_params = {
                "messages": [{"role": "user", "content": "test query"}],
                "system": "test system prompt"
            }
            
            # Mock tool manager
            mock_tool_manager = Mock()
            mock_tool_manager.execute_tool.return_value = "Tool execution result"
            
            # Call the method
            result = generator._handle_tool_execution(initial_response, base_params, mock_tool_manager)
            
            assert result == "Final response after tool use"
            
            # Verify tool was executed
            mock_tool_manager.execute_tool.assert_called_once_with("search_tool", query="test")
            
            # Verify final API call structure
            call_args = mock_client.messages.create.call_args
            messages = call_args[1]["messages"]
            
            # Should have original user message, AI tool use message, and tool result message
            assert len(messages) == 3
            assert messages[0]["role"] == "user"
            assert messages[1]["role"] == "assistant" 
            assert messages[2]["role"] == "user"
            
            # Tool result should be in proper format
            tool_result = messages[2]["content"][0]
            assert tool_result["type"] == "tool_result"
            assert tool_result["tool_use_id"] == "tool_123"
            assert tool_result["content"] == "Tool execution result"
    
    def test_handle_tool_execution_no_tool_blocks(self):
        """Test handling when response contains no tool use blocks"""
        with patch('anthropic.Anthropic') as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client
            
            mock_final_response = Mock()
            mock_final_response.content = [Mock(text="No tools used")]
            mock_client.messages.create.return_value = mock_final_response
            
            generator = AIGenerator("test-key", "claude-sonnet-4")
            
            # Create mock initial response with no tool blocks
            initial_response = Mock()
            text_block = Mock()
            text_block.type = "text"
            initial_response.content = [text_block]
            
            base_params = {
                "messages": [{"role": "user", "content": "test query"}],
                "system": "test system prompt"
            }
            
            mock_tool_manager = Mock()
            
            result = generator._handle_tool_execution(initial_response, base_params, mock_tool_manager)
            
            assert result == "No tools used"
            # Tool manager should not be called
            mock_tool_manager.execute_tool.assert_not_called()


class TestAIGeneratorErrorHandling:
    """Test error handling in AIGenerator"""
    
    @patch('anthropic.Anthropic')
    def test_anthropic_api_error(self, mock_anthropic):
        """Test handling of Anthropic API errors"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock API to raise exception
        mock_client.messages.create.side_effect = Exception("API rate limit exceeded")
        
        generator = AIGenerator("test-key", "claude-sonnet-4")
        
        with pytest.raises(Exception) as exc_info:
            generator.generate_response("test query")
        
        assert "API rate limit exceeded" in str(exc_info.value)
    
    @patch('anthropic.Anthropic')
    def test_tool_execution_error(self, mock_anthropic):
        """Test handling of tool execution errors"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock initial tool use response
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "failing_tool"
        tool_block.id = "tool_123"
        tool_block.input = {"param": "value"}
        mock_tool_response.content = [tool_block]
        
        # Mock final response
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Handled tool error")]
        
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        
        # Mock tool manager to raise exception
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")
        
        mock_tools = [{"name": "failing_tool", "description": "A failing tool"}]
        
        generator = AIGenerator("test-key", "claude-sonnet-4")
        
        # Should propagate the tool execution error
        with pytest.raises(Exception) as exc_info:
            generator.generate_response(
                "test query",
                tools=mock_tools,
                tool_manager=mock_tool_manager
            )
        
        assert "Tool execution failed" in str(exc_info.value)
    
    @patch('anthropic.Anthropic')
    def test_malformed_tool_response(self, mock_anthropic):
        """Test handling of malformed tool response"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock response with malformed content
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_response.content = []  # Empty content
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Handled malformed response")]
        
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        
        mock_tool_manager = Mock()
        mock_tools = [{"name": "test_tool", "description": "Test tool"}]
        
        generator = AIGenerator("test-key", "claude-sonnet-4")
        result = generator.generate_response(
            "test query",
            tools=mock_tools,
            tool_manager=mock_tool_manager
        )
        
        # Should handle gracefully and not call any tools
        assert result == "Handled malformed response"
        mock_tool_manager.execute_tool.assert_not_called()


class TestAIGeneratorSystemPrompt:
    """Test system prompt construction and usage"""
    
    def test_system_prompt_content(self):
        """Test that system prompt contains expected content"""
        generator = AIGenerator("test-key", "claude-sonnet-4")
        
        assert "specialized in course materials" in generator.SYSTEM_PROMPT
        assert "search_course_content tool" in generator.SYSTEM_PROMPT
        assert "get_course_outline tool" in generator.SYSTEM_PROMPT
        assert "ONE tool per query maximum" in generator.SYSTEM_PROMPT
    
    @patch('anthropic.Anthropic')
    def test_system_prompt_with_history(self, mock_anthropic):
        """Test system prompt construction with conversation history"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        mock_response = Mock()
        mock_response.content = [Mock(text="Response with history")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        
        generator = AIGenerator("test-key", "claude-sonnet-4")
        generator.generate_response(
            "Current question",
            conversation_history="User: Previous question\nAssistant: Previous answer"
        )
        
        # Verify system prompt includes history
        call_args = mock_client.messages.create.call_args
        system_content = call_args[1]["system"]
        
        assert generator.SYSTEM_PROMPT in system_content
        assert "Previous conversation:" in system_content
        assert "User: Previous question" in system_content
        assert "Assistant: Previous answer" in system_content
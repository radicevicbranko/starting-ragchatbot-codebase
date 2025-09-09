from typing import Any

import anthropic


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """You are an AI assistant specialized in course materials and educational content with access to specialized tools for course information.

Tool Usage Guidelines:
- **Multiple tool calls allowed**: You can use tools sequentially to gather comprehensive information
- **Tool call strategy**: Start with broader searches, then narrow down with specific filters if needed
- **Course outline queries**: Use get_course_outline tool for course structure questions
- **Content queries**: Use search_course_content tool for topic-specific questions
- **Sequential refinement**: If initial results are insufficient, you may make additional tool calls with different parameters
- **Maximum efficiency**: Use tools thoughtfully - each call should add value to your response

Response Protocol:
- **Comprehensive answers**: Use multiple tool calls when necessary to provide complete responses
- **Source synthesis**: Combine information from multiple searches coherently
- **Clear attribution**: Reference sources appropriately
- **Direct answers**: Provide the information requested without meta-commentary about your process

All responses must be:
1. **Brief and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Well-sourced** - Include relevant examples and references
5. **Complete** - Address all aspects of the question

Provide comprehensive, well-researched answers using available tools as needed.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: str | None = None,
        tools: list | None = None,
        tool_manager=None,
        max_rounds: int = 2,
    ) -> str:
        """
        Generate AI response with sequential tool calling support up to max_rounds.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of sequential tool calling rounds (default: 2)

        Returns:
            Generated response as string
        """

        # Build system content efficiently
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Initialize conversation context for sequential rounds
        messages = [{"role": "user", "content": query}]

        # Execute sequential rounds
        try:
            final_response = self._execute_sequential_rounds(
                messages=messages,
                system_content=system_content,
                tools=tools,
                tool_manager=tool_manager,
                max_rounds=max_rounds,
            )
            return final_response

        except Exception as e:
            # Graceful fallback on any error
            print(f"Error in sequential tool calling: {e}")
            return self._fallback_response(query, system_content)

    def _execute_sequential_rounds(
        self,
        messages: list[dict],
        system_content: str,
        tools: list | None,
        tool_manager,
        max_rounds: int,
    ) -> str:
        """
        Execute up to max_rounds of sequential tool calling.

        Args:
            messages: Conversation messages
            system_content: System prompt with context
            tools: Available tools
            tool_manager: Tool execution manager
            max_rounds: Maximum rounds allowed

        Returns:
            Final response text

        Raises:
            Exception: On unrecoverable errors
        """

        current_round = 0

        while current_round < max_rounds:
            current_round += 1

            # Prepare API parameters for this round
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content,
            }

            # Add tools if available and tool manager exists
            if tools and tool_manager:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}

            # Make API call
            try:
                response = self.client.messages.create(**api_params)
            except Exception as e:
                raise Exception(f"API call failed in round {current_round}: {str(e)}")

            # Add Claude's response to conversation
            messages.append({"role": "assistant", "content": response.content})

            # Check termination conditions
            termination_result = self._check_termination_conditions(
                response, current_round, max_rounds
            )

            if termination_result["should_terminate"]:
                return termination_result["response"]

            # Execute tools and continue to next round
            try:
                tool_results = self._execute_tools_for_round(response, tool_manager)
                if tool_results:
                    messages.append({"role": "user", "content": tool_results})
                else:
                    # No tools executed - this shouldn't happen if stop_reason is tool_use
                    return self._extract_text_response(response)

            except Exception as e:
                # Tool execution failed - terminate gracefully
                print(f"Tool execution failed in round {current_round}: {e}")
                return f"I encountered an error while using tools to answer your question. {self._extract_text_response(response)}"

        # If we reach here, we've exhausted max_rounds
        # Make final call without tools to get conclusion
        return self._make_final_call_without_tools(messages, system_content)

    def _check_termination_conditions(
        self, response, current_round: int, max_rounds: int
    ) -> dict[str, Any]:
        """
        Check if we should terminate the sequential tool calling.

        Termination occurs when:
        1. Claude's response has no tool_use blocks
        2. Maximum rounds completed

        Args:
            response: Claude's response
            current_round: Current round number
            max_rounds: Maximum allowed rounds

        Returns:
            Dict with 'should_terminate' boolean and 'response' text if terminating
        """

        # Condition 1: No tool use - Claude provided final answer
        if response.stop_reason != "tool_use":
            return {
                "should_terminate": True,
                "response": self._extract_text_response(response),
            }

        # Condition 2: Max rounds completed - only terminate if we exceed max rounds
        # Note: We should allow max_rounds to complete, so only terminate if current_round > max_rounds
        if current_round > max_rounds:
            return {
                "should_terminate": True,
                "response": self._extract_text_response(response),
            }

        # Continue to next round
        return {"should_terminate": False, "response": None}

    def _execute_tools_for_round(self, response, tool_manager) -> list[dict] | None:
        """
        Execute all tool calls in the current response.

        Args:
            response: Claude's response containing tool_use blocks
            tool_manager: Tool execution manager

        Returns:
            List of tool results or None if no tools executed

        Raises:
            Exception: On tool execution failures
        """

        tool_results = []

        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    # Execute the tool
                    tool_result = tool_manager.execute_tool(
                        content_block.name, **content_block.input
                    )

                    # Handle tool execution errors
                    if isinstance(tool_result, str) and tool_result.startswith(
                        "Error:"
                    ):
                        # Tool returned an error - we can continue but should handle gracefully
                        print(f"Tool execution error: {tool_result}")

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result,
                        }
                    )

                except Exception as e:
                    # Critical tool execution failure
                    error_msg = (
                        f"Failed to execute tool '{content_block.name}': {str(e)}"
                    )
                    print(f"Critical tool error: {error_msg}")

                    # Re-raise the exception to terminate the sequential tool calling
                    raise e

        return tool_results if tool_results else None

    def _handle_tool_execution(
        self, initial_response, base_params: dict[str, Any], tool_manager
    ):
        """
        Handle execution of tool calls and get follow-up response.

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()

        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})

        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, **content_block.input
                )

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result,
                    }
                )

        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"],
        }

        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text

    def _extract_text_response(self, response) -> str:
        """Extract text content from Claude's response, handling mixed content."""
        text_parts = []
        for content_block in response.content:
            if hasattr(content_block, "type") and str(content_block.type) == "text":
                # Real API response or Mock with type="text"
                text_parts.append(str(content_block.text))
            elif hasattr(content_block, "text"):
                # Mock object for testing - convert to string
                text_parts.append(str(content_block.text))

        return (
            "".join(text_parts)
            if text_parts
            else "I don't have a clear answer to provide."
        )

    def _make_final_call_without_tools(
        self, messages: list[dict], system_content: str
    ) -> str:
        """Make final API call without tools to get conclusion."""

        # Add instruction for Claude to provide final answer
        messages.append(
            {
                "role": "user",
                "content": "Please provide your final answer based on the information gathered.",
            }
        )

        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
            # Explicitly no tools parameter
        }

        try:
            final_response = self.client.messages.create(**api_params)
            return self._extract_text_response(final_response)
        except Exception as e:
            print(f"Final call failed: {e}")
            return "I apologize, but I encountered an error while formulating my final response."

    def _fallback_response(self, query: str, system_content: str) -> str:
        """Fallback to single API call without tools on error."""
        try:
            api_params = {
                **self.base_params,
                "messages": [{"role": "user", "content": query}],
                "system": system_content,
            }

            response = self.client.messages.create(**api_params)
            return self._extract_text_response(response)

        except Exception as e:
            print(f"Fallback response failed: {e}")
            return "I'm sorry, I'm unable to process your request at this time. Please try again later."

from typing import Callable, Awaitable, Any, cast

from langchain.agents.middleware.types import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

# Import the necessary helpers from deepagents based on your provided code
from deepagents.backends import StateBackend, CompositeBackend
from deepagents.backends.protocol import BackendProtocol
from deepagents.middleware._message_eviction import (
    _extract_text_from_message,
    _offload_tool_message_content,
    _aoffload_tool_message_content,
)

NUM_CHARS_PER_TOKEN = 4

# Tools that you want to ignore for eviction
TOOLS_EXCLUDED_FROM_EVICTION = (
    "ls",
    "glob",
    "grep",
    "read_file",
    "edit_file",
    "write_file",
)

class LargeToolEvictionMiddleware(AgentMiddleware):
    """
    Middleware exclusively for evicting large tool results to a backend.
    Does not provide any filesystem tools to the agent.
    """

    def __init__(
        self,
        *,
        backend: Any = None,
        tool_token_limit_before_evict: int | None = 50000,
    ) -> None:
        self.backend = backend if backend is not None else StateBackend()
        self._tool_token_limit_before_evict = tool_token_limit_before_evict

        # Setup the prefix for saving large results
        artifacts_root = self.backend.artifacts_root if isinstance(self.backend, CompositeBackend) else "/"
        _root = artifacts_root.rstrip("/")
        self._large_tool_results_prefix = f"{_root}/large_tool_results"

        # Notice we DO NOT initialize self.tools = [...] here.

    def _get_backend(self, runtime) -> BackendProtocol:
        if callable(self.backend):
            return self.backend(runtime)
        return self.backend

    def _process_large_message(self, message: ToolMessage, resolved_backend: BackendProtocol) -> tuple[ToolMessage, bool]:
        if not self._tool_token_limit_before_evict:
            return message, False

        content_str = _extract_text_from_message(message)

        if len(content_str) <= NUM_CHARS_PER_TOKEN * self._tool_token_limit_before_evict:
            return message, False

        processed_message = _offload_tool_message_content(
            message,
            content_str,
            resolved_backend,
            self._large_tool_results_prefix,
        )
        if processed_message is None:
            return message, False
        return processed_message, True

    async def _aprocess_large_message(self, message: ToolMessage, resolved_backend: BackendProtocol) -> tuple[ToolMessage, bool]:
        if not self._tool_token_limit_before_evict:
            return message, False

        content_str = _extract_text_from_message(message)

        if len(content_str) <= NUM_CHARS_PER_TOKEN * self._tool_token_limit_before_evict:
            return message, False

        processed_message = await _aoffload_tool_message_content(
            message,
            content_str,
            resolved_backend,
            self._large_tool_results_prefix,
        )
        if processed_message is None:
            return message, False
        return processed_message, True

    def _intercept_large_tool_result(self, tool_result: ToolMessage | Command, runtime) -> ToolMessage | Command:
        if isinstance(tool_result, ToolMessage):
            resolved_backend = self._get_backend(runtime)
            processed_message, _ = self._process_large_message(tool_result, resolved_backend)
            return processed_message

        if isinstance(tool_result, Command):
            update = tool_result.update
            if update is None:
                return tool_result
            command_messages = update.get("messages", [])
            resolved_backend = self._get_backend(runtime)
            processed_messages = []
            for message in command_messages:
                if not isinstance(message, ToolMessage):
                    processed_messages.append(message)
                    continue

                processed_message, _ = self._process_large_message(message, resolved_backend)
                processed_messages.append(processed_message)
            return Command(update={**update, "messages": processed_messages})

        raise AssertionError(f"Unreachable code reached in _intercept_large_tool_result: for tool_result of type {type(tool_result)}")

    async def _aintercept_large_tool_result(self, tool_result: ToolMessage | Command, runtime) -> ToolMessage | Command:
        if isinstance(tool_result, ToolMessage):
            resolved_backend = self._get_backend(runtime)
            processed_message, _ = await self._aprocess_large_message(tool_result, resolved_backend)
            return processed_message

        if isinstance(tool_result, Command):
            update = tool_result.update
            if update is None:
                return tool_result
            command_messages = update.get("messages", [])
            resolved_backend = self._get_backend(runtime)
            processed_messages = []
            for message in command_messages:
                if not isinstance(message, ToolMessage):
                    processed_messages.append(message)
                    continue

                processed_message, _ = await self._aprocess_large_message(message, resolved_backend)
                processed_messages.append(processed_message)
            return Command(update={**update, "messages": processed_messages})

        raise AssertionError(f"Unreachable code reached in _aintercept_large_tool_result: for tool_result of type {type(tool_result)}")

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        tool_result = handler(request)

        if self._tool_token_limit_before_evict is None or request.tool_call["name"] in TOOLS_EXCLUDED_FROM_EVICTION:
            return tool_result

        return self._intercept_large_tool_result(tool_result, request.runtime)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        tool_result = await handler(request)

        if self._tool_token_limit_before_evict is None or request.tool_call["name"] in TOOLS_EXCLUDED_FROM_EVICTION:
            return tool_result

        return await self._aintercept_large_tool_result(tool_result, request.runtime)

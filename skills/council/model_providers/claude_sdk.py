"""
Claude SDK provider using the Claude Agent SDK.

Uses the claude-agent-sdk package which authenticates via the bundled
Claude Code CLI - no API key required.
"""

import asyncio
import shutil
import time
from typing import AsyncIterator, Optional

from .base import BaseProvider
from core.models import LLMResponse

# Lazy import to avoid hard dependency
_claude_sdk_available: Optional[bool] = None


def _check_claude_sdk() -> bool:
    """Check if claude-agent-sdk is installed."""
    global _claude_sdk_available
    if _claude_sdk_available is None:
        try:
            import claude_agent_sdk
            _claude_sdk_available = True
        except ImportError:
            _claude_sdk_available = False
    return _claude_sdk_available


class ClaudeSDKProvider(BaseProvider):
    """
    Provider for Claude using the Claude Agent SDK.

    Uses existing Claude Code CLI authentication - no API key needed.
    Supports streaming responses.
    """

    def __init__(self):
        super().__init__('claude')

    def is_available(self) -> bool:
        """Check if Claude SDK is available and CLI is authenticated."""
        # Check if SDK is installed
        if not _check_claude_sdk():
            return False
        # Check if Claude CLI is available (SDK uses it for auth)
        return shutil.which('claude') is not None

    def supports_streaming(self) -> bool:
        """Claude SDK natively supports streaming."""
        return True

    async def query(self, prompt: str, timeout: int) -> LLMResponse:
        """
        Query Claude and wait for the complete response.

        Args:
            prompt: The prompt to send to Claude.
            timeout: Maximum time to wait in seconds.

        Returns:
            LLMResponse with the complete response.
        """
        if not self.is_available():
            return LLMResponse(
                content='',
                model=self._name,
                latency_ms=0,
                success=False,
                error='Claude SDK not available or CLI not authenticated'
            )

        start = time.time()

        try:
            from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock

            # Configure for simple text response (no tools)
            options = ClaudeAgentOptions(
                allowed_tools=[],  # No tools - just text response
                max_turns=1,       # Single turn query
            )

            result_text = ""

            # Use asyncio.wait_for for timeout
            async def collect_response():
                nonlocal result_text
                async for message in query(prompt=prompt, options=options):
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                result_text += block.text

            await asyncio.wait_for(collect_response(), timeout=timeout)

            return self._create_response(
                content=result_text,
                start_time=start,
                success=True
            )

        except asyncio.TimeoutError:
            return self._create_error_response('TIMEOUT', start)
        except Exception as e:
            return self._create_error_response(str(e), start)

    async def query_stream(self, prompt: str, timeout: int = 420) -> AsyncIterator[str]:
        """
        Query Claude and stream the response.

        Args:
            prompt: The prompt to send to Claude.
            timeout: Maximum time to wait in seconds.

        Yields:
            Text chunks as they become available.
        """
        if not self.is_available():
            raise RuntimeError('Claude SDK not available or CLI not authenticated')

        try:
            from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock

            options = ClaudeAgentOptions(
                allowed_tools=[],
                max_turns=1,
            )

            async def stream_with_timeout():
                async for message in query(prompt=prompt, options=options):
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                yield block.text

            # Wrap in timeout
            start = time.time()
            async for chunk in stream_with_timeout():
                if time.time() - start > timeout:
                    raise asyncio.TimeoutError()
                yield chunk

        except asyncio.TimeoutError:
            raise RuntimeError('TIMEOUT')
        except Exception as e:
            raise RuntimeError(str(e))

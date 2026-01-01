"""
Base protocol for model providers.

Defines the unified interface that all providers must implement.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional
import time

from core.models import LLMResponse


class ProviderProtocol(ABC):
    """
    Abstract base class for model providers.

    All providers (SDK and CLI) must implement this interface to ensure
    consistent behavior across different model backends.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the model name (e.g., 'claude', 'gemini', 'codex')."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the provider is available and authenticated.

        Returns:
            True if the provider can be used, False otherwise.
        """
        ...

    @abstractmethod
    async def query(self, prompt: str, timeout: int) -> LLMResponse:
        """
        Send a prompt and wait for the complete response.

        Args:
            prompt: The prompt to send to the model.
            timeout: Maximum time to wait in seconds.

        Returns:
            LLMResponse with the complete response.
        """
        ...

    async def query_stream(self, prompt: str, timeout: int = 420) -> AsyncIterator[str]:
        """
        Send a prompt and stream the response.

        Default implementation calls query() and yields the full result.
        Override for true streaming support.

        Args:
            prompt: The prompt to send to the model.
            timeout: Maximum time to wait in seconds.

        Yields:
            Text chunks as they become available.
        """
        response = await self.query(prompt, timeout)
        if response.success:
            yield response.content
        else:
            raise RuntimeError(f"Query failed: {response.error}")

    def supports_streaming(self) -> bool:
        """
        Check if the provider supports true streaming.

        Returns:
            True if streaming is natively supported, False if emulated.
        """
        return False


class BaseProvider(ProviderProtocol):
    """
    Base implementation with common utilities.

    Provides timing measurement and error handling helpers.
    """

    def __init__(self, model_name: str):
        self._name = model_name
        self._last_latency_ms: Optional[int] = None

    @property
    def name(self) -> str:
        return self._name

    def _create_response(
        self,
        content: str,
        start_time: float,
        success: bool = True,
        error: Optional[str] = None
    ) -> LLMResponse:
        """Create an LLMResponse with timing information."""
        latency_ms = int((time.time() - start_time) * 1000)
        self._last_latency_ms = latency_ms
        return LLMResponse(
            content=content,
            model=self._name,
            latency_ms=latency_ms,
            success=success,
            error=error
        )

    def _create_error_response(
        self,
        error: str,
        start_time: float
    ) -> LLMResponse:
        """Create an error LLMResponse."""
        return self._create_response(
            content='',
            start_time=start_time,
            success=False,
            error=error
        )

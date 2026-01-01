"""
Codex SDK provider using the OpenAI SDK.

Uses the openai package with local auth from ~/.codex/auth.json
NO API key required - reads existing CLI login tokens.
"""

import asyncio
import time
from typing import AsyncIterator, Optional

from .base import BaseProvider
from .local_auth import get_codex_auth
from core.models import LLMResponse

# Lazy import to avoid hard dependency
_openai_sdk_available: Optional[bool] = None


def _check_openai_sdk() -> bool:
    """Check if openai SDK is installed."""
    global _openai_sdk_available
    if _openai_sdk_available is None:
        try:
            import openai
            _openai_sdk_available = True
        except ImportError:
            _openai_sdk_available = False
    return _openai_sdk_available


class CodexSDKProvider(BaseProvider):
    """
    Provider for Codex using the OpenAI SDK.

    Uses local auth from ~/.codex/auth.json (stored by `codex login`).
    Supports streaming responses.
    """

    def __init__(self, model: str = 'gpt-4o'):
        super().__init__('codex')
        self._model_id = model
        self._client = None

    def _get_client(self):
        """Lazy initialization of OpenAI client with local auth."""
        if self._client is None:
            auth = get_codex_auth()
            if not auth:
                raise RuntimeError("No local Codex auth found")

            import openai
            self._client = openai.AsyncOpenAI(api_key=auth.token)
        return self._client

    def is_available(self) -> bool:
        """Check if OpenAI SDK is available and local auth exists."""
        if not _check_openai_sdk():
            return False
        return get_codex_auth() is not None

    def supports_streaming(self) -> bool:
        """OpenAI SDK supports streaming."""
        return True

    async def query(self, prompt: str, timeout: int) -> LLMResponse:
        """
        Query Codex/GPT and wait for the complete response.

        Args:
            prompt: The prompt to send.
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
                error='Codex SDK not available or no local auth'
            )

        start = time.time()

        try:
            client = self._get_client()

            async def generate():
                response = await client.chat.completions.create(
                    model=self._model_id,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=timeout
                )
                return response.choices[0].message.content

            result_text = await asyncio.wait_for(generate(), timeout=timeout)

            return self._create_response(
                content=result_text or '',
                start_time=start,
                success=True
            )

        except asyncio.TimeoutError:
            return self._create_error_response('TIMEOUT', start)
        except Exception as e:
            return self._create_error_response(str(e), start)

    async def query_stream(self, prompt: str, timeout: int = 420) -> AsyncIterator[str]:
        """
        Query Codex/GPT and stream the response.

        Args:
            prompt: The prompt to send.
            timeout: Maximum time to wait in seconds.

        Yields:
            Text chunks as they become available.
        """
        if not self.is_available():
            raise RuntimeError('Codex SDK not available or no local auth')

        try:
            client = self._get_client()
            start = time.time()

            stream = await client.chat.completions.create(
                model=self._model_id,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                timeout=timeout
            )

            async for chunk in stream:
                if time.time() - start > timeout:
                    raise asyncio.TimeoutError()
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except asyncio.TimeoutError:
            raise RuntimeError('TIMEOUT')
        except Exception as e:
            raise RuntimeError(str(e))

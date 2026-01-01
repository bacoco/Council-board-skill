"""
Gemini SDK provider using Google Generative AI.

Uses Application Default Credentials (ADC) for authentication.
No explicit API key required if gcloud auth is configured.
"""

import asyncio
import shutil
import time
from typing import AsyncIterator, Optional

from .base import BaseProvider
from core.models import LLMResponse

# Lazy import to avoid hard dependency
_gemini_sdk_available: Optional[bool] = None
_adc_available: Optional[bool] = None


def _check_gemini_sdk() -> bool:
    """Check if google-generativeai is installed."""
    global _gemini_sdk_available
    if _gemini_sdk_available is None:
        try:
            import google.generativeai
            _gemini_sdk_available = True
        except ImportError:
            _gemini_sdk_available = False
    return _gemini_sdk_available


def _check_adc() -> bool:
    """Check if Application Default Credentials are available."""
    global _adc_available
    if _adc_available is None:
        try:
            import google.auth
            credentials, project = google.auth.default()
            _adc_available = credentials is not None
        except Exception:
            _adc_available = False
    return _adc_available


class GeminiSDKProvider(BaseProvider):
    """
    Provider for Gemini using Google Generative AI SDK.

    Uses Application Default Credentials (ADC) - no explicit API key needed.
    Requires: gcloud auth application-default login
    Supports streaming responses.
    """

    def __init__(self, model_name: str = 'gemini-2.0-flash'):
        super().__init__('gemini')
        self._model_id = model_name
        self._model = None

    def _get_model(self):
        """Lazy initialization of the Gemini model."""
        if self._model is None:
            import google.generativeai as genai
            # ADC will be used automatically if available
            # No explicit configure() call needed with ADC
            self._model = genai.GenerativeModel(self._model_id)
        return self._model

    def is_available(self) -> bool:
        """Check if Gemini SDK and ADC are available."""
        # Check if SDK is installed
        if not _check_gemini_sdk():
            return False
        # Check if ADC is configured (gcloud auth)
        if not _check_adc():
            # Fallback: check if gemini CLI is available
            return shutil.which('gemini') is not None
        return True

    def supports_streaming(self) -> bool:
        """Gemini SDK natively supports streaming."""
        return True

    async def query(self, prompt: str, timeout: int) -> LLMResponse:
        """
        Query Gemini and wait for the complete response.

        Args:
            prompt: The prompt to send to Gemini.
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
                error='Gemini SDK not available or ADC not configured'
            )

        start = time.time()

        try:
            model = self._get_model()

            # Use asyncio.wait_for for timeout
            async def generate():
                response = await model.generate_content_async(prompt)
                return response.text

            result_text = await asyncio.wait_for(generate(), timeout=timeout)

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
        Query Gemini and stream the response.

        Args:
            prompt: The prompt to send to Gemini.
            timeout: Maximum time to wait in seconds.

        Yields:
            Text chunks as they become available.
        """
        if not self.is_available():
            raise RuntimeError('Gemini SDK not available or ADC not configured')

        try:
            model = self._get_model()

            start = time.time()

            # Generate with streaming
            response = await model.generate_content_async(
                prompt,
                stream=True
            )

            async for chunk in response:
                if time.time() - start > timeout:
                    raise asyncio.TimeoutError()
                if chunk.text:
                    yield chunk.text

        except asyncio.TimeoutError:
            raise RuntimeError('TIMEOUT')
        except Exception as e:
            raise RuntimeError(str(e))

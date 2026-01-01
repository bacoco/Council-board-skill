"""
Model adapters for Council queries.

Provides unified interface for querying Claude, Gemini, and Codex.
Uses SDK providers when available (with streaming), falls back to CLI.
Includes retry logic, circuit breaker protection, and chairman failover.
"""

import asyncio
import functools
import random
import shutil
import time
from typing import AsyncIterator, Dict, List, Optional, TYPE_CHECKING

from .models import LLMResponse
from .emit import emit, emit_perf_metric
from .state import CIRCUIT_BREAKER, DEFAULT_TIMEOUT
from .retry import is_retriable_error

# Import provider factory
from model_providers import get_provider, get_available_providers, get_provider_info, ProviderProtocol

if TYPE_CHECKING:
    from .models import SessionConfig


# Cache for providers (one per model)
_PROVIDER_CACHE: Dict[str, ProviderProtocol] = {}


def _get_cached_provider(model: str) -> ProviderProtocol:
    """Get a cached provider for the model."""
    if model not in _PROVIDER_CACHE:
        _PROVIDER_CACHE[model] = get_provider(model, prefer_streaming=True)
    return _PROVIDER_CACHE[model]


@functools.lru_cache(maxsize=32)
def check_cli_available(cli: str) -> bool:
    """
    Check if a CLI tool is available in PATH. Results are cached.

    Synchronous version for use in non-async contexts (e.g., main() startup).
    """
    start = time.perf_counter()
    try:
        # Use shutil.which() instead of subprocess for better cross-platform support
        result = shutil.which(cli)
        available = result is not None
    except Exception:
        available = False

    elapsed_ms = (time.perf_counter() - start) * 1000
    emit_perf_metric("check_cli_available", elapsed_ms, cli=cli, available=available)
    return available


async def check_cli_available_async(cli: str) -> bool:
    """
    Async version of CLI availability check. Non-blocking.

    Uses asyncio.to_thread to run shutil.which() in thread pool.
    """
    start = time.perf_counter()
    try:
        # Run shutil.which in thread pool to avoid blocking event loop
        result = await asyncio.to_thread(shutil.which, cli)
        available = result is not None
    except Exception:
        available = False

    elapsed_ms = (time.perf_counter() - start) * 1000
    emit_perf_metric("check_cli_available_async", elapsed_ms, cli=cli, available=available)
    return available


def check_model_available(model: str) -> bool:
    """
    Check if a model is available via any provider (SDK or CLI).

    Args:
        model: Model name to check

    Returns:
        True if the model is available
    """
    try:
        provider = get_provider(model)
        return provider.is_available()
    except (ValueError, RuntimeError):
        return False


def get_available_models(requested_models: List[str]) -> List[str]:
    """
    Detect which models are available (via SDK or CLI).

    Args:
        requested_models: List of model names to check

    Returns:
        List of available model names
    """
    available = []
    for model in requested_models:
        if check_model_available(model):
            available.append(model)
    return available


def expand_models_with_fallback(requested_models: List[str], min_models: int = 3) -> List[str]:
    """
    Get available models from requested list.

    Does NOT duplicate models - duplicating the same CLI creates fake diversity,
    triples API costs, and biases convergence (same model agrees with itself).
    Instead, returns available models as-is and lets DegradationState apply
    confidence penalties appropriately.

    Args:
        requested_models: List of requested model names
        min_models: Advisory minimum (used for warning only, not enforced here)

    Returns:
        List of available model names (no duplicates)
    """
    available = get_available_models(requested_models)

    if len(available) == 0:
        raise RuntimeError("No models are available. Please install and authenticate at least one of: claude, gemini, codex")

    if len(available) < min_models:
        # Warn about degraded operation - DegradationState will apply confidence penalty
        emit({
            'type': 'degraded_start',
            'requested': requested_models,
            'available': available,
            'msg': f'Operating with {len(available)}/{min_models} models - confidence penalty will apply'
        })

    return available


async def query_model(model_name: str, prompt: str, timeout: int) -> LLMResponse:
    """
    Query a model using the best available provider.

    Uses SDK providers when available (with streaming support),
    falls back to CLI providers otherwise.

    Args:
        model_name: Name of the model (claude, gemini, codex)
        prompt: The prompt to send to the model
        timeout: Timeout in seconds

    Returns:
        LLMResponse with content, latency, and success status
    """
    try:
        provider = _get_cached_provider(model_name)
        return await provider.query(prompt, timeout)
    except Exception as e:
        return LLMResponse(
            content='',
            model=model_name,
            latency_ms=0,
            success=False,
            error=str(e)
        )


async def query_model_stream(model_name: str, prompt: str, timeout: int = 420) -> AsyncIterator[str]:
    """
    Query a model with streaming response.

    Uses SDK providers when available for true streaming,
    falls back to batch response for CLI providers.

    Args:
        model_name: Name of the model
        prompt: The prompt to send
        timeout: Timeout in seconds

    Yields:
        Text chunks as they become available
    """
    provider = _get_cached_provider(model_name)
    async for chunk in provider.query_stream(prompt, timeout):
        yield chunk


async def query_with_retry(model_name: str, prompt: str, timeout: int, max_retries: int = 3) -> LLMResponse:
    """
    Query model with exponential backoff retry logic and circuit breaker protection.

    Uses SDK providers when available, falls back to CLI.

    Args:
        model_name: Name of the model
        prompt: The prompt to send
        timeout: Timeout per attempt in seconds
        max_retries: Maximum number of retry attempts (default: 3)

    Returns:
        LLMResponse with content, latency, and success status
    """
    # Check circuit breaker before calling
    if not CIRCUIT_BREAKER.can_call(model_name):
        return LLMResponse(
            content='',
            model=model_name,
            latency_ms=0,
            success=False,
            error=f'Circuit breaker OPEN for {model_name} - model temporarily excluded due to repeated failures'
        )

    last_error = None

    for attempt in range(max_retries):
        result = await query_model(model_name, prompt, timeout)

        # Success - record and return immediately
        if result.success:
            CIRCUIT_BREAKER.record_success(model_name)
            if attempt > 0:
                emit_perf_metric("query_retry_success", 0, model=model_name, attempt=attempt + 1)
            return result

        # Failure - store error and check if retriable
        last_error = result.error

        # Check if error is permanent (auth, not found, etc.) - don't retry these
        if not is_retriable_error(last_error):
            emit_perf_metric("query_permanent_error", 0, model=model_name, error=last_error)
            CIRCUIT_BREAKER.record_failure(model_name, last_error)
            return result  # Return immediately, no point retrying

        if attempt < max_retries - 1:
            # Exponential backoff with jitter to prevent timing attacks
            backoff = (2 ** attempt) + random.uniform(0, 1)
            emit_perf_metric("query_retry_backoff", backoff * 1000, model=model_name, attempt=attempt + 1)
            await asyncio.sleep(backoff)

    # All retries exhausted - record failure and return
    CIRCUIT_BREAKER.record_failure(model_name, last_error)
    emit_perf_metric("query_retry_exhausted", 0, model=model_name, retries=max_retries)
    return LLMResponse(
        content='',
        model=model_name,
        latency_ms=0,
        success=False,
        error=f'All {max_retries} retries failed. Last error: {last_error}'
    )


# Adapter functions (use retry logic for improved resilience)
async def query_claude(prompt: str, timeout: int) -> LLMResponse:
    """Query Claude using SDK (preferred) or CLI fallback."""
    return await query_with_retry('claude', prompt, timeout, max_retries=3)


async def query_gemini(prompt: str, timeout: int) -> LLMResponse:
    """Query Gemini using SDK with ADC (preferred) or CLI fallback."""
    return await query_with_retry('gemini', prompt, timeout, max_retries=3)


async def query_codex(prompt: str, timeout: int) -> LLMResponse:
    """Query Codex using CLI (no SDK without API key)."""
    return await query_with_retry('codex', prompt, timeout, max_retries=3)


# Backward-compatible ADAPTERS dict
ADAPTERS = {
    'claude': query_claude,
    'gemini': query_gemini,
    'codex': query_codex,
}


# Chairman failover chain (Council recommendation #1)
CHAIRMAN_FALLBACK_ORDER = ['claude', 'gemini', 'codex']


def get_chairman_with_fallback(preferred_chairman: str) -> str:
    """
    Get a working chairman model with failover to alternates.

    If the preferred chairman (typically Claude) is unavailable due to circuit
    breaker state or provider unavailability, falls back to next available model.

    Args:
        preferred_chairman: The configured chairman model (e.g., 'claude')

    Returns:
        Name of the best available model to use as chairman
    """
    # Build fallback order: preferred first, then others in priority order
    fallback_order = [preferred_chairman] + [m for m in CHAIRMAN_FALLBACK_ORDER if m != preferred_chairman]

    for model in fallback_order:
        if CIRCUIT_BREAKER.can_call(model) and check_model_available(model):
            if model != preferred_chairman:
                emit({"type": "chairman_failover", "preferred": preferred_chairman, "using": model})
            return model

    # All models unavailable - return preferred and let caller handle failure
    emit({"type": "chairman_failover_exhausted", "msg": "All chairman candidates unavailable"})
    return preferred_chairman


async def query_chairman(prompt: str, config: 'SessionConfig') -> LLMResponse:
    """
    Query the chairman model with automatic failover.

    Uses get_chairman_with_fallback to select best available model,
    then queries it with retry logic.

    Args:
        prompt: The prompt to send
        config: Session configuration containing chairman and timeout

    Returns:
        LLMResponse from the best available chairman model
    """
    chairman = get_chairman_with_fallback(config.chairman)

    if chairman in ADAPTERS:
        return await ADAPTERS[chairman](prompt, config.timeout)

    return LLMResponse(
        content='',
        model=chairman,
        latency_ms=0,
        success=False,
        error=f'No adapter available for chairman: {chairman}'
    )

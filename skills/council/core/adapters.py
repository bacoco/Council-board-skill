"""
CLI adapters for model queries.

Provides unified interface for querying Claude, Gemini, and Codex CLIs.
Includes retry logic, circuit breaker protection, and chairman failover.
"""

import asyncio
import functools
import json
import random
import shutil
import time
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from .models import LLMResponse, CLIConfig
from .emit import emit, emit_perf_metric
from .state import CIRCUIT_BREAKER, DEFAULT_TIMEOUT
from .retry import is_retriable_error

if TYPE_CHECKING:
    from .models import SessionConfig


# Cache for project root (computed once per session)
_PROJECT_ROOT_CACHE: Optional[Path] = None


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


def get_available_models(requested_models: List[str]) -> List[str]:
    """
    Detect which CLI tools are available.

    Args:
        requested_models: List of model names to check

    Returns:
        List of available model names
    """
    available = []
    for model in requested_models:
        if check_cli_available(model):
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
        raise RuntimeError("No model CLIs are available. Please install and authenticate at least one of: claude, gemini, codex")

    if len(available) < min_models:
        # Warn about degraded operation - DegradationState will apply confidence penalty
        emit({
            'type': 'degraded_start',
            'requested': requested_models,
            'available': available,
            'msg': f'Operating with {len(available)}/{min_models} models - confidence penalty will apply'
        })

    return available


def find_project_root() -> Optional[Path]:
    """
    Find the project root directory by looking for common project markers.

    Searches upward from current directory for .git, package.json, pyproject.toml, etc.
    Result is cached for the session to avoid repeated filesystem lookups.

    Returns:
        Path to project root, or current directory if no markers found
    """
    global _PROJECT_ROOT_CACHE
    if _PROJECT_ROOT_CACHE is not None:
        return _PROJECT_ROOT_CACHE

    cwd = Path.cwd()
    markers = ['.git', 'package.json', 'pyproject.toml', 'Cargo.toml', 'go.mod', 'pom.xml']

    for parent in [cwd] + list(cwd.parents):
        for marker in markers:
            if (parent / marker).exists():
                _PROJECT_ROOT_CACHE = parent
                return parent

    # No marker found, use current directory
    _PROJECT_ROOT_CACHE = cwd
    return cwd


async def query_cli(model_name: str, cli_config: CLIConfig, prompt: str, timeout: int) -> LLMResponse:
    """
    Generic CLI query function that works for all model CLIs.

    Args:
        model_name: Name of the model (for response tracking)
        cli_config: CLI configuration (command, args, stdin usage)
        prompt: The prompt to send to the model
        timeout: Timeout in seconds

    Returns:
        LLMResponse with content, latency, and success status
    """
    start = time.time()
    proc = None  # Track subprocess for cleanup on timeout/error

    try:
        # Build command
        cmd = [cli_config.name] + cli_config.args

        # Get project root for CLI working directory
        # This allows CLIs like codex to explore the project context
        project_root = find_project_root()

        # Create subprocess with project root as working directory
        if cli_config.use_stdin:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=project_root
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=prompt.encode()),
                timeout=timeout
            )
        else:
            # Add prompt to args
            cmd.extend(['-p', prompt])
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=project_root
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)

        latency = int((time.time() - start) * 1000)

        if proc.returncode == 0:
            content = stdout.decode()

            # Special handling for Claude's JSON output format
            if model_name == 'claude' and '--output-format' in cmd:
                try:
                    data = json.loads(content)
                    content = data.get('result', content)
                except json.JSONDecodeError:
                    pass  # Use raw content if not valid JSON

            return LLMResponse(
                content=content,
                model=model_name,
                latency_ms=latency,
                success=True
            )
        else:
            return LLMResponse(
                content='',
                model=model_name,
                latency_ms=latency,
                success=False,
                error=stderr.decode()
            )

    except asyncio.TimeoutError:
        latency = int((time.time() - start) * 1000)
        # Clean up subprocess on timeout - close pipes first to prevent event loop warnings
        if proc is not None:
            try:
                if proc.stdin:
                    proc.stdin.close()
                if proc.stdout:
                    proc.stdout.close()
                if proc.stderr:
                    proc.stderr.close()
                proc.kill()
            except Exception:
                pass
        return LLMResponse(
            content='',
            model=model_name,
            latency_ms=latency,
            success=False,
            error='TIMEOUT'
        )
    except Exception as e:
        latency = int((time.time() - start) * 1000)
        # Clean up subprocess on error - close pipes first to prevent event loop warnings
        if proc is not None:
            try:
                if proc.stdin:
                    proc.stdin.close()
                if proc.stdout:
                    proc.stdout.close()
                if proc.stderr:
                    proc.stderr.close()
                proc.kill()
            except Exception:
                pass
        return LLMResponse(
            content='',
            model=model_name,
            latency_ms=latency,
            success=False,
            error=str(e)
        )


async def query_cli_with_retry(model_name: str, cli_config: CLIConfig, prompt: str, timeout: int, max_retries: int = 3) -> LLMResponse:
    """
    Query CLI with exponential backoff retry logic and circuit breaker protection.

    Args:
        model_name: Name of the model
        cli_config: CLI configuration
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
        result = await query_cli(model_name, cli_config, prompt, timeout)

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


# CLI configurations for each model
CLI_CONFIGS = {
    'claude': CLIConfig(
        name='claude',
        args=['--output-format', 'json'],
        use_stdin=False
    ),
    'gemini': CLIConfig(
        name='gemini',
        args=[],
        use_stdin=False
    ),
    'codex': CLIConfig(
        name='codex',
        args=['exec'],
        use_stdin=True
    ),
}


# Adapter functions (use retry logic for improved resilience)
async def query_claude(prompt: str, timeout: int) -> LLMResponse:
    return await query_cli_with_retry('claude', CLI_CONFIGS['claude'], prompt, timeout, max_retries=3)


async def query_gemini(prompt: str, timeout: int) -> LLMResponse:
    return await query_cli_with_retry('gemini', CLI_CONFIGS['gemini'], prompt, timeout, max_retries=3)


async def query_codex(prompt: str, timeout: int) -> LLMResponse:
    return await query_cli_with_retry('codex', CLI_CONFIGS['codex'], prompt, timeout, max_retries=3)


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
    breaker state or CLI unavailability, falls back to next available model.

    Args:
        preferred_chairman: The configured chairman model (e.g., 'claude')

    Returns:
        Name of the best available model to use as chairman
    """
    # Build fallback order: preferred first, then others in priority order
    fallback_order = [preferred_chairman] + [m for m in CHAIRMAN_FALLBACK_ORDER if m != preferred_chairman]

    for model in fallback_order:
        if CIRCUIT_BREAKER.can_call(model) and check_cli_available(model):
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

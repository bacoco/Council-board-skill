"""
CLI provider for models accessed via command-line tools.

Provides subprocess-based access to models (Codex, or fallback for Claude/Gemini).
Uses existing CLI authentication - no API keys required.
"""

import asyncio
import functools
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, List, Optional

from .base import BaseProvider
from core.models import LLMResponse


@dataclass
class CLIConfig:
    """Configuration for a CLI tool invocation."""
    name: str
    args: List[str]
    use_stdin: bool = False


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

# Cache for project root (computed once per session)
_PROJECT_ROOT_CACHE: Optional[Path] = None


def _cleanup_subprocess(proc) -> None:
    """
    Safely cleanup a subprocess by closing pipes and killing the process.

    Closes stdin/stdout/stderr first to prevent event loop warnings,
    then kills the process. Silently ignores any errors during cleanup.
    """
    if proc is None:
        return
    try:
        if proc.stdin:
            proc.stdin.close()
        if proc.stdout:
            proc.stdout.close()
        if proc.stderr:
            proc.stderr.close()
        proc.kill()
    except Exception:
        pass  # Cleanup errors are non-fatal


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


class CLIProvider(BaseProvider):
    """
    Provider for models accessed via command-line tools.

    Uses subprocess to call CLI tools (claude, gemini, codex).
    Authentication is handled by the CLI tools themselves.
    """

    def __init__(self, model_name: str):
        super().__init__(model_name)
        if model_name not in CLI_CONFIGS:
            raise ValueError(f"Unknown CLI model: {model_name}. Available: {list(CLI_CONFIGS.keys())}")
        self._config = CLI_CONFIGS[model_name]

    @functools.lru_cache(maxsize=32)
    def is_available(self) -> bool:
        """Check if the CLI tool is available in PATH."""
        return shutil.which(self._config.name) is not None

    def supports_streaming(self) -> bool:
        """CLI providers don't support streaming (batch only)."""
        return False

    async def query(self, prompt: str, timeout: int) -> LLMResponse:
        """
        Query the model via CLI subprocess.

        Args:
            prompt: The prompt to send to the model.
            timeout: Maximum time to wait in seconds.

        Returns:
            LLMResponse with the complete response.
        """
        start = time.time()
        proc = None

        try:
            # Build command
            cmd = [self._config.name] + self._config.args

            # Get project root for CLI working directory
            project_root = find_project_root()

            # Create subprocess with project root as working directory
            if self._config.use_stdin:
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

            if proc.returncode == 0:
                content = stdout.decode()

                # Special handling for Claude's JSON output format
                if self._name == 'claude' and '--output-format' in cmd:
                    try:
                        data = json.loads(content)
                        content = data.get('result', content)
                    except json.JSONDecodeError:
                        pass  # Use raw content if not valid JSON

                return self._create_response(
                    content=content,
                    start_time=start,
                    success=True
                )
            else:
                return self._create_error_response(stderr.decode(), start)

        except asyncio.TimeoutError:
            _cleanup_subprocess(proc)
            return self._create_error_response('TIMEOUT', start)
        except Exception as e:
            _cleanup_subprocess(proc)
            return self._create_error_response(str(e), start)

    async def query_stream(self, prompt: str, timeout: int = 420) -> AsyncIterator[str]:
        """
        CLI providers don't support true streaming.

        Falls back to the base implementation (query then yield).
        """
        response = await self.query(prompt, timeout)
        if response.success:
            yield response.content
        else:
            raise RuntimeError(f"Query failed: {response.error}")

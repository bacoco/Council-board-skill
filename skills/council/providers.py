"""
ModelProvider Abstraction Layer

Decouples Council from specific CLI implementations, enabling:
- Easy addition of new models (local LLMs, direct APIs)
- Testability via mock providers
- Consistent interface across all model types
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path
import asyncio
import subprocess
import time
import json
import yaml


@dataclass
class ProviderResponse:
    """Standardized response from any model provider."""
    content: str
    model: str
    latency_ms: float
    success: bool
    error: Optional[str] = None
    raw_response: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelProvider(ABC):
    """Abstract base class for all model providers."""

    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}

    @abstractmethod
    async def query(self, prompt: str, timeout: int = 60) -> ProviderResponse:
        """
        Send a prompt to the model and return the response.

        Args:
            prompt: The prompt to send
            timeout: Maximum time to wait for response (seconds)

        Returns:
            ProviderResponse with content and metadata
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available and configured."""
        pass

    @abstractmethod
    def get_version(self) -> Optional[str]:
        """Get the version of this provider/model."""
        pass


class CLIProvider(ModelProvider):
    """Base class for CLI-based model providers."""

    def __init__(self, name: str, command: str, args: List[str] = None,
                 use_stdin: bool = False, config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.command = command
        self.args = args or []
        self.use_stdin = use_stdin
        self._version_cache: Optional[str] = None

    def is_available(self) -> bool:
        """Check if CLI is installed and accessible."""
        try:
            result = subprocess.run(
                [self.command, '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def get_version(self) -> Optional[str]:
        """Get CLI version."""
        if self._version_cache:
            return self._version_cache

        try:
            result = subprocess.run(
                [self.command, '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                self._version_cache = result.stdout.strip().split('\n')[0]
                return self._version_cache
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return None

    async def query(self, prompt: str, timeout: int = 60) -> ProviderResponse:
        """Execute CLI command and return response."""
        start_time = time.perf_counter()

        try:
            if self.use_stdin:
                # Pass prompt via stdin
                cmd = [self.command] + self.args
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input=prompt.encode('utf-8')),
                    timeout=timeout
                )
            else:
                # Pass prompt as argument
                cmd = [self.command] + self.args + [prompt]
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )

            latency_ms = (time.perf_counter() - start_time) * 1000
            output = stdout.decode('utf-8').strip()

            if process.returncode != 0:
                error_msg = stderr.decode('utf-8').strip() or f'Exit code {process.returncode}'
                return ProviderResponse(
                    content='',
                    model=self.name,
                    latency_ms=latency_ms,
                    success=False,
                    error=error_msg
                )

            return ProviderResponse(
                content=output,
                model=self.name,
                latency_ms=latency_ms,
                success=True,
                raw_response=output
            )

        except asyncio.TimeoutError:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return ProviderResponse(
                content='',
                model=self.name,
                latency_ms=latency_ms,
                success=False,
                error=f'Timeout after {timeout}s'
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return ProviderResponse(
                content='',
                model=self.name,
                latency_ms=latency_ms,
                success=False,
                error=str(e)
            )


class ClaudeProvider(CLIProvider):
    """Claude CLI provider."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name='claude',
            command='claude',
            args=['--output-format', 'json'],
            use_stdin=False,
            config=config
        )

    async def query(self, prompt: str, timeout: int = 60) -> ProviderResponse:
        response = await super().query(prompt, timeout)

        # Parse JSON output from Claude CLI
        if response.success and response.content:
            try:
                data = json.loads(response.content)
                if isinstance(data, dict):
                    # Extract text from Claude's JSON response
                    text = data.get('result', data.get('content', response.content))
                    response.content = text if isinstance(text, str) else str(text)
                    response.metadata['raw_json'] = data
            except json.JSONDecodeError:
                # Keep raw output if not valid JSON
                pass

        return response


class GeminiProvider(CLIProvider):
    """Gemini CLI provider."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name='gemini',
            command='gemini',
            args=[],
            use_stdin=False,
            config=config
        )


class CodexProvider(CLIProvider):
    """Codex CLI provider."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name='codex',
            command='codex',
            args=['exec'],
            use_stdin=True,
            config=config
        )


# Provider registry
PROVIDERS: Dict[str, type] = {
    'claude': ClaudeProvider,
    'gemini': GeminiProvider,
    'codex': CodexProvider,
}


def get_provider(name: str, config: Dict[str, Any] = None) -> ModelProvider:
    """
    Get a provider instance by name.

    Args:
        name: Provider name (claude, gemini, codex)
        config: Optional provider-specific configuration

    Returns:
        Configured ModelProvider instance

    Raises:
        ValueError: If provider name is not recognized
    """
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {name}. Available: {list(PROVIDERS.keys())}")

    return PROVIDERS[name](config)


def get_available_providers() -> List[str]:
    """Get list of available (installed) providers."""
    available = []
    for name, provider_class in PROVIDERS.items():
        provider = provider_class()
        if provider.is_available():
            available.append(name)
    return available


@dataclass
class CouncilConfig:
    """Configuration for Council sessions."""
    providers: List[str] = field(default_factory=lambda: ['claude', 'gemini', 'codex'])
    chairman: str = 'claude'
    timeout: int = 60
    max_rounds: int = 3
    mode: str = 'adaptive'
    convergence_threshold: float = 0.8
    min_quorum: int = 2

    @classmethod
    def from_file(cls, path: Path) -> 'CouncilConfig':
        """Load configuration from YAML file."""
        if not path.exists():
            return cls()

        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}

        return cls(
            providers=data.get('providers', cls.providers),
            chairman=data.get('chairman', 'claude'),
            timeout=data.get('timeout', 60),
            max_rounds=data.get('max_rounds', 3),
            mode=data.get('mode', 'adaptive'),
            convergence_threshold=data.get('convergence_threshold', 0.8),
            min_quorum=data.get('min_quorum', 2)
        )

    @classmethod
    def default_path(cls) -> Path:
        """Get default config file path."""
        return Path(__file__).parent.parent / 'council.config.yaml'

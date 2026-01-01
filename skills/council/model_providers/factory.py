"""
Provider factory for automatic provider selection.

Selects the best available provider for each model:
- SDK providers are preferred (better streaming, lower latency)
- CLI providers are used as fallback
"""

from typing import Dict, List, Optional, Type

from .base import ProviderProtocol
from .cli_provider import CLIProvider


# Lazy imports to avoid hard dependencies
def _get_claude_sdk_provider():
    """Lazy import of Claude SDK provider."""
    try:
        from .claude_sdk import ClaudeSDKProvider
        return ClaudeSDKProvider
    except ImportError:
        return None


def _get_gemini_sdk_provider():
    """Lazy import of Gemini SDK provider."""
    try:
        from .gemini_sdk import GeminiSDKProvider
        return GeminiSDKProvider
    except ImportError:
        return None


# Provider preference order per model
# First available provider is used
PROVIDER_PREFERENCES: Dict[str, List[str]] = {
    'claude': ['sdk', 'cli'],   # Prefer Claude Agent SDK, fallback to CLI
    'gemini': ['sdk', 'cli'],   # Prefer Gemini SDK with ADC, fallback to CLI
    'codex': ['cli'],           # Codex only has CLI (no SDK without API key)
}


def get_provider(model: str, prefer_streaming: bool = True) -> ProviderProtocol:
    """
    Get the best available provider for a model.

    Args:
        model: Model name ('claude', 'gemini', 'codex')
        prefer_streaming: If True, prefer providers with streaming support

    Returns:
        The best available provider for the model.

    Raises:
        RuntimeError: If no provider is available for the model.
    """
    if model not in PROVIDER_PREFERENCES:
        raise ValueError(f"Unknown model: {model}. Available: {list(PROVIDER_PREFERENCES.keys())}")

    preferences = PROVIDER_PREFERENCES[model]

    for pref in preferences:
        provider = _try_get_provider(model, pref)
        if provider is not None and provider.is_available():
            # If streaming is preferred and this provider supports it, use it
            if prefer_streaming and provider.supports_streaming():
                return provider
            # If streaming not preferred or not available, still use if available
            if not prefer_streaming or pref == preferences[-1]:
                return provider
            # Keep looking for a streaming provider
            continue

    # Try CLI as last resort
    cli_provider = CLIProvider(model)
    if cli_provider.is_available():
        return cli_provider

    raise RuntimeError(f"No provider available for model: {model}")


def _try_get_provider(model: str, provider_type: str) -> Optional[ProviderProtocol]:
    """
    Try to instantiate a specific provider type for a model.

    Args:
        model: Model name
        provider_type: 'sdk' or 'cli'

    Returns:
        Provider instance or None if not available.
    """
    if provider_type == 'sdk':
        if model == 'claude':
            provider_class = _get_claude_sdk_provider()
            if provider_class:
                return provider_class()
        elif model == 'gemini':
            provider_class = _get_gemini_sdk_provider()
            if provider_class:
                return provider_class()
        # Codex has no SDK without API key
        return None

    elif provider_type == 'cli':
        try:
            return CLIProvider(model)
        except ValueError:
            return None

    return None


def get_available_providers() -> Dict[str, ProviderProtocol]:
    """
    Get all available providers.

    Returns:
        Dict mapping model names to their available providers.
    """
    available = {}

    for model in PROVIDER_PREFERENCES:
        try:
            provider = get_provider(model, prefer_streaming=True)
            available[model] = provider
        except RuntimeError:
            pass  # Model not available

    return available


def get_provider_info() -> Dict[str, Dict[str, str]]:
    """
    Get information about available providers.

    Returns:
        Dict with provider status for each model.
    """
    info = {}

    for model in PROVIDER_PREFERENCES:
        model_info = {
            'available': False,
            'provider_type': 'none',
            'streaming': False,
        }

        try:
            provider = get_provider(model, prefer_streaming=True)
            model_info['available'] = True
            model_info['provider_type'] = type(provider).__name__
            model_info['streaming'] = provider.supports_streaming()
        except RuntimeError:
            pass

        info[model] = model_info

    return info

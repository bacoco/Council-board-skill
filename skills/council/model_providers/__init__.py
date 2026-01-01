"""
Provider abstraction layer for Council.

Provides unified interface for model providers:
- Claude: via Claude Agent SDK (uses existing CLI auth)
- Gemini: via Google Generative AI SDK (uses ADC)
- Codex: via CLI subprocess (existing implementation)
"""

from .base import ProviderProtocol, BaseProvider
from .cli_provider import CLIProvider
from .factory import get_provider, get_available_providers, get_provider_info

__all__ = [
    'ProviderProtocol',
    'BaseProvider',
    'CLIProvider',
    'get_provider',
    'get_available_providers',
    'get_provider_info',
]

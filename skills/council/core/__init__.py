"""
Council Core - Shared infrastructure for multi-model deliberation.

Modules:
- models: Data classes (LLMResponse, SessionConfig, CLIConfig)
- state: Circuit breaker, degradation, adaptive timeout
- metrics: Session observability
- emit: Event emission with secret redaction
- adapters: CLI query wrappers
- prompts: Prompt builders
- parsing: JSON extraction and response parsing
- convergence: Convergence detection
- personas: Dynamic persona generation
- review: Peer review
- synthesis: Answer synthesis
- trail: Deliberation trail generation
- retry: Error classification
"""

from .models import LLMResponse, SessionConfig, CLIConfig, VoteBallot, VoteResult
from .retry import is_retriable_error, TRANSIENT_ERRORS, PERMANENT_ERRORS

__all__ = [
    'LLMResponse',
    'SessionConfig',
    'CLIConfig',
    'VoteBallot',
    'VoteResult',
    'is_retriable_error',
    'TRANSIENT_ERRORS',
    'PERMANENT_ERRORS',
]

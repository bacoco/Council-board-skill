"""
Data classes for Council deliberation.

Contains all structured data types used across the deliberation pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional, List


# Default constants (defined here to avoid circular imports)
DEFAULT_MIN_QUORUM = 2  # Minimum valid responses required per round


@dataclass
class LLMResponse:
    """Response from a single LLM query via CLI."""
    content: str
    model: str
    latency_ms: int
    success: bool
    error: Optional[str] = None
    parsed_json: Optional[dict] = field(default=None, repr=False)  # Cached JSON parsing


@dataclass
class SessionConfig:
    """Configuration for a council deliberation session."""
    query: str
    mode: str
    models: List[str]
    chairman: str
    timeout: int
    anonymize: bool
    council_budget: str
    output_level: str
    max_rounds: int
    min_quorum: int = DEFAULT_MIN_QUORUM  # Minimum valid responses per round
    enable_perf_metrics: bool = False
    enable_trail: bool = False  # Include detailed deliberation trail in output
    context: Optional[str] = None  # Code or additional context for analysis


@dataclass
class VoteBallot:
    """A single vote from a model in Vote mode."""
    model: str
    vote: str  # The option chosen (e.g., "A", "B", "C" or custom option)
    weight: float  # Confidence/reliability weight (0.0-1.0)
    justification: str  # Reasoning for the vote
    confidence: float  # Self-reported confidence
    latency_ms: int


@dataclass
class VoteResult:
    """Aggregated result from Vote mode deliberation."""
    winning_option: str
    vote_counts: dict  # {option: count}
    weighted_scores: dict  # {option: weighted_score}
    total_votes: int
    quorum_met: bool
    margin: float  # Winning margin as percentage
    ballots: List[VoteBallot]
    tie_broken: bool
    tie_breaker_method: Optional[str] = None


@dataclass
class CLIConfig:
    """Configuration for a CLI tool invocation."""
    name: str
    args: List[str]
    use_stdin: bool = False

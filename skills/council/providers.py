"""
Council Configuration

Provides CouncilConfig dataclass for loading settings from council.config.yaml.

Note: ModelProvider abstraction was removed as dead code. If you need to add
new model backends (local LLMs, direct APIs), consider refactoring council.py
to use a provider pattern.
"""

from dataclasses import dataclass, field
from typing import List
from pathlib import Path
import yaml


@dataclass
class CouncilConfig:
    """Configuration for Council sessions."""
    providers: List[str] = field(default_factory=lambda: ['claude', 'gemini', 'codex'])
    chairman: str = 'claude'
    timeout: int = 420  # 7 minutes - Codex needs time for code exploration
    max_rounds: int = 3
    mode: str = 'adaptive'
    convergence_threshold: float = 0.8
    min_quorum: int = 2
    enable_perf_metrics: bool = False
    enable_trail: bool = True  # Show deliberation trail by default
    # Pipeline selection: 'classic' (original) or 'storm' (STORM-inspired with Moderator/KB)
    pipeline: str = 'classic'

    @classmethod
    def from_file(cls, path: Path) -> 'CouncilConfig':
        """Load configuration from YAML file."""
        if not path.exists():
            return cls()

        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}

        defaults = cls()
        return cls(
            providers=data.get('providers', defaults.providers),
            chairman=data.get('chairman', defaults.chairman),
            timeout=data.get('timeout', defaults.timeout),
            max_rounds=data.get('max_rounds', defaults.max_rounds),
            mode=data.get('mode', defaults.mode),
            convergence_threshold=data.get('convergence_threshold', defaults.convergence_threshold),
            min_quorum=data.get('min_quorum', defaults.min_quorum),
            enable_perf_metrics=data.get('enable_perf_metrics', defaults.enable_perf_metrics),
            enable_trail=data.get('enable_trail', defaults.enable_trail),
            pipeline=data.get('pipeline', defaults.pipeline)
        )

    @classmethod
    def default_path(cls) -> Path:
        """Get default config file path (same directory as providers.py)."""
        return Path(__file__).parent / 'council.config.yaml'

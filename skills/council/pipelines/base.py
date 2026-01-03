"""
Base class for Council pipelines.

A Pipeline defines the high-level orchestration strategy for a deliberation session.
Different pipelines can implement different flows while sharing the same underlying
model adapters and mode implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import SessionConfig


@dataclass
class PipelineResult:
    """Result from a pipeline execution."""
    answer: str
    confidence: float
    pipeline: str  # 'classic' or 'storm'
    mode_used: str  # actual mode that ran (e.g., 'consensus', 'debate')
    rounds: int
    trail_file: Optional[str] = None
    # STORM-specific fields (None for classic pipeline)
    knowledge_base: Optional[Dict[str, Any]] = None
    evidence_coverage: Optional[float] = None
    unresolved_objections: Optional[List[str]] = None
    # Raw result from underlying mode (for backwards compatibility)
    raw_result: Optional[Dict[str, Any]] = None


class Pipeline(ABC):
    """
    Abstract base class for deliberation pipelines.

    Pipelines orchestrate the high-level flow of a deliberation session.
    They delegate to mode implementations for the actual deliberation logic.
    """

    name: str = "base"

    def __init__(self, config: SessionConfig):
        self.config = config

    @abstractmethod
    async def run(self) -> PipelineResult:
        """
        Execute the pipeline and return results.

        Returns:
            PipelineResult with answer, confidence, and metadata
        """
        pass

    @classmethod
    def supports_mode(cls, mode: str) -> bool:
        """Check if this pipeline supports the given mode."""
        return True  # Override in subclasses to restrict modes

    @classmethod
    def available_modes(cls) -> List[str]:
        """Return list of modes this pipeline supports."""
        return []  # Override in subclasses

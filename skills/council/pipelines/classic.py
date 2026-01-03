"""
Classic Pipeline - Original Council deliberation flow.

This pipeline preserves the original Council behavior:
1. Gather opinions from all models (parallel, with personas)
2. Multi-round deliberation with convergence detection
3. Peer review scoring
4. Chairman synthesis

Modes supported: adaptive, consensus, debate, vote, devil_advocate
"""

from typing import List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .base import Pipeline, PipelineResult
from core.models import SessionConfig

# Import existing mode implementations
from modes.consensus import run_council
from modes.vote import run_vote_council
from modes.adaptive import run_adaptive_cascade


class ClassicPipeline(Pipeline):
    """
    Original Council deliberation pipeline.

    Dispatches to existing mode implementations without modification.
    This ensures 100% backwards compatibility with the pre-STORM behavior.
    """

    name = "classic"

    # Modes supported by classic pipeline
    SUPPORTED_MODES = {'adaptive', 'consensus', 'debate', 'vote', 'devil_advocate'}

    def __init__(self, config: SessionConfig):
        super().__init__(config)

    @classmethod
    def supports_mode(cls, mode: str) -> bool:
        return mode in cls.SUPPORTED_MODES

    @classmethod
    def available_modes(cls) -> List[str]:
        return list(cls.SUPPORTED_MODES)

    async def run(self) -> PipelineResult:
        """
        Execute classic pipeline by dispatching to appropriate mode.

        Returns:
            PipelineResult wrapping the mode's output
        """
        mode = self.config.mode

        # Dispatch to appropriate mode implementation
        if mode == 'adaptive':
            raw_result = await run_adaptive_cascade(self.config)
        elif mode == 'vote':
            raw_result = await run_vote_council(self.config)
        else:
            # consensus, debate, devil_advocate all use run_council
            raw_result = await run_council(self.config)

        # Wrap result in PipelineResult for consistent interface
        return PipelineResult(
            answer=raw_result.get('answer', ''),
            confidence=raw_result.get('confidence', 0.0),
            pipeline='classic',
            mode_used=raw_result.get('mode', mode),
            rounds=raw_result.get('rounds', 1),
            trail_file=raw_result.get('trail_file'),
            # Classic pipeline doesn't produce STORM-specific fields
            knowledge_base=None,
            evidence_coverage=None,
            unresolved_objections=None,
            raw_result=raw_result
        )

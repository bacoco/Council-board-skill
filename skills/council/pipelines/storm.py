"""
STORM Pipeline - STORM/Co-STORM-inspired deliberation flow.

This pipeline implements the enhanced deliberation approach from the PRD:
1. Moderator selects workflow graph (decision/research/code-review)
2. Panelists generate opinions; KnowledgeBase tracks claims
3. Moderator detects shallow consensus → triggers Researcher
4. Evidence Judge scores claims against sources
5. Convergence based on evidence coverage + agreement
6. Chairman synthesizes with explicit evidence citations

Modes supported: storm_decision, storm_research, storm_review
(Also supports classic modes with STORM enhancements)

Status: STUB - Not yet implemented. Falls back to classic pipeline.
"""

from typing import List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .base import Pipeline, PipelineResult
from .classic import ClassicPipeline
from core.models import SessionConfig
from core.emit import emit


class StormPipeline(Pipeline):
    """
    STORM-inspired deliberation pipeline with Moderator and KnowledgeBase.

    NOTE: This is a stub implementation. Currently falls back to ClassicPipeline
    while emitting a notice. Full implementation is tracked in the PRD.

    New STORM-specific modes:
    - storm_decision: Decision workflow graph with rubric scoring and tripwires
    - storm_research: Research workflow with perspective generation and retrieval
    - storm_review: Code review workflow with threat modeling and patch suggestions

    Classic modes (adaptive, consensus, debate, vote, devil_advocate) are also
    supported with STORM enhancements (Moderator oversight, KB tracking).
    """

    name = "storm"

    # STORM-native modes (workflow graphs)
    STORM_MODES = {'storm_decision', 'storm_research', 'storm_review'}

    # Classic modes that can run with STORM enhancements
    ENHANCED_CLASSIC_MODES = {'adaptive', 'consensus', 'debate', 'vote', 'devil_advocate'}

    SUPPORTED_MODES = STORM_MODES | ENHANCED_CLASSIC_MODES

    def __init__(self, config: SessionConfig):
        super().__init__(config)
        # Will hold Moderator, KnowledgeBase, etc. when implemented
        self._moderator = None
        self._knowledge_base = None
        self._evidence_judge = None
        self._researcher = None

    @classmethod
    def supports_mode(cls, mode: str) -> bool:
        return mode in cls.SUPPORTED_MODES

    @classmethod
    def available_modes(cls) -> List[str]:
        return list(cls.SUPPORTED_MODES)

    async def run(self) -> PipelineResult:
        """
        Execute STORM pipeline.

        Currently: Falls back to classic pipeline with a notice.
        Future: Full Moderator-led flow with KnowledgeBase.

        Returns:
            PipelineResult with answer, confidence, and KB snapshot
        """
        mode = self.config.mode

        # Check if this is a STORM-native mode (not yet implemented)
        if mode in self.STORM_MODES:
            emit({
                'type': 'warning',
                'msg': f"STORM mode '{mode}' not yet implemented. Falling back to 'consensus' mode with classic pipeline."
            })
            # Override to consensus for now
            self.config.mode = 'consensus'

        # TODO: Implement STORM pipeline phases:
        # 1. Initialize KnowledgeBase
        # 2. Moderator selects workflow graph
        # 3. Run deliberation with KB updates per turn
        # 4. Moderator detects shallow consensus → trigger Researcher
        # 5. Evidence Judge scores claims
        # 6. Evidence-aware convergence check
        # 7. Chairman synthesizes with KB citations

        emit({
            'type': 'info',
            'msg': 'STORM pipeline not yet implemented. Using classic pipeline as fallback.'
        })

        # Fallback to classic pipeline
        classic = ClassicPipeline(self.config)
        classic_result = await classic.run()

        # Wrap with STORM metadata (empty for now)
        return PipelineResult(
            answer=classic_result.answer,
            confidence=classic_result.confidence,
            pipeline='storm',  # Mark as storm even though we fell back
            mode_used=classic_result.mode_used,
            rounds=classic_result.rounds,
            trail_file=classic_result.trail_file,
            # STORM fields - empty until implemented
            knowledge_base={
                '_status': 'not_implemented',
                'concepts': [],
                'claims': [],
                'sources': [],
                'open_questions': [],
                'decisions': []
            },
            evidence_coverage=None,
            unresolved_objections=None,
            raw_result=classic_result.raw_result
        )

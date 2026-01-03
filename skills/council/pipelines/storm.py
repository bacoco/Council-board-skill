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
"""

from typing import Any, Dict, List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .base import Pipeline, PipelineResult
from .classic import ClassicPipeline
from core.models import SessionConfig
from core.emit import emit

# STORM components
from knowledge_base import KnowledgeBase
from agents.moderator import Moderator, ModeratorAction, WorkflowType
from agents.researcher import Researcher
from agents.evidence_judge import EvidenceJudge
from convergence import ConvergenceDetector, ConvergenceResult


class StormPipeline(Pipeline):
    """
    STORM-inspired deliberation pipeline with Moderator and KnowledgeBase.

    This pipeline adds:
    - Moderator-led workflow selection and routing
    - Shared KnowledgeBase tracking claims and evidence
    - Shallow consensus detection with evidence requirements
    - Evidence-aware convergence scoring

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
        # Initialize STORM components
        self._knowledge_base = KnowledgeBase()
        self._moderator = Moderator(self._knowledge_base)
        self._researcher = Researcher(self._knowledge_base)
        self._evidence_judge = EvidenceJudge(self._knowledge_base)
        self._convergence = ConvergenceDetector(
            threshold=config.convergence_threshold,
            evidence_aware=True
        )
        # Track moderator decisions for trail
        self._routing_history: List[Dict[str, Any]] = []

    @classmethod
    def supports_mode(cls, mode: str) -> bool:
        return mode in cls.SUPPORTED_MODES

    @classmethod
    def available_modes(cls) -> List[str]:
        return list(cls.SUPPORTED_MODES)

    async def run(self) -> PipelineResult:
        """
        Execute STORM pipeline.

        Flow:
        1. Moderator detects workflow type from query
        2. Run classic deliberation (temporary - will be replaced with workflow graphs)
        3. Extract claims from results and populate KB
        4. Moderator analyzes round for shallow consensus
        5. If shallow, trigger researcher (stub for now)
        6. Evidence Judge evaluates claims
        7. Calculate evidence-aware convergence
        8. Return result with KB snapshot

        Returns:
            PipelineResult with answer, confidence, KB snapshot, and evidence metrics
        """
        mode = self.config.mode
        original_mode = mode

        # Step 1: Detect workflow type
        workflow = self._moderator.detect_workflow(
            self.config.query,
            self.config.context or ""
        )
        emit({
            'type': 'storm_workflow',
            'workflow': workflow.value,
            'query_preview': self.config.query[:100]
        })

        # Step 2: Handle STORM-native modes (not yet fully implemented)
        if mode in self.STORM_MODES:
            emit({
                'type': 'info',
                'msg': f"STORM mode '{mode}' using enhanced consensus flow. "
                       f"Full workflow graphs coming in Phase 2."
            })
            # Map to appropriate classic mode for now
            mode_map = {
                'storm_decision': 'consensus',
                'storm_research': 'consensus',
                'storm_review': 'devil_advocate'  # Red-team approach for reviews
            }
            self.config.mode = mode_map.get(mode, 'consensus')

        # Step 3: Run classic deliberation to get base result
        classic = ClassicPipeline(self.config)
        classic_result = await classic.run()

        # Step 4: Extract claims from result and populate KB
        self._extract_claims_from_result(classic_result.raw_result or {})

        # Step 5: Moderator analyzes for shallow consensus
        # Simulate round outputs from the classic result
        round_outputs = self._extract_round_outputs(classic_result.raw_result or {})
        moderator_decision = self._moderator.analyze_round(
            round_num=classic_result.rounds,
            round_outputs=round_outputs,
            max_rounds=self.config.max_rounds
        )

        self._routing_history.append(
            self._moderator.format_routing_event(moderator_decision, classic_result.rounds)
        )

        emit({
            'type': 'moderator_decision',
            'action': moderator_decision.action.value,
            'reason': moderator_decision.reason,
            'shallow_consensus': moderator_decision.shallow_consensus_detected
        })

        # Step 6: If shallow consensus or low coverage, note it (retrieval is stub)
        if moderator_decision.action == ModeratorAction.RETRIEVE:
            emit({
                'type': 'info',
                'msg': f"Moderator detected need for evidence retrieval. "
                       f"{len(moderator_decision.claims_needing_evidence)} claims need sources. "
                       f"(Retrieval not yet implemented)"
            })
            # In future: await self._researcher.retrieve_for_claims(...)

        # Step 7: Evidence Judge evaluates claims
        evidence_report = await self._evidence_judge.evaluate_all_claims()

        # Step 8: Calculate evidence-aware convergence
        convergence = self._convergence.check_evidence_aware(
            round_outputs=round_outputs,
            kb=self._knowledge_base
        )

        # Adjust final confidence based on evidence
        adjusted_confidence = self._evidence_judge.calculate_evidence_adjusted_confidence(
            classic_result.confidence
        )

        # Build enhanced answer with evidence rationale
        enhanced_answer = self._enhance_answer_with_evidence(
            classic_result.answer,
            convergence,
            evidence_report
        )

        # Build unresolved objections list
        unresolved = [
            self._knowledge_base.get_claim(cid).text
            for cid in evidence_report.contradicted_claims
            if self._knowledge_base.get_claim(cid)
        ]

        return PipelineResult(
            answer=enhanced_answer,
            confidence=adjusted_confidence,
            pipeline='storm',
            mode_used=original_mode,
            rounds=classic_result.rounds,
            trail_file=classic_result.trail_file,
            knowledge_base=self._knowledge_base.to_dict(),
            evidence_coverage=self._knowledge_base.evidence_coverage(),
            unresolved_objections=unresolved if unresolved else None,
            raw_result={
                **(classic_result.raw_result or {}),
                'storm_metadata': {
                    'workflow': workflow.value,
                    'moderator_history': self._routing_history,
                    'convergence': convergence.to_dict(),
                    'evidence_report': evidence_report.to_dict()
                }
            }
        )

    def _extract_claims_from_result(self, result: Dict[str, Any]) -> None:
        """
        Extract claims from deliberation result and add to KB.

        Parses the trail/contributions to identify assertions made by panelists.
        """
        # Extract from trail if available
        trail = result.get('trail', [])
        for entry in trail:
            if isinstance(entry, dict):
                persona = entry.get('persona', 'unknown')
                content = entry.get('content', '') or entry.get('answer', '')

                # Simple heuristic: treat each sentence-like segment as a claim
                # In production, this would use NLP or LLM extraction
                if content and len(content) > 20:
                    # Add the main response as a claim
                    confidence = entry.get('confidence', 0.5)
                    if isinstance(confidence, (int, float)):
                        self._knowledge_base.add_claim(
                            text=content[:500],  # Truncate long content
                            owner=persona,
                            confidence=confidence,
                            round_num=entry.get('round', 1)
                        )

        # If no trail, use the final answer as a claim
        if not trail and result.get('answer'):
            self._knowledge_base.add_claim(
                text=result['answer'][:500],
                owner='synthesis',
                confidence=result.get('confidence', 0.5)
            )

    def _extract_round_outputs(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract round outputs for convergence calculation.
        """
        outputs = []
        trail = result.get('trail', [])

        for entry in trail:
            if isinstance(entry, dict):
                outputs.append({
                    'confidence': entry.get('confidence', 0.5),
                    'converged': entry.get('converged', False),
                    'agrees_with_consensus': entry.get('agrees', False)
                })

        # Fallback if no trail
        if not outputs:
            outputs.append({
                'confidence': result.get('confidence', 0.5),
                'converged': True
            })

        return outputs

    def _enhance_answer_with_evidence(self, answer: str,
                                       convergence: ConvergenceResult,
                                       evidence_report) -> str:
        """
        Append evidence rationale to the answer.
        """
        # Add confidence rationale
        rationale = f"\n\n---\n**Confidence Rationale:** {convergence.confidence_rationale}"

        # Add evidence notes if any
        if evidence_report.evaluation_notes:
            rationale += f"\n**Evidence Notes:** {'; '.join(evidence_report.evaluation_notes)}"

        # Note unsupported claims if significant
        if len(evidence_report.unsupported_claims) > 0:
            rationale += f"\n**Note:** {len(evidence_report.unsupported_claims)} claim(s) await evidence verification."

        return answer + rationale

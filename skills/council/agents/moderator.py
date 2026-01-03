"""
Moderator Agent - Orchestrates STORM deliberation flow.

The Moderator:
1. Selects the appropriate workflow graph based on query type
2. Maintains the open_questions queue
3. Detects shallow consensus (agreement without evidence)
4. Routes to retrieval/verification when needed
5. Decides when to finalize or continue deliberation
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import re

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_base import KnowledgeBase, ClaimStatus


class WorkflowType(Enum):
    """Workflow graphs available for different query types."""
    DECISION = "decision"      # Options → Rubric → Red-team → Recommend
    RESEARCH = "research"      # Perspectives → Questions → Retrieve → Report
    CODE_REVIEW = "code_review"  # Static scan → Threat model → Patches → Checklist


class ModeratorAction(Enum):
    """Actions the Moderator can take."""
    CONTINUE = "continue"      # Proceed to next deliberation round
    RETRIEVE = "retrieve"      # Trigger Researcher for evidence gathering
    VERIFY = "verify"          # Trigger Evidence Judge for claim verification
    DEBATE = "debate"          # Escalate to structured debate
    FINALIZE = "finalize"      # Ready for synthesis


@dataclass
class ModeratorDecision:
    """Decision made by the Moderator after analyzing round state."""
    action: ModeratorAction
    reason: str
    workflow: WorkflowType
    # Claims/questions that need attention
    claims_needing_evidence: List[str]
    open_questions: List[str]
    # Shallow consensus detection
    shallow_consensus_detected: bool
    # Metrics at decision time
    evidence_coverage: float
    agreement_level: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'action': self.action.value,
            'reason': self.reason,
            'workflow': self.workflow.value,
            'claims_needing_evidence': self.claims_needing_evidence,
            'open_questions': self.open_questions,
            'shallow_consensus_detected': self.shallow_consensus_detected,
            'evidence_coverage': round(self.evidence_coverage, 3),
            'agreement_level': round(self.agreement_level, 3)
        }


class Moderator:
    """
    Orchestrates the STORM deliberation flow.

    The Moderator acts as a meta-agent that doesn't participate in the
    deliberation itself but manages the process flow.
    """

    # Keywords for workflow detection
    DECISION_KEYWORDS = [
        'should', 'choose', 'decide', 'recommend', 'best', 'compare',
        'vs', 'versus', 'or', 'tradeoff', 'option', 'alternative'
    ]
    RESEARCH_KEYWORDS = [
        'what is', 'how does', 'explain', 'why', 'when', 'where',
        'history', 'overview', 'summary', 'understand'
    ]
    CODE_REVIEW_KEYWORDS = [
        'review', 'code', 'security', 'vulnerability', 'bug', 'fix',
        'refactor', 'improve', 'audit', 'scan'
    ]

    # Thresholds
    SHALLOW_CONSENSUS_THRESHOLD = 0.7  # High agreement with low evidence
    MIN_EVIDENCE_COVERAGE = 0.3        # Below this triggers retrieval
    FINALIZE_EVIDENCE_THRESHOLD = 0.6  # Need at least this to finalize

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self._detected_workflow: Optional[WorkflowType] = None

    def detect_workflow(self, query: str, context: str = "") -> WorkflowType:
        """
        Detect the appropriate workflow based on query content.

        Args:
            query: The user's question
            context: Optional context (code, docs, etc.)

        Returns:
            WorkflowType indicating the recommended workflow
        """
        query_lower = query.lower()
        context_lower = context.lower() if context else ""
        combined = query_lower + " " + context_lower

        # Check for code review indicators (highest priority if context has code)
        if context and any(kw in combined for kw in self.CODE_REVIEW_KEYWORDS):
            # Additional heuristics for code
            code_indicators = ['def ', 'class ', 'function', 'import ', '```', '{', '}']
            if any(ind in context for ind in code_indicators):
                self._detected_workflow = WorkflowType.CODE_REVIEW
                return WorkflowType.CODE_REVIEW

        # Check for decision indicators
        decision_score = sum(1 for kw in self.DECISION_KEYWORDS if kw in query_lower)
        research_score = sum(1 for kw in self.RESEARCH_KEYWORDS if kw in query_lower)

        if decision_score > research_score:
            self._detected_workflow = WorkflowType.DECISION
            return WorkflowType.DECISION
        elif research_score > 0:
            self._detected_workflow = WorkflowType.RESEARCH
            return WorkflowType.RESEARCH
        else:
            # Default to decision for ambiguous queries
            self._detected_workflow = WorkflowType.DECISION
            return WorkflowType.DECISION

    def calculate_agreement_level(self, round_outputs: List[Dict[str, Any]]) -> float:
        """
        Calculate how much the panelists agree.

        Args:
            round_outputs: List of outputs from panelists with 'confidence' and 'position'

        Returns:
            Float 0.0-1.0 indicating agreement level
        """
        if not round_outputs:
            return 0.0

        # Extract confidence scores
        confidences = [
            o.get('confidence', 0.5)
            for o in round_outputs
            if isinstance(o.get('confidence'), (int, float))
        ]

        if not confidences:
            return 0.5

        avg_confidence = sum(confidences) / len(confidences)

        # Check for explicit convergence signals
        converged_count = sum(
            1 for o in round_outputs
            if o.get('converged', False) or o.get('agrees_with_consensus', False)
        )
        convergence_ratio = converged_count / len(round_outputs) if round_outputs else 0

        # Blend confidence and convergence signals
        return 0.6 * avg_confidence + 0.4 * convergence_ratio

    def detect_shallow_consensus(self, round_outputs: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """
        Detect if panelists agree but without evidence.

        Shallow consensus = high agreement + low evidence coverage.

        Returns:
            (is_shallow, reason)
        """
        agreement = self.calculate_agreement_level(round_outputs)
        coverage = self.kb.evidence_coverage()

        is_shallow = (
            agreement >= self.SHALLOW_CONSENSUS_THRESHOLD and
            coverage < self.MIN_EVIDENCE_COVERAGE
        )

        if is_shallow:
            reason = (
                f"High agreement ({agreement:.0%}) but low evidence coverage ({coverage:.0%}). "
                f"Claims need verification before finalizing."
            )
        else:
            reason = ""

        return is_shallow, reason

    def analyze_round(self, round_num: int,
                      round_outputs: List[Dict[str, Any]],
                      max_rounds: int = 3) -> ModeratorDecision:
        """
        Analyze the current round and decide next action.

        Args:
            round_num: Current round number
            round_outputs: Outputs from all panelists this round
            max_rounds: Maximum allowed rounds

        Returns:
            ModeratorDecision indicating what to do next
        """
        workflow = self._detected_workflow or WorkflowType.DECISION
        coverage = self.kb.evidence_coverage()
        agreement = self.calculate_agreement_level(round_outputs)
        is_shallow, shallow_reason = self.detect_shallow_consensus(round_outputs)

        # Get claims needing evidence
        unsupported = self.kb.get_unsupported_claims()
        claims_needing_evidence = [c.id for c in unsupported]

        # Get open questions
        open_qs = self.kb.get_open_questions()
        open_question_ids = [q.id for q in open_qs]

        # Decision logic
        if round_num >= max_rounds:
            # Force finalize at max rounds
            return ModeratorDecision(
                action=ModeratorAction.FINALIZE,
                reason=f"Reached maximum rounds ({max_rounds}). Finalizing with current state.",
                workflow=workflow,
                claims_needing_evidence=claims_needing_evidence,
                open_questions=open_question_ids,
                shallow_consensus_detected=is_shallow,
                evidence_coverage=coverage,
                agreement_level=agreement
            )

        if is_shallow:
            # Shallow consensus detected - trigger retrieval
            return ModeratorDecision(
                action=ModeratorAction.RETRIEVE,
                reason=shallow_reason,
                workflow=workflow,
                claims_needing_evidence=claims_needing_evidence,
                open_questions=open_question_ids,
                shallow_consensus_detected=True,
                evidence_coverage=coverage,
                agreement_level=agreement
            )

        if coverage < self.MIN_EVIDENCE_COVERAGE and unsupported:
            # Low evidence coverage - trigger retrieval
            return ModeratorDecision(
                action=ModeratorAction.RETRIEVE,
                reason=f"Evidence coverage ({coverage:.0%}) below threshold. "
                       f"{len(unsupported)} claims need supporting evidence.",
                workflow=workflow,
                claims_needing_evidence=claims_needing_evidence,
                open_questions=open_question_ids,
                shallow_consensus_detected=False,
                evidence_coverage=coverage,
                agreement_level=agreement
            )

        if self.kb.unresolved_objections_count() > 0:
            # Contradicted claims exist - need verification or debate
            return ModeratorDecision(
                action=ModeratorAction.VERIFY,
                reason=f"{self.kb.unresolved_objections_count()} contradicted claims "
                       f"need resolution.",
                workflow=workflow,
                claims_needing_evidence=claims_needing_evidence,
                open_questions=open_question_ids,
                shallow_consensus_detected=False,
                evidence_coverage=coverage,
                agreement_level=agreement
            )

        if agreement >= 0.8 and coverage >= self.FINALIZE_EVIDENCE_THRESHOLD:
            # Good agreement and evidence - ready to finalize
            return ModeratorDecision(
                action=ModeratorAction.FINALIZE,
                reason=f"Strong agreement ({agreement:.0%}) with adequate evidence "
                       f"coverage ({coverage:.0%}). Ready for synthesis.",
                workflow=workflow,
                claims_needing_evidence=claims_needing_evidence,
                open_questions=open_question_ids,
                shallow_consensus_detected=False,
                evidence_coverage=coverage,
                agreement_level=agreement
            )

        # Default: continue deliberation
        return ModeratorDecision(
            action=ModeratorAction.CONTINUE,
            reason=f"Agreement ({agreement:.0%}) and coverage ({coverage:.0%}) "
                   f"not yet sufficient. Continuing deliberation.",
            workflow=workflow,
            claims_needing_evidence=claims_needing_evidence,
            open_questions=open_question_ids,
            shallow_consensus_detected=False,
            evidence_coverage=coverage,
            agreement_level=agreement
        )

    def format_routing_event(self, decision: ModeratorDecision,
                             round_num: int) -> Dict[str, Any]:
        """Format decision for trail/logging."""
        return {
            'type': 'moderator_routing',
            'round': round_num,
            'decision': decision.to_dict()
        }

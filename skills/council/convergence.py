"""
Convergence Detection - Evidence-aware convergence for STORM pipeline.

Classic convergence: agreement (confidence) + explicit signals
STORM convergence: agreement + evidence coverage + objections + diversity

This module provides both classic and evidence-aware convergence detection.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from knowledge_base import KnowledgeBase


@dataclass
class ConvergenceResult:
    """Result of convergence check."""
    converged: bool
    score: float                    # 0.0-1.0 convergence score
    threshold: float                # Threshold used
    components: Dict[str, float]    # Breakdown by component
    reason: str                     # Human-readable explanation
    confidence_rationale: str       # Why this confidence level

    def to_dict(self) -> Dict[str, Any]:
        return {
            'converged': self.converged,
            'score': round(self.score, 3),
            'threshold': self.threshold,
            'components': {k: round(v, 3) for k, v in self.components.items()},
            'reason': self.reason,
            'confidence_rationale': self.confidence_rationale
        }


class ConvergenceDetector:
    """
    Detects convergence in deliberation rounds.

    Supports two modes:
    - Classic: Based on confidence scores and explicit convergence signals
    - Evidence-aware: Adds evidence coverage, objections, and source diversity
    """

    # Default weights for evidence-aware convergence
    DEFAULT_WEIGHTS = {
        'agreement': 0.40,      # Panelist agreement/confidence
        'evidence': 0.30,       # Evidence coverage
        'objections': 0.20,     # Inverse of unresolved objections
        'diversity': 0.10       # Source diversity
    }

    def __init__(self, threshold: float = 0.8,
                 weights: Dict[str, float] = None,
                 evidence_aware: bool = True):
        """
        Initialize convergence detector.

        Args:
            threshold: Convergence threshold (0.0-1.0)
            weights: Custom weights for components (must sum to 1.0)
            evidence_aware: Whether to use evidence-aware mode
        """
        self.threshold = threshold
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.evidence_aware = evidence_aware

        # Validate weights sum to 1.0
        if evidence_aware:
            weight_sum = sum(self.weights.values())
            if abs(weight_sum - 1.0) > 0.01:
                raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")

    def check_classic(self, round_outputs: List[Dict[str, Any]]) -> ConvergenceResult:
        """
        Classic convergence check (pre-STORM behavior).

        Uses 60% confidence + 40% explicit signals.
        """
        if not round_outputs:
            return ConvergenceResult(
                converged=False,
                score=0.0,
                threshold=self.threshold,
                components={'agreement': 0.0, 'signals': 0.0},
                reason="No outputs to evaluate",
                confidence_rationale="No panelist responses received"
            )

        # Calculate average confidence
        confidences = [
            o.get('confidence', 0.5)
            for o in round_outputs
            if isinstance(o.get('confidence'), (int, float))
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5

        # Count explicit convergence signals
        converged_count = sum(
            1 for o in round_outputs
            if o.get('converged', False) or o.get('agrees_with_consensus', False)
        )
        signal_ratio = converged_count / len(round_outputs)

        # Classic formula: 60% confidence + 40% signals
        score = 0.6 * avg_confidence + 0.4 * signal_ratio
        converged = score >= self.threshold

        return ConvergenceResult(
            converged=converged,
            score=score,
            threshold=self.threshold,
            components={
                'agreement': avg_confidence,
                'signals': signal_ratio
            },
            reason=f"Agreement: {avg_confidence:.0%}, Signals: {signal_ratio:.0%}",
            confidence_rationale=self._classic_rationale(avg_confidence, signal_ratio, converged)
        )

    def check_evidence_aware(self, round_outputs: List[Dict[str, Any]],
                              kb: KnowledgeBase) -> ConvergenceResult:
        """
        Evidence-aware convergence check (STORM mode).

        Blends agreement, evidence coverage, objections, and diversity.
        """
        if not round_outputs:
            return ConvergenceResult(
                converged=False,
                score=0.0,
                threshold=self.threshold,
                components={k: 0.0 for k in self.weights.keys()},
                reason="No outputs to evaluate",
                confidence_rationale="No panelist responses received"
            )

        # Agreement component (same as classic)
        confidences = [
            o.get('confidence', 0.5)
            for o in round_outputs
            if isinstance(o.get('confidence'), (int, float))
        ]
        agreement = sum(confidences) / len(confidences) if confidences else 0.5

        # Evidence component
        evidence = kb.evidence_coverage()

        # Objections component (inverse - fewer objections = higher score)
        objection_count = kb.unresolved_objections_count()
        # Cap at 5 objections for scoring purposes
        objections_score = max(0.0, 1.0 - (objection_count / 5.0))

        # Diversity component
        diversity = kb.source_diversity()

        # Calculate weighted score
        components = {
            'agreement': agreement,
            'evidence': evidence,
            'objections': objections_score,
            'diversity': diversity
        }

        score = sum(
            self.weights.get(k, 0) * v
            for k, v in components.items()
        )

        converged = score >= self.threshold

        return ConvergenceResult(
            converged=converged,
            score=score,
            threshold=self.threshold,
            components=components,
            reason=self._evidence_reason(components),
            confidence_rationale=self._evidence_rationale(components, kb, converged)
        )

    def check(self, round_outputs: List[Dict[str, Any]],
              kb: Optional[KnowledgeBase] = None) -> ConvergenceResult:
        """
        Check convergence using appropriate mode.

        Args:
            round_outputs: Outputs from current round
            kb: KnowledgeBase (required for evidence-aware mode)

        Returns:
            ConvergenceResult with convergence decision
        """
        if self.evidence_aware and kb is not None:
            return self.check_evidence_aware(round_outputs, kb)
        else:
            return self.check_classic(round_outputs)

    def _classic_rationale(self, agreement: float, signals: float,
                           converged: bool) -> str:
        """Generate rationale for classic convergence."""
        if converged:
            return (f"Converged with {agreement:.0%} agreement and "
                    f"{signals:.0%} explicit consensus signals.")
        else:
            issues = []
            if agreement < 0.7:
                issues.append(f"low agreement ({agreement:.0%})")
            if signals < 0.5:
                issues.append(f"few consensus signals ({signals:.0%})")
            return f"Not converged due to {' and '.join(issues)}."

    def _evidence_reason(self, components: Dict[str, float]) -> str:
        """Generate short reason string for evidence-aware check."""
        parts = [
            f"Agreement: {components['agreement']:.0%}",
            f"Evidence: {components['evidence']:.0%}",
            f"Objections: {components['objections']:.0%}",
            f"Diversity: {components['diversity']:.0%}"
        ]
        return ", ".join(parts)

    def _evidence_rationale(self, components: Dict[str, float],
                            kb: KnowledgeBase, converged: bool) -> str:
        """Generate detailed rationale for evidence-aware convergence."""
        unsupported = len(kb.get_unsupported_claims())
        contradicted = kb.unresolved_objections_count()
        open_qs = len(kb.get_open_questions())

        if converged:
            rationale = f"Confidence {components['agreement']:.0%}"
            notes = []
            if unsupported == 0:
                notes.append("all claims have evidence")
            if contradicted == 0:
                notes.append("no unresolved objections")
            if components['diversity'] > 0.5:
                notes.append("diverse sources")
            if notes:
                rationale += f" because {', '.join(notes)}."
            else:
                rationale += "."
            return rationale
        else:
            issues = []
            if unsupported > 0:
                issues.append(f"{unsupported} claims lack evidence")
            if contradicted > 0:
                issues.append(f"{contradicted} objections unresolved")
            if open_qs > 0:
                issues.append(f"{open_qs} questions open")
            if components['agreement'] < 0.6:
                issues.append(f"low agreement ({components['agreement']:.0%})")

            return f"Not ready: {'; '.join(issues) if issues else 'threshold not met'}."


# Convenience function for quick checks
def check_convergence(round_outputs: List[Dict[str, Any]],
                      kb: Optional[KnowledgeBase] = None,
                      threshold: float = 0.8,
                      evidence_aware: bool = True) -> ConvergenceResult:
    """
    Quick convergence check with default settings.

    Args:
        round_outputs: Panelist outputs from current round
        kb: KnowledgeBase (enables evidence-aware mode)
        threshold: Convergence threshold
        evidence_aware: Use evidence-aware mode if kb provided

    Returns:
        ConvergenceResult
    """
    detector = ConvergenceDetector(
        threshold=threshold,
        evidence_aware=evidence_aware and kb is not None
    )
    return detector.check(round_outputs, kb)

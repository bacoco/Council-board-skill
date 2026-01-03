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
        Classic convergence check with semantic validation.

        Security: Does NOT blindly trust model-reported confidence scores.
        Validates claims against actual semantic agreement between responses.
        Prevents gaming via spoofed confidence/convergence_signal.

        Uses:
        - 40% validated confidence (penalized if anomalous)
        - 40% semantic agreement (computed, not claimed)
        - 20% convergence signals (only if agreement > 30%)
        """
        if not round_outputs:
            return ConvergenceResult(
                converged=False,
                score=0.0,
                threshold=self.threshold,
                components={'agreement': 0.0, 'signals': 0.0, 'semantic': 0.0},
                reason="No outputs to evaluate",
                confidence_rationale="No panelist responses received"
            )

        # Extract raw claimed values
        raw_confidences = [
            o.get('confidence', 0.5)
            for o in round_outputs
            if isinstance(o.get('confidence'), (int, float))
        ]
        avg_raw_confidence = sum(raw_confidences) / len(raw_confidences) if raw_confidences else 0.5

        # Count explicit convergence signals
        converged_count = sum(
            1 for o in round_outputs
            if o.get('converged', False) or o.get('agrees_with_consensus', False)
        )
        signal_ratio = converged_count / len(round_outputs)

        # Perform semantic validation
        semantic_result = self._validate_semantic_agreement(round_outputs)
        semantic_agreement = semantic_result['agreement']
        validated_confidence = semantic_result['validated_confidence']
        anomaly_count = semantic_result['anomaly_count']

        # Calculate raw score (for comparison)
        raw_score = 0.6 * avg_raw_confidence + 0.4 * signal_ratio

        # Calculate validated score with semantic agreement
        # Only count signals if there's actual agreement
        signal_weight = 0.2 if semantic_agreement > 0.3 else 0.0
        conf_weight = 0.4 + (0.2 - signal_weight)

        validated_score = (
            conf_weight * validated_confidence +
            0.4 * semantic_agreement +
            signal_weight * signal_ratio
        )

        # Apply anomaly penalty (-15% per anomaly)
        anomaly_penalty = 1.0 - (anomaly_count * 0.15)
        validated_score *= max(0.5, anomaly_penalty)

        # Security: require validation for convergence
        is_validated = (
            anomaly_count <= len(round_outputs) / 2 and
            raw_score <= validated_score + 0.2
        )

        converged = validated_score >= self.threshold and is_validated

        return ConvergenceResult(
            converged=converged,
            score=validated_score,
            threshold=self.threshold,
            components={
                'agreement': validated_confidence,
                'signals': signal_ratio,
                'semantic': semantic_agreement,
                'raw_confidence': avg_raw_confidence,
                'anomaly_count': anomaly_count
            },
            reason=f"Agreement: {validated_confidence:.0%}, Semantic: {semantic_agreement:.0%}, Signals: {signal_ratio:.0%}",
            confidence_rationale=self._classic_rationale_secure(
                validated_confidence, semantic_agreement, signal_ratio,
                anomaly_count, is_validated, converged
            )
        )

    def _validate_semantic_agreement(self, round_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate confidence claims against actual semantic agreement.

        Returns dict with:
        - agreement: actual semantic agreement (0.0-1.0)
        - validated_confidence: adjusted confidence after validation
        - anomaly_count: number of anomalous responses
        """
        import re
        import math

        if len(round_outputs) < 2:
            conf = round_outputs[0].get('confidence', 0.5) if round_outputs else 0.5
            return {
                'agreement': 1.0,
                'validated_confidence': conf,
                'anomaly_count': 0
            }

        # Extract response texts and confidences
        responses = []
        confidences = []
        for o in round_outputs:
            text = o.get('response', '') or o.get('text', '') or o.get('argument', '') or str(o)
            conf = o.get('confidence', 0.5)
            if isinstance(conf, (int, float)):
                conf = max(0.0, min(1.0, float(conf)))
            else:
                conf = 0.5
            responses.append(str(text))
            confidences.append(conf)

        # Compute semantic agreement using key phrase overlap
        def extract_key_phrases(text):
            text = text.lower()
            text = re.sub(r'\*\*|__|```|\n+', ' ', text)
            text = re.sub(r'[^\w\s]', ' ', text)
            words = text.split()
            stopwords = {
                'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
                'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
                'and', 'but', 'if', 'or', 'because', 'this', 'that', 'these', 'those',
                'i', 'you', 'he', 'she', 'it', 'we', 'they', 'not', 'so', 'very'
            }
            meaningful = [w for w in words if w not in stopwords and len(w) > 2]
            phrases = set(meaningful)
            for i in range(len(meaningful) - 1):
                phrases.add(f"{meaningful[i]} {meaningful[i+1]}")
            return phrases

        tokenized = [extract_key_phrases(r) for r in responses]

        # Compute pairwise Jaccard similarities
        n = len(responses)
        similarities = []
        for i in range(n):
            for j in range(i + 1, n):
                if tokenized[i] and tokenized[j]:
                    intersection = len(tokenized[i] & tokenized[j])
                    union = len(tokenized[i] | tokenized[j])
                    sim = intersection / union if union > 0 else 0.0
                    similarities.append(sim)

        semantic_agreement = sum(similarities) / len(similarities) if similarities else 0.0

        # Detect confidence anomalies
        mean_conf = sum(confidences) / len(confidences)
        variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)
        stddev = math.sqrt(variance) if variance > 0 else 0.0

        anomaly_count = 0
        validated_confidences = []
        for i, conf in enumerate(confidences):
            penalty = 1.0

            # Check for statistical outlier (high confidence)
            if stddev > 0:
                z_score = (conf - mean_conf) / stddev
                if conf > mean_conf and z_score > 1.5:
                    penalty *= 0.8
                    anomaly_count += 1

            # Check for high confidence with low agreement
            if conf > 0.8 and semantic_agreement < 0.5:
                penalty *= 0.5
                if penalty < 0.8:  # Only count once
                    pass
                else:
                    anomaly_count += 1

            # Check response length (short responses are suspicious)
            if len(responses[i]) < 100:
                penalty *= 0.7

            validated_confidences.append(conf * penalty)

        avg_validated = sum(validated_confidences) / len(validated_confidences)

        return {
            'agreement': semantic_agreement,
            'validated_confidence': avg_validated,
            'anomaly_count': anomaly_count
        }

    def _classic_rationale_secure(
        self,
        confidence: float,
        semantic: float,
        signals: float,
        anomalies: int,
        validated: bool,
        converged: bool
    ) -> str:
        """Generate rationale for secure classic convergence."""
        if converged:
            return (
                f"Converged with {confidence:.0%} validated confidence, "
                f"{semantic:.0%} semantic agreement, and {signals:.0%} explicit signals."
            )
        else:
            issues = []
            if not validated:
                issues.append("validation failed")
            if confidence < 0.7:
                issues.append(f"low confidence ({confidence:.0%})")
            if semantic < 0.5:
                issues.append(f"low semantic agreement ({semantic:.0%})")
            if signals < 0.5:
                issues.append(f"few consensus signals ({signals:.0%})")
            if anomalies > 0:
                issues.append(f"{anomalies} anomalous response(s)")
            return f"Not converged due to {' and '.join(issues)}."

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

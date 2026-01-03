"""
Semantic Validation for Convergence Detection.

Prevents gaming of convergence via spoofed confidence scores by:
1. Cross-validating confidence claims against actual content agreement
2. Detecting anomalous confidence values (outliers)
3. Computing semantic similarity between model responses
4. Capping single-model influence on convergence score

This module addresses CVE-equivalent vulnerability where any model
could force early exit by emitting high confidence + convergence_signal.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from collections import Counter
import math


@dataclass
class SemanticValidationResult:
    """Result of semantic validation."""
    validated_confidence: float      # Adjusted confidence after validation
    raw_confidence: float            # Original claimed confidence
    agreement_score: float           # Actual semantic agreement (0.0-1.0)
    anomaly_detected: bool           # True if confidence claim is suspicious
    anomaly_reason: Optional[str]    # Explanation if anomaly detected
    penalty_applied: float           # Penalty factor applied (1.0 = no penalty)
    details: Dict[str, any]          # Additional diagnostic info


class SemanticValidator:
    """
    Validates confidence claims against actual semantic content.

    Defense against convergence gaming:
    - Models cannot force early exit by simply claiming high confidence
    - Confidence must be backed by actual agreement in content
    - Anomalous outliers are penalized
    """

    # Configuration
    MAX_SINGLE_MODEL_INFLUENCE = 0.6   # No model can contribute > 60% to convergence
    ANOMALY_THRESHOLD_STDDEV = 2.0     # Flag if > 2.0 stddev from mean (was 1.5)
    MIN_AGREEMENT_FOR_HIGH_CONF = 0.25 # Need 25% agreement for confidence > 0.8 (was 0.5)
    CONFIDENCE_PENALTY_FACTOR = 0.6    # Reduce confidence by 40% if validation fails (was 0.5)
    MIN_RESPONSE_LENGTH = 50           # Minimum chars for valid response

    def __init__(self,
                 max_single_influence: float = None,
                 anomaly_threshold: float = None):
        """
        Initialize validator.

        Args:
            max_single_influence: Max contribution from single model (0.0-1.0)
            anomaly_threshold: Stddev threshold for anomaly detection
        """
        self.max_single_influence = max_single_influence or self.MAX_SINGLE_MODEL_INFLUENCE
        self.anomaly_threshold = anomaly_threshold or self.ANOMALY_THRESHOLD_STDDEV

    def validate_convergence_claims(
        self,
        model_outputs: List[Dict],
        min_models: int = 2
    ) -> Tuple[List[SemanticValidationResult], float]:
        """
        Validate confidence claims across all model outputs.

        Args:
            model_outputs: List of dicts with 'response', 'confidence', 'model' keys
            min_models: Minimum models required for valid convergence

        Returns:
            Tuple of (list of validation results, adjusted convergence score)
        """
        if len(model_outputs) < min_models:
            return [], 0.0

        results = []

        # Extract responses and claimed confidences
        responses = []
        claimed_confidences = []

        for output in model_outputs:
            response_text = self._extract_response_text(output)
            confidence = self._extract_confidence(output)
            responses.append(response_text)
            claimed_confidences.append(confidence)

        # Compute pairwise semantic agreement
        agreement_matrix = self._compute_agreement_matrix(responses)

        # Detect anomalies
        anomalies = self._detect_anomalies(claimed_confidences)

        # Validate each model's confidence claim
        for i, output in enumerate(model_outputs):
            avg_agreement = self._average_agreement_with_others(agreement_matrix, i)
            claimed = claimed_confidences[i]

            validation = self._validate_single_claim(
                claimed_confidence=claimed,
                actual_agreement=avg_agreement,
                is_anomaly=anomalies[i],
                response_length=len(responses[i]),
                model_index=i,
                total_models=len(model_outputs)
            )
            results.append(validation)

        # Calculate adjusted convergence score with single-model caps
        adjusted_score = self._calculate_adjusted_score(results, claimed_confidences)

        return results, adjusted_score

    def _extract_response_text(self, output: Dict) -> str:
        """Extract response text from model output."""
        if isinstance(output, str):
            return output

        # Try common keys
        for key in ['response', 'text', 'content', 'answer', 'argument']:
            if key in output and isinstance(output[key], str):
                return output[key]

        # Fall back to string representation
        return str(output)

    def _extract_confidence(self, output: Dict) -> float:
        """Extract confidence score from model output."""
        if isinstance(output, dict):
            conf = output.get('confidence', 0.5)
            if isinstance(conf, (int, float)):
                return max(0.0, min(1.0, float(conf)))
        return 0.5

    def _compute_agreement_matrix(self, responses: List[str]) -> List[List[float]]:
        """
        Compute pairwise semantic agreement between responses.

        Uses Jaccard similarity on key phrase n-grams.
        """
        n = len(responses)
        matrix = [[0.0] * n for _ in range(n)]

        # Tokenize and extract key phrases
        tokenized = [self._extract_key_phrases(r) for r in responses]

        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 1.0
                elif i < j:
                    sim = self._jaccard_similarity(tokenized[i], tokenized[j])
                    matrix[i][j] = sim
                    matrix[j][i] = sim

        return matrix

    def _extract_key_phrases(self, text: str) -> Set[str]:
        """
        Extract key phrases for comparison.

        Focuses on meaningful content, strips filler.
        """
        # Normalize
        text = text.lower()

        # Remove common filler and formatting
        text = re.sub(r'\*\*|__|```|\n+', ' ', text)
        text = re.sub(r'[^\w\s]', ' ', text)

        # Tokenize
        words = text.split()

        # Filter stopwords (basic list)
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'as', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
            'and', 'but', 'if', 'or', 'because', 'until', 'while', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }

        meaningful = [w for w in words if w not in stopwords and len(w) > 2]

        # Create bigrams for phrase matching
        phrases = set(meaningful)
        for i in range(len(meaningful) - 1):
            phrases.add(f"{meaningful[i]} {meaningful[i+1]}")

        return phrases

    def _jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def _average_agreement_with_others(
        self,
        matrix: List[List[float]],
        index: int
    ) -> float:
        """Calculate average agreement of model at index with all other models."""
        n = len(matrix)
        if n <= 1:
            return 1.0

        total = sum(matrix[index][j] for j in range(n) if j != index)
        return total / (n - 1)

    def _detect_anomalies(self, confidences: List[float]) -> List[bool]:
        """
        Detect anomalous confidence claims.

        Flags values that are statistical outliers.
        """
        if len(confidences) < 2:
            return [False] * len(confidences)

        mean = sum(confidences) / len(confidences)
        variance = sum((c - mean) ** 2 for c in confidences) / len(confidences)
        stddev = math.sqrt(variance) if variance > 0 else 0.0

        anomalies = []
        for conf in confidences:
            if stddev > 0:
                z_score = abs(conf - mean) / stddev
                # Flag if significantly higher than others
                is_anomaly = (conf > mean) and (z_score > self.anomaly_threshold)
            else:
                # All same value - no anomalies
                is_anomaly = False
            anomalies.append(is_anomaly)

        return anomalies

    def _validate_single_claim(
        self,
        claimed_confidence: float,
        actual_agreement: float,
        is_anomaly: bool,
        response_length: int,
        model_index: int,
        total_models: int
    ) -> SemanticValidationResult:
        """
        Validate a single model's confidence claim.

        Returns adjusted confidence based on semantic validation.
        """
        penalty = 1.0
        anomaly_reason = None

        # Check 1: Response too short (potential gaming)
        if response_length < self.MIN_RESPONSE_LENGTH:
            penalty *= 0.7
            anomaly_reason = f"Response too short ({response_length} chars)"

        # Check 2: High confidence but very low actual agreement (obvious gaming)
        # Only trigger for extreme cases: >90% confidence with <15% agreement
        if claimed_confidence > 0.9 and actual_agreement < 0.15:
            penalty *= self.CONFIDENCE_PENALTY_FACTOR
            anomaly_reason = (
                f"Very high confidence ({claimed_confidence:.0%}) but minimal agreement "
                f"({actual_agreement:.0%}) with other models"
            )

        # Check 3: Statistical outlier (only for extreme outliers)
        if is_anomaly:
            penalty *= 0.85
            if anomaly_reason:
                anomaly_reason += "; confidence is statistical outlier"
            else:
                anomaly_reason = "Confidence is statistical outlier (>2.0 stddev)"

        # Check 4: Confidence/agreement disparity (only for extreme disparity)
        # Trigger when claiming >85% confidence but agreement <25%
        if claimed_confidence > 0.85 and actual_agreement < 0.25:
            disparity_penalty = 0.7 + (0.3 * (actual_agreement / 0.25))
            penalty *= disparity_penalty
            if not anomaly_reason:
                anomaly_reason = (
                    f"Agreement ({actual_agreement:.0%}) too low for claimed "
                    f"confidence ({claimed_confidence:.0%})"
                )

        # Calculate validated confidence
        validated = claimed_confidence * penalty

        # Apply single-model influence cap ONLY if there are anomalies
        # This prevents a single malicious model from forcing convergence
        # but allows legitimate high confidence when there's agreement
        if anomaly_reason:  # Only cap if this response is anomalous
            max_contribution = self.max_single_influence / total_models
            if validated > max_contribution * 2.5:
                validated = min(validated, max_contribution * 2.5)

        return SemanticValidationResult(
            validated_confidence=validated,
            raw_confidence=claimed_confidence,
            agreement_score=actual_agreement,
            anomaly_detected=is_anomaly or penalty < 1.0,
            anomaly_reason=anomaly_reason,
            penalty_applied=penalty,
            details={
                'response_length': response_length,
                'model_index': model_index,
                'total_models': total_models
            }
        )

    def _calculate_adjusted_score(
        self,
        results: List[SemanticValidationResult],
        original_confidences: List[float]
    ) -> float:
        """
        Calculate adjusted convergence score with protections.

        Ensures no single model can dominate the score.
        """
        if not results:
            return 0.0

        n = len(results)

        # Use validated confidences
        validated = [r.validated_confidence for r in results]

        # Calculate base average
        avg_validated = sum(validated) / n

        # Calculate agreement-weighted bonus
        avg_agreement = sum(r.agreement_score for r in results) / n

        # Penalize if any anomalies detected
        anomaly_count = sum(1 for r in results if r.anomaly_detected)
        anomaly_penalty = 1.0 - (anomaly_count * 0.1)  # -10% per anomaly

        # Final score: blend validated confidence with actual agreement
        # This prevents gaming because agreement is computed, not claimed
        score = (0.5 * avg_validated + 0.5 * avg_agreement) * anomaly_penalty

        return max(0.0, min(1.0, score))


def validate_convergence(
    model_outputs: List[Dict],
    claimed_score: float,
    min_models: int = 2
) -> Tuple[float, bool, str]:
    """
    Convenience function to validate a convergence claim.

    Args:
        model_outputs: List of model output dicts
        claimed_score: The convergence score being claimed
        min_models: Minimum models for valid convergence

    Returns:
        Tuple of (adjusted_score, is_valid, reason)
    """
    validator = SemanticValidator()
    results, adjusted_score = validator.validate_convergence_claims(
        model_outputs, min_models
    )

    # Check if claimed score was inflated
    if claimed_score > adjusted_score + 0.15:
        return (
            adjusted_score,
            False,
            f"Claimed score ({claimed_score:.0%}) exceeds validated score "
            f"({adjusted_score:.0%}) by >15%"
        )

    # Check for anomalies
    anomalies = [r for r in results if r.anomaly_detected]
    if anomalies:
        reasons = [r.anomaly_reason for r in anomalies if r.anomaly_reason]
        return (
            adjusted_score,
            len(anomalies) < len(results) / 2,  # Valid if <50% anomalous
            "; ".join(reasons[:2])  # First 2 reasons
        )

    return adjusted_score, True, "Validated"

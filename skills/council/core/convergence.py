"""
Convergence detection for multi-round deliberation.

Security: Uses semantic validation to prevent gaming via spoofed confidence.
Models cannot force early exit by simply claiming high confidence - claims
must be backed by actual agreement in response content.
"""

from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from .parsing import extract_json
from .semantic_validator import SemanticValidator, validate_convergence as semantic_validate

# Convergence detection weights and thresholds
CONVERGENCE_CONFIDENCE_WEIGHT = 0.4  # Reduced from 0.6 - less trust in claimed confidence
CONVERGENCE_SIGNAL_WEIGHT = 0.2      # Reduced from 0.4 - signals easily spoofed
CONVERGENCE_AGREEMENT_WEIGHT = 0.4   # NEW: Weight for actual semantic agreement
CONVERGENCE_THRESHOLD = 0.8          # Threshold for declaring convergence

# Security settings
MAX_CONFIDENCE_FROM_SINGLE_MODEL = 0.4  # No model can contribute > 40% to confidence
ANOMALY_PENALTY = 0.15                   # Penalty per anomalous response
MIN_RESPONSE_LENGTH = 100                # Minimum response length for validity


@dataclass
class ConvergenceCheckResult:
    """Extended result with validation details."""
    converged: bool
    score: float
    raw_score: float              # Score before semantic validation
    validated: bool               # Whether semantic validation passed
    validation_reason: str        # Explanation of validation result
    anomaly_count: int            # Number of anomalous responses detected
    semantic_agreement: float     # Actual agreement between responses


def check_convergence(
    round_responses: List[dict],
    threshold: float = CONVERGENCE_THRESHOLD
) -> Tuple[bool, float]:
    """
    Check if models have converged with semantic validation.

    Security: Does NOT blindly trust model-reported confidence scores.
    Instead, validates claims against actual semantic agreement between
    responses. Prevents gaming via spoofed confidence/convergence_signal.

    Args:
        round_responses: List of response dicts from each round
        threshold: Convergence threshold (default 0.8)

    Returns:
        Tuple of (converged: bool, convergence_score: float)
    """
    result = check_convergence_detailed(round_responses, threshold)
    return result.converged, result.score


def check_convergence_detailed(
    round_responses: List[dict],
    threshold: float = CONVERGENCE_THRESHOLD
) -> ConvergenceCheckResult:
    """
    Detailed convergence check with full validation info.

    Args:
        round_responses: List of response dicts from each round
        threshold: Convergence threshold

    Returns:
        ConvergenceCheckResult with full details
    """
    if not round_responses:
        return ConvergenceCheckResult(
            converged=False,
            score=0.0,
            raw_score=0.0,
            validated=False,
            validation_reason="No responses to evaluate",
            anomaly_count=0,
            semantic_agreement=0.0
        )

    latest_round = round_responses[-1]

    # Extract data from each model's response
    model_outputs = []
    convergence_signals = []
    raw_confidences = []

    for model, response in latest_round.items():
        try:
            data = extract_json(response)
            confidence = data.get('confidence', 0.5)

            # Clamp confidence to valid range
            confidence = max(0.0, min(1.0, float(confidence)))

            # Cap single-model influence
            capped_confidence = min(confidence, MAX_CONFIDENCE_FROM_SINGLE_MODEL * 2)

            convergence_signals.append(data.get('convergence_signal', False))
            raw_confidences.append(confidence)

            model_outputs.append({
                'model': model,
                'response': response if isinstance(response, str) else str(data),
                'confidence': capped_confidence,
                'raw_confidence': confidence,
                'convergence_signal': data.get('convergence_signal', False)
            })
        except Exception:
            convergence_signals.append(False)
            raw_confidences.append(0.5)
            model_outputs.append({
                'model': model,
                'response': str(response),
                'confidence': 0.5,
                'raw_confidence': 0.5,
                'convergence_signal': False
            })

    # Calculate raw score (old method, for comparison)
    avg_raw_confidence = sum(raw_confidences) / len(raw_confidences) if raw_confidences else 0.0
    explicit_convergence = sum(convergence_signals) / len(convergence_signals) if convergence_signals else 0.0
    raw_score = (avg_raw_confidence * 0.6) + (explicit_convergence * 0.4)

    # Perform semantic validation
    validator = SemanticValidator(
        max_single_influence=MAX_CONFIDENCE_FROM_SINGLE_MODEL,
        anomaly_threshold=1.5
    )
    validation_results, semantic_score = validator.validate_convergence_claims(
        model_outputs, min_models=2
    )

    # Count anomalies
    anomaly_count = sum(1 for r in validation_results if r.anomaly_detected)

    # Calculate actual semantic agreement
    semantic_agreement = (
        sum(r.agreement_score for r in validation_results) / len(validation_results)
        if validation_results else 0.0
    )

    # Build validated score:
    # - 40% validated confidence (penalized if anomalous)
    # - 40% semantic agreement (computed, not claimed)
    # - 20% convergence signals (but only if agreement is high)
    validated_conf = sum(r.validated_confidence for r in validation_results) / len(validation_results) if validation_results else 0.0

    # Only count signals if there's actual agreement
    signal_weight = CONVERGENCE_SIGNAL_WEIGHT if semantic_agreement > 0.3 else 0.0
    adjusted_conf_weight = CONVERGENCE_CONFIDENCE_WEIGHT + (CONVERGENCE_SIGNAL_WEIGHT - signal_weight)

    validated_score = (
        (validated_conf * adjusted_conf_weight) +
        (semantic_agreement * CONVERGENCE_AGREEMENT_WEIGHT) +
        (explicit_convergence * signal_weight)
    )

    # Apply anomaly penalty
    validated_score *= (1.0 - (anomaly_count * ANOMALY_PENALTY))
    validated_score = max(0.0, min(1.0, validated_score))

    # Determine validation status
    # Note: raw_score includes full signal weight, validated uses reduced weight
    # so some difference is expected. Only fail if anomalies or extreme divergence.
    if anomaly_count > len(model_outputs) / 2:
        validation_reason = f"Majority anomalous ({anomaly_count}/{len(model_outputs)} responses)"
        is_validated = False
    elif anomaly_count > 0 and raw_score > validated_score + 0.3:
        # Only fail if there ARE anomalies AND large score divergence
        validation_reason = f"Anomaly detected with score divergence (raw: {raw_score:.0%}, validated: {validated_score:.0%})"
        is_validated = False
    else:
        validation_reason = "Validated - claims consistent with semantic agreement"
        is_validated = True

    return ConvergenceCheckResult(
        converged=validated_score >= threshold and is_validated,
        score=validated_score,
        raw_score=raw_score,
        validated=is_validated,
        validation_reason=validation_reason,
        anomaly_count=anomaly_count,
        semantic_agreement=semantic_agreement
    )

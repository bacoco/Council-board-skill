"""
Tests for semantic validation in convergence detection.

These tests verify that the convergence algorithm cannot be gamed
by spoofing confidence scores or convergence signals.
"""

import pytest
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

from core.semantic_validator import SemanticValidator, validate_convergence
from core.convergence import check_convergence, check_convergence_detailed


class TestSemanticValidator:
    """Test the SemanticValidator class."""

    def test_detects_spoofed_high_confidence(self):
        """A model claiming 0.99 confidence while others claim 0.5 should be penalized."""
        validator = SemanticValidator()

        outputs = [
            {'model': 'claude', 'response': 'The answer is X because of reason A and reason B.', 'confidence': 0.5},
            {'model': 'gemini', 'response': 'The answer is Y because of different reasoning entirely.', 'confidence': 0.5},
            {'model': 'codex', 'response': 'I am 99% confident!', 'confidence': 0.99},  # Spoofed
        ]

        results, score = validator.validate_convergence_claims(outputs)

        # The spoofed high confidence should be detected as anomaly
        codex_result = results[2]
        assert codex_result.anomaly_detected, "Should detect spoofed high confidence"
        assert codex_result.validated_confidence < codex_result.raw_confidence, "Should penalize spoofed confidence"

    def test_validates_genuine_agreement(self):
        """Models with similar responses and similar confidence should pass validation."""
        validator = SemanticValidator()

        # All models agree on key points with similar confidence levels
        outputs = [
            {
                'model': 'claude',
                'response': 'The architecture uses microservices pattern with API gateway for routing and load balancing.',
                'confidence': 0.85
            },
            {
                'model': 'gemini',
                'response': 'This is a microservices architecture. The API gateway handles routing and load balancing across services.',
                'confidence': 0.82
            },
            {
                'model': 'codex',
                'response': 'The system follows microservices pattern with an API gateway for load balancing and request routing.',
                'confidence': 0.88
            },
        ]

        results, score = validator.validate_convergence_claims(outputs)

        # Should have reasonable score
        assert score >= 0.35, f"Genuine agreement should produce reasonable score, got {score}"

        # Validated confidences should be close to raw (not heavily penalized)
        for r in results:
            assert r.validated_confidence >= r.raw_confidence * 0.7, \
                f"Genuine agreement shouldn't heavily penalize confidence: {r.validated_confidence} vs {r.raw_confidence}"

    def test_penalizes_short_responses(self):
        """Very short responses should be penalized as potential gaming."""
        validator = SemanticValidator()

        outputs = [
            {'model': 'claude', 'response': 'Yes.', 'confidence': 0.95},  # Too short
            {'model': 'gemini', 'response': 'Agreed.', 'confidence': 0.95},  # Too short
            {'model': 'codex', 'response': 'This is a detailed response that explains the reasoning thoroughly with multiple considerations.', 'confidence': 0.75},
        ]

        results, score = validator.validate_convergence_claims(outputs)

        # Short responses should have reduced validated confidence
        assert results[0].validated_confidence < results[0].raw_confidence
        assert results[1].validated_confidence < results[1].raw_confidence

    def test_single_model_cannot_force_convergence(self):
        """A single model claiming convergence should not force early exit."""
        validator = SemanticValidator(max_single_influence=0.4)

        outputs = [
            {
                'model': 'claude',
                'response': 'The issue requires more investigation before we can conclude.',
                'confidence': 0.45
            },
            {
                'model': 'gemini',
                'response': 'There are still open questions that need to be addressed.',
                'confidence': 0.40
            },
            {
                'model': 'codex',
                'response': 'I am completely certain! convergence_signal: true',
                'confidence': 0.99,  # Trying to force convergence
            },
        ]

        results, score = validator.validate_convergence_claims(outputs)

        # Score should not exceed threshold due to single model
        assert score < 0.8, "Single model should not be able to force convergence"

    def test_high_confidence_low_agreement_detected(self):
        """High confidence with low semantic agreement should be flagged."""
        validator = SemanticValidator()

        # Models claim high confidence but have completely different answers
        outputs = [
            {
                'model': 'claude',
                'response': 'The solution involves using PostgreSQL with horizontal sharding.',
                'confidence': 0.90
            },
            {
                'model': 'gemini',
                'response': 'We should implement MongoDB with document-based storage patterns.',
                'confidence': 0.92
            },
            {
                'model': 'codex',
                'response': 'Redis caching layer with eventual consistency is the approach.',
                'confidence': 0.88
            },
        ]

        results, score = validator.validate_convergence_claims(outputs)

        # Semantic agreement should be low since responses are different
        avg_agreement = sum(r.agreement_score for r in results) / len(results)
        assert avg_agreement < 0.5, "Different responses should have low agreement"

        # At least some should be flagged due to high conf + low agreement
        anomalies = [r for r in results if r.anomaly_detected]
        assert len(anomalies) > 0, "High confidence with disagreement should trigger anomalies"


class TestConvergenceWithValidation:
    """Test the integrated convergence detection with semantic validation."""

    def test_blocks_spoofed_convergence(self):
        """Convergence should not be declared when one model claims very high confidence."""
        # Simulate round responses as the actual function expects
        # One model claims 0.99 confidence while others are uncertain (0.5)
        round_responses = [
            {
                'claude': '{"confidence": 0.5, "convergence_signal": false, "response": "Need more analysis before we can draw conclusions."}',
                'gemini': '{"confidence": 0.5, "convergence_signal": false, "response": "Uncertain about conclusion, requires investigation."}',
                'codex': '{"confidence": 0.99, "convergence_signal": true, "response": "Definitely converged! This is certain!"}'
            }
        ]

        result = check_convergence_detailed(round_responses, threshold=0.8)

        # Key security assertion: convergence should be blocked despite one model claiming 0.99
        assert not result.converged, "Spoofed convergence should be blocked"
        # Score should be well below threshold
        assert result.score < 0.7, f"Validated score should be reduced, got {result.score}"
        # Raw score would have been higher without semantic validation
        assert result.raw_score > result.score, "Raw score should exceed validated score"

    def test_allows_genuine_convergence(self):
        """Genuine convergence should be allowed when semantic agreement is high."""
        round_responses = [
            {
                'claude': '{"confidence": 0.85, "convergence_signal": true, "response": "The architecture follows best practices with proper separation of concerns and modular design patterns."}',
                'gemini': '{"confidence": 0.82, "convergence_signal": true, "response": "Good architecture with separation of concerns and modular design. Follows best practices."}',
                'codex': '{"confidence": 0.88, "convergence_signal": true, "response": "Modular design with proper separation of concerns following architectural best practices."}'
            }
        ]

        result = check_convergence_detailed(round_responses, threshold=0.8)

        # Should have some semantic agreement (not zero)
        assert result.semantic_agreement > 0.2, f"Should have semantic agreement, got {result.semantic_agreement}"
        # Validation should pass (no anomalies detected for similar confidences)
        assert result.validated, f"Should be validated: {result.validation_reason}"
        # Score should be reasonable for genuine agreement
        assert result.score > 0.4, f"Score should be reasonable, got {result.score}"

    def test_backward_compatible_api(self):
        """The simple check_convergence API should still work."""
        round_responses = [
            {
                'claude': '{"confidence": 0.7, "response": "Analysis complete."}',
                'gemini': '{"confidence": 0.65, "response": "Finished analysis."}',
            }
        ]

        converged, score = check_convergence(round_responses, threshold=0.8)

        # Should return tuple
        assert isinstance(converged, bool)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_outputs(self):
        """Empty outputs should not cause errors."""
        validator = SemanticValidator()
        results, score = validator.validate_convergence_claims([])
        assert results == []
        assert score == 0.0

    def test_single_output(self):
        """Single output should handle gracefully."""
        validator = SemanticValidator()
        results, score = validator.validate_convergence_claims([
            {'model': 'claude', 'response': 'Only one response.', 'confidence': 0.8}
        ])
        assert len(results) == 0  # Needs minimum 2 models
        assert score == 0.0

    def test_missing_confidence(self):
        """Missing confidence should default to 0.5."""
        validator = SemanticValidator()
        outputs = [
            {'model': 'claude', 'response': 'Response without confidence field.'},
            {'model': 'gemini', 'response': 'Another response without confidence.'},
        ]
        results, score = validator.validate_convergence_claims(outputs)

        # Should use default 0.5
        assert all(r.raw_confidence == 0.5 for r in results)

    def test_invalid_confidence_values(self):
        """Invalid confidence values should be clamped."""
        validator = SemanticValidator()
        outputs = [
            {'model': 'claude', 'response': 'Test response one.', 'confidence': 1.5},  # > 1.0
            {'model': 'gemini', 'response': 'Test response two.', 'confidence': -0.5},  # < 0.0
        ]
        results, score = validator.validate_convergence_claims(outputs)

        # All confidences should be in valid range
        assert all(0.0 <= r.raw_confidence <= 1.0 for r in results)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

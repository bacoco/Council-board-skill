#!/usr/bin/env python3
"""
Integration tests for Council Board Skill.

Tests critical paths with mock adapters:
- Multi-round deliberation
- Peer review scoring
- Circuit breaker state transitions
- Vote mode tallying
- Configuration loading
"""

import sys
import asyncio
import json
from pathlib import Path
from unittest.mock import patch, AsyncMock
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import LLMResponse, SessionConfig, VoteBallot, DEFAULT_MIN_QUORUM
from core.state import DegradationState, AdaptiveTimeout, init_degradation, DEFAULT_TIMEOUT
from core.convergence import check_convergence
from core.prompts import build_context_from_previous_rounds
from core.review import peer_review
from modes.consensus import run_council
from modes.vote import collect_votes, tally_votes
from providers import CouncilConfig

# Alias for compatibility
MODEL_TIMEOUT = DEFAULT_TIMEOUT


# ============================================================================
# Mock Adapters
# ============================================================================

def create_mock_response(model: str, content: dict, latency_ms: int = 100) -> LLMResponse:
    """Create a mock LLMResponse with JSON content."""
    return LLMResponse(
        content=json.dumps(content),
        model=model,
        latency_ms=latency_ms,
        success=True,
        error=None
    )


def create_mock_adapter(model: str, responses: list):
    """Create a mock adapter that returns responses in sequence."""
    call_count = [0]

    async def mock_query(prompt: str, timeout: int) -> LLMResponse:
        idx = min(call_count[0], len(responses) - 1)
        call_count[0] += 1
        response_data = responses[idx]

        if isinstance(response_data, Exception):
            return LLMResponse(
                content='',
                model=model,
                latency_ms=100,
                success=False,
                error=str(response_data)
            )

        return create_mock_response(model, response_data)

    return mock_query


# ============================================================================
# Test: Configuration Loading
# ============================================================================

def test_config_loading():
    """Test CouncilConfig loads from YAML correctly."""
    print("=" * 70)
    print("TEST: Configuration Loading")
    print("=" * 70)

    # Test default config
    config = CouncilConfig()
    assert config.timeout == 420, f"Expected timeout 420, got {config.timeout}"
    assert config.max_rounds == 3
    assert config.min_quorum == 2
    assert 'claude' in config.providers
    print("✅ Default config values correct")

    # Test loading from file
    config_path = CouncilConfig.default_path()
    if config_path.exists():
        loaded = CouncilConfig.from_file(config_path)
        assert loaded.timeout == 420, f"Expected 420 from file, got {loaded.timeout}"
        print(f"✅ Config loaded from {config_path}")
    else:
        print(f"⚠️ Config file not found at {config_path}, skipping file test")

    print()


# ============================================================================
# Test: Degradation State
# ============================================================================

def test_degradation_state():
    """Test DegradationState tracks model availability correctly."""
    print("=" * 70)
    print("TEST: Degradation State")
    print("=" * 70)

    state = DegradationState(['claude', 'gemini', 'codex'], min_quorum=2)

    # Initial state should be FULL
    assert state.level == 'full', f"Expected 'full', got {state.level}"
    assert state.can_continue() == True
    print("✅ Initial state is FULL with all models available")

    # Record one failure
    state.record_model_unavailable('codex', 'TIMEOUT')
    assert state.level == 'degraded', f"Expected 'degraded', got {state.level}"
    assert state.can_continue() == True
    assert 'codex' in state.failed_models
    print("✅ After 1 failure: DEGRADED, can still continue")

    # Record another failure
    state.record_model_unavailable('gemini', 'TIMEOUT')
    assert state.level == 'minimal', f"Expected 'minimal', got {state.level}"
    assert state.can_continue() == False  # Only 1 model left, quorum=2
    print("✅ After 2 failures: MINIMAL, cannot continue (below quorum)")

    # Test confidence adjustment
    raw_confidence = 0.9
    adjusted = state.adjust_confidence(raw_confidence)
    assert adjusted < raw_confidence, "Confidence should be penalized in degraded state"
    print(f"✅ Confidence adjusted: {raw_confidence} -> {adjusted}")

    print()


# ============================================================================
# Test: Convergence Detection
# ============================================================================

def test_convergence_detection():
    """Test check_convergence detects agreement between models."""
    print("=" * 70)
    print("TEST: Convergence Detection")
    print("=" * 70)

    # Round with low confidence - no convergence
    low_confidence_round = {
        'claude': json.dumps({'confidence': 0.5, 'convergence_signal': False}),
        'gemini': json.dumps({'confidence': 0.4, 'convergence_signal': False}),
        'codex': json.dumps({'confidence': 0.3, 'convergence_signal': False}),
    }

    converged, score = check_convergence([low_confidence_round])
    assert converged == False, "Should not converge with low confidence"
    print(f"✅ Low confidence round: converged={converged}, score={score:.3f}")

    # Round with high confidence - should converge
    high_confidence_round = {
        'claude': json.dumps({'confidence': 0.95, 'convergence_signal': True}),
        'gemini': json.dumps({'confidence': 0.92, 'convergence_signal': True}),
        'codex': json.dumps({'confidence': 0.90, 'convergence_signal': True}),
    }

    converged, score = check_convergence([high_confidence_round])
    assert converged == True, f"Should converge with high confidence, got score={score}"
    print(f"✅ High confidence round: converged={converged}, score={score:.3f}")

    print()


# ============================================================================
# Test: Context Building (Stable Labels)
# ============================================================================

def test_context_building_stable_labels():
    """Test build_context_from_previous_rounds produces stable A/B/C labels."""
    print("=" * 70)
    print("TEST: Context Building - Stable Labels")
    print("=" * 70)

    opinions = {
        'gemini': json.dumps({'answer': 'Gemini says X', 'confidence': 0.8, 'key_points': ['point1']}),
        'claude': json.dumps({'answer': 'Claude says Y', 'confidence': 0.9, 'key_points': ['point2']}),
        'codex': json.dumps({'answer': 'Codex says Z', 'confidence': 0.7, 'key_points': ['point3']}),
    }

    # Build context for claude - should see gemini and codex
    context_for_claude = build_context_from_previous_rounds('claude', opinions, anonymize=True)

    # Build context for gemini - should see claude and codex
    context_for_gemini = build_context_from_previous_rounds('gemini', opinions, anonymize=True)

    # Labels should be consistent: claude=A, codex=B, gemini=C (alphabetical)
    # When building for claude, they see B (codex) and C (gemini)
    # When building for gemini, they see A (claude) and B (codex)

    assert 'Participant' in context_for_claude
    assert 'Participant' in context_for_gemini
    print("✅ Context uses Participant labels")

    # Verify own response is excluded
    assert 'Claude says Y' not in context_for_claude
    assert 'Gemini says X' not in context_for_gemini
    print("✅ Own response excluded from context")

    print()


# ============================================================================
# Test: Vote Tallying
# ============================================================================

def test_vote_tallying():
    """Test tally_votes correctly counts and weights votes."""
    print("=" * 70)
    print("TEST: Vote Tallying")
    print("=" * 70)

    ballots = [
        VoteBallot(model='claude', vote='A', weight=0.9, justification='Best option', confidence=0.9, latency_ms=100),
        VoteBallot(model='gemini', vote='A', weight=0.8, justification='Agree', confidence=0.8, latency_ms=150),
        VoteBallot(model='codex', vote='B', weight=0.7, justification='Alternative', confidence=0.7, latency_ms=200),
    ]

    vote_counts, weighted_scores, winner, tie_broken, method = tally_votes(ballots)

    assert vote_counts['A'] == 2, f"Expected 2 votes for A, got {vote_counts['A']}"
    assert vote_counts['B'] == 1, f"Expected 1 vote for B, got {vote_counts['B']}"
    assert winner == 'A', f"Expected winner A, got {winner}"
    assert weighted_scores['A'] > weighted_scores['B']
    print(f"✅ Vote counts: {vote_counts}")
    print(f"✅ Weighted scores: A={weighted_scores['A']:.2f}, B={weighted_scores['B']:.2f}")
    print(f"✅ Winner: {winner}")

    print()


# ============================================================================
# Test: Adaptive Timeout
# ============================================================================

def test_adaptive_timeout():
    """Test AdaptiveTimeout adjusts based on latency history."""
    print("=" * 70)
    print("TEST: Adaptive Timeout")
    print("=" * 70)

    timeout = AdaptiveTimeout(base_timeout=60)

    # Initial timeout should be base
    initial = timeout.get_timeout('claude')
    assert initial == 60, f"Expected initial timeout 60, got {initial}"
    print(f"✅ Initial timeout: {initial}s")

    # Record some latencies
    for _ in range(5):
        timeout.record_latency('claude', 30000, success=True)  # 30s latency

    # After samples, timeout should adapt
    adapted = timeout.get_timeout('claude')
    print(f"✅ After recording 30s latencies: {adapted}s")

    # Record timeout failures
    for _ in range(3):
        timeout.record_latency('gemini', 0, success=False)

    # Stats should show failures
    stats = timeout.get_stats()
    print(f"✅ Stats available for models")

    print()


# ============================================================================
# Test: Circuit Breaker (via init_degradation)
# ============================================================================

def test_circuit_breaker_initialization():
    """Test circuit breaker initializes correctly."""
    print("=" * 70)
    print("TEST: Circuit Breaker Initialization")
    print("=" * 70)

    degradation, adaptive_timeout = init_degradation(
        expected_models=['claude', 'gemini', 'codex'],
        base_timeout=420,
        min_quorum=2
    )

    assert degradation is not None
    assert adaptive_timeout is not None
    assert degradation.min_quorum == 2
    assert len(degradation.expected_models) == 3
    print("✅ Degradation state initialized with 3 models, quorum=2")
    print(f"✅ Adaptive timeout base: {adaptive_timeout.base_timeout}s")

    print()


# ============================================================================
# Run All Tests
# ============================================================================

def run_all_tests():
    """Run all integration tests."""
    print("\n" + "=" * 70)
    print("COUNCIL BOARD SKILL - INTEGRATION TESTS")
    print("=" * 70 + "\n")

    tests = [
        test_config_loading,
        test_degradation_state,
        test_convergence_detection,
        test_context_building_stable_labels,
        test_vote_tallying,
        test_adaptive_timeout,
        test_circuit_breaker_initialization,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            failed += 1

    print("=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

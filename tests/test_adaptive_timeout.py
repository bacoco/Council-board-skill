import sys
from pathlib import Path

# Make council package importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.state import AdaptiveTimeout


def test_mode_specific_adaptation():
    """Ensure adaptive timeout keeps separate histories per (model, mode)."""
    timeout = AdaptiveTimeout(base_timeout=10)

    for latency in [1000, 1100, 1200]:
        timeout.record_latency("claude_instance_1", latency, success=True, mode="consensus")

    consensus_timeout = timeout.get_timeout("claude_instance_1", mode="consensus")

    for latency in [4000, 4200, 4500]:
        timeout.record_latency("claude_instance_1", latency, success=True, mode="debate")

    debate_timeout = timeout.get_timeout("claude_instance_1", mode="debate")

    # Debate mode should adapt based on its longer history without inflating consensus.
    assert debate_timeout > consensus_timeout
    assert timeout.get_timeout("claude_instance_1", mode="consensus") == consensus_timeout

    stats = timeout.get_stats()
    assert "claude" in stats
    assert "debate" in stats["claude"]
    assert stats["claude"]["debate"]["count"] == 3
    assert stats["claude"]["debate"]["p95_ms"] is not None
    assert stats["claude"]["debate"]["mode"] == "debate"


def test_timeout_burst_applies_capped_boost():
    """Verify consecutive timeouts trigger a capped temporary increase."""
    timeout = AdaptiveTimeout(base_timeout=10)

    base_timeout = timeout.get_timeout("claude", mode="consensus")

    # Introduce a burst of timeouts to build a streak.
    for _ in range(3):
        timeout.record_latency("claude", 0, success=False, mode="consensus")

    boosted_timeout = timeout.get_timeout("claude", mode="consensus")
    assert boosted_timeout > base_timeout

    # Additional timeouts should increase timeout but stay within cap.
    for _ in range(5):
        timeout.record_latency("claude", 0, success=False, mode="consensus")

    capped_timeout = timeout.get_timeout("claude", mode="consensus")
    assert capped_timeout >= boosted_timeout
    assert capped_timeout <= int(timeout.base_timeout * timeout.MAX_TIMEOUT_FACTOR)

    stats = timeout.get_stats()
    assert stats["claude"]["consensus"]["timeout_count"] >= 8
    assert stats["claude"]["consensus"]["consecutive_timeouts"] >= 8

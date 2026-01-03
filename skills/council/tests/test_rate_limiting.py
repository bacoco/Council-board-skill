"""
Tests for rate limiting and DoS protection.

Verifies that the RateLimiter correctly limits:
- Sessions per time window
- Queries per session
- Model calls per session
- Total session time
"""

import pytest
import time
import threading
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from security.input_validator import RateLimiter, RateLimitConfig, RATE_LIMITER


class TestRateLimiterBasic:
    """Basic rate limiter functionality tests."""

    def test_allows_first_session(self):
        """First session should be allowed."""
        limiter = RateLimiter()
        allowed, reason = limiter.start_session("session_1")
        assert allowed == True
        assert reason is None

    def test_allows_first_query(self):
        """First query in a session should be allowed."""
        limiter = RateLimiter()
        limiter.start_session("session_1")

        allowed, reason = limiter.check_query("session_1")
        assert allowed == True
        assert reason is None

    def test_tracks_query_count(self):
        """Query count should be tracked."""
        limiter = RateLimiter()
        limiter.start_session("session_1")

        for i in range(5):
            limiter.record_query("session_1")

        stats = limiter.get_session_stats("session_1")
        assert stats['query_count'] == 5

    def test_tracks_model_calls(self):
        """Model call count should be tracked."""
        limiter = RateLimiter()
        limiter.start_session("session_1")

        for i in range(10):
            limiter.record_model_call("session_1")

        stats = limiter.get_session_stats("session_1")
        assert stats['model_calls'] == 10


class TestRateLimitEnforcement:
    """Tests for rate limit enforcement."""

    def test_blocks_excessive_sessions(self):
        """Should block sessions when rate limit exceeded."""
        config = RateLimitConfig(max_sessions_per_minute=3, window_seconds=60)
        limiter = RateLimiter(config)

        # Start 3 sessions (at limit)
        for i in range(3):
            allowed, _ = limiter.start_session(f"session_{i}")
            assert allowed == True

        # 4th session should be blocked
        allowed, reason = limiter.start_session("session_4")
        assert allowed == False
        assert "Rate limit exceeded" in reason

    def test_blocks_excessive_queries(self):
        """Should block queries when per-session limit exceeded."""
        config = RateLimitConfig(max_queries_per_session=5)
        limiter = RateLimiter(config)
        limiter.start_session("session_1")

        # Record 5 queries (at limit)
        for i in range(5):
            limiter.record_query("session_1")

        # 6th query should be blocked
        allowed, reason = limiter.check_query("session_1")
        assert allowed == False
        assert "Query limit exceeded" in reason

    def test_blocks_excessive_model_calls(self):
        """Should block model calls when per-session limit exceeded."""
        config = RateLimitConfig(max_model_calls_per_session=10)
        limiter = RateLimiter(config)
        limiter.start_session("session_1")

        # Record 10 model calls (at limit)
        for i in range(10):
            limiter.record_model_call("session_1")

        # 11th call should be blocked
        allowed, reason = limiter.check_model_call("session_1")
        assert allowed == False
        assert "Model call limit exceeded" in reason

    def test_blocks_expired_sessions(self):
        """Should block queries in expired sessions."""
        config = RateLimitConfig(max_total_time_seconds=1)  # 1 second limit
        limiter = RateLimiter(config)
        limiter.start_session("session_1")

        # Wait for session to expire
        time.sleep(1.1)

        allowed, reason = limiter.check_query("session_1")
        assert allowed == False
        assert "Session time limit exceeded" in reason


class TestSessionManagement:
    """Tests for session lifecycle management."""

    def test_end_session_cleans_up(self):
        """Ending a session should remove it from tracking."""
        limiter = RateLimiter()
        limiter.start_session("session_1")

        stats = limiter.get_session_stats("session_1")
        assert stats is not None

        limiter.end_session("session_1")

        stats = limiter.get_session_stats("session_1")
        assert stats is None

    def test_cleanup_stale_sessions(self):
        """Stale sessions should be cleaned up."""
        limiter = RateLimiter()
        limiter.start_session("session_1")

        # Cleanup with 0 idle time (everything is stale)
        cleaned = limiter.cleanup_stale_sessions(max_idle_seconds=0)
        assert cleaned == 1

        stats = limiter.get_session_stats("session_1")
        assert stats is None

    def test_unknown_session_rejected(self):
        """Unknown session IDs should be rejected."""
        limiter = RateLimiter()

        allowed, reason = limiter.check_query("nonexistent")
        assert allowed == False
        assert "Session not found" in reason


class TestRateLimiterThreadSafety:
    """Thread safety tests for rate limiter."""

    def test_concurrent_session_starts(self):
        """Concurrent session starts should be thread-safe."""
        config = RateLimitConfig(max_sessions_per_minute=100)
        limiter = RateLimiter(config)
        errors = []

        def start_sessions(start_id):
            try:
                for i in range(10):
                    limiter.start_session(f"session_{start_id}_{i}")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=start_sessions, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_query_recording(self):
        """Concurrent query recording should be thread-safe."""
        limiter = RateLimiter()
        limiter.start_session("session_1")
        errors = []

        def record_queries():
            try:
                for _ in range(100):
                    limiter.record_query("session_1")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=record_queries) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = limiter.get_session_stats("session_1")
        assert stats['query_count'] == 1000  # 10 threads * 100 queries


class TestGlobalRateLimiter:
    """Tests for the global RATE_LIMITER instance."""

    def test_global_limiter_exists(self):
        """Global rate limiter should exist."""
        assert RATE_LIMITER is not None

    def test_global_limiter_works(self):
        """Global rate limiter should be functional."""
        session_id = f"test_global_{time.time()}"
        allowed, _ = RATE_LIMITER.start_session(session_id)
        assert allowed == True

        RATE_LIMITER.end_session(session_id)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

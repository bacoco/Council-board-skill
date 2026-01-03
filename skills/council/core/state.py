"""
State management for Council deliberation.

Contains:
- CircuitBreaker: Model failure handling with state machine
- DegradationLevel/DegradationState: Graceful degradation tracking
- AdaptiveTimeout: Dynamic timeout based on performance history
"""

import threading
import time
from typing import List, Optional, Tuple

from .emit import emit
from .models import DEFAULT_MIN_QUORUM

# Timeout for model calls - same for all since they run in parallel
# Total wait time = slowest model, so no point having different values
DEFAULT_TIMEOUT = 420  # 7 minutes - Codex needs time to explore code with tools


def get_base_model(model_instance: str) -> str:
    """Extract base model name from instance ID (e.g., 'claude_instance_1' -> 'claude')."""
    if '_instance_' in model_instance:
        return model_instance.split('_instance_')[0]
    return model_instance


class CircuitBreaker:
    """
    Circuit breaker for model failure handling.

    Prevents cascading failures by temporarily excluding models that fail repeatedly.
    States: CLOSED (normal) -> OPEN (failing, excluded) -> HALF_OPEN (testing recovery)

    Thread-safe: All mutations are protected by an internal lock.

    Usage:
        breaker = CircuitBreaker()
        if breaker.can_call("claude"):
            result = call_model(...)
            if result.success:
                breaker.record_success("claude")
            else:
                breaker.record_failure("claude")
    """

    # Circuit breaker configuration
    FAILURE_THRESHOLD = 3      # Failures before opening circuit
    RECOVERY_TIMEOUT = 60      # Seconds before trying again (HALF_OPEN)
    SUCCESS_THRESHOLD = 2      # Successes in HALF_OPEN to close circuit

    def __init__(self):
        self._lock = threading.RLock()  # Reentrant lock for nested calls
        self._failures = {}      # model -> failure count
        self._successes = {}     # model -> success count in half-open
        self._state = {}         # model -> 'closed' | 'open' | 'half_open'
        self._last_failure = {}  # model -> timestamp of last failure
        self._total_calls = {}   # model -> total call count (for metrics)
        self._total_failures = {}  # model -> total failure count (for metrics)

    def _get_state_unlocked(self, model: str) -> str:
        """
        Get current state for a model, checking for recovery timeout.

        INTERNAL: Must be called while holding self._lock.
        """
        state = self._state.get(model, 'closed')

        if state == 'open':
            # Check if recovery timeout has passed
            last_fail = self._last_failure.get(model, 0)
            if time.time() - last_fail >= self.RECOVERY_TIMEOUT:
                self._state[model] = 'half_open'
                self._successes[model] = 0
                return 'half_open'

        return state

    def can_call(self, model: str) -> bool:
        """
        Check if a model can be called (circuit not open).

        Thread-safe.
        """
        with self._lock:
            state = self._get_state_unlocked(model)
            return state != 'open'

    def record_success(self, model: str):
        """
        Record a successful call to a model.

        Thread-safe.
        """
        with self._lock:
            self._total_calls[model] = self._total_calls.get(model, 0) + 1
            state = self._get_state_unlocked(model)

            if state == 'half_open':
                self._successes[model] = self._successes.get(model, 0) + 1
                if self._successes[model] >= self.SUCCESS_THRESHOLD:
                    # Close the circuit - model recovered
                    self._state[model] = 'closed'
                    self._failures[model] = 0
                    emit({
                        "type": "circuit_breaker",
                        "model": model,
                        "event": "closed",
                        "msg": f"Circuit closed for {model} - model recovered"
                    })
            elif state == 'closed':
                # Reset failure count on success
                self._failures[model] = 0

    def record_failure(self, model: str, error: str = None):
        """
        Record a failed call to a model.

        Thread-safe.
        """
        with self._lock:
            self._total_calls[model] = self._total_calls.get(model, 0) + 1
            self._total_failures[model] = self._total_failures.get(model, 0) + 1
            self._last_failure[model] = time.time()

            state = self._get_state_unlocked(model)

            if state == 'half_open':
                # Failure during recovery - reopen circuit
                self._state[model] = 'open'
                emit({
                    "type": "circuit_breaker",
                    "model": model,
                    "event": "reopened",
                    "msg": f"Circuit reopened for {model} - failed during recovery",
                    "error": error
                })
            elif state == 'closed':
                self._failures[model] = self._failures.get(model, 0) + 1
                if self._failures[model] >= self.FAILURE_THRESHOLD:
                    # Open the circuit
                    self._state[model] = 'open'
                    emit({
                        "type": "circuit_breaker",
                        "model": model,
                        "event": "opened",
                        "msg": f"Circuit opened for {model} - {self._failures[model]} consecutive failures",
                        "error": error
                    })

    def get_available_models(self, models: List[str]) -> List[str]:
        """
        Filter models to only those with closed or half-open circuits.

        Thread-safe.
        """
        with self._lock:
            return [m for m in models if self._get_state_unlocked(m) != 'open']

    def get_status(self) -> dict:
        """
        Get circuit breaker status for all tracked models.

        Thread-safe: Returns a snapshot of current state.
        """
        with self._lock:
            status = {}
            for model in set(self._state.keys()) | set(self._failures.keys()):
                status[model] = {
                    "state": self._get_state_unlocked(model),
                    "failures": self._failures.get(model, 0),
                    "total_calls": self._total_calls.get(model, 0),
                    "total_failures": self._total_failures.get(model, 0),
                    "failure_rate": (
                        self._total_failures.get(model, 0) / self._total_calls.get(model, 1)
                        if self._total_calls.get(model, 0) > 0 else 0
                    )
                }
            return status

    def reset(self, model: str = None):
        """
        Reset circuit breaker state for a model or all models.

        Thread-safe.
        """
        with self._lock:
            if model:
                self._failures.pop(model, None)
                self._successes.pop(model, None)
                self._state.pop(model, None)
                self._last_failure.pop(model, None)
            else:
                self._failures.clear()
                self._successes.clear()
                self._state.clear()
                self._last_failure.clear()


# Global circuit breaker instance
CIRCUIT_BREAKER = CircuitBreaker()


class DegradationLevel:
    """Enumeration of degradation levels."""
    FULL = 'full'           # All models operational
    DEGRADED = 'degraded'   # Some models failed, operating with reduced capacity
    MINIMAL = 'minimal'     # Critical degradation, minimum viable operation


class DegradationState:
    """
    Tracks graceful degradation state across a council session.

    Monitors model availability, adjusts quality expectations, and provides
    degradation-aware confidence scoring.

    Thread-safe: All mutations are protected by an internal lock.

    Usage:
        state = DegradationState(expected_models=['claude', 'gemini', 'codex'])
        state.record_model_unavailable('gemini', 'TIMEOUT')
        adjusted_confidence = state.adjust_confidence(0.85)
        summary = state.get_summary()
    """

    # Confidence penalties per degradation level
    CONFIDENCE_PENALTIES = {
        DegradationLevel.FULL: 0.0,
        DegradationLevel.DEGRADED: 0.10,  # 10% penalty
        DegradationLevel.MINIMAL: 0.25,   # 25% penalty
    }

    def __init__(self, expected_models: List[str], min_quorum: int = DEFAULT_MIN_QUORUM):
        self._lock = threading.RLock()  # Reentrant lock for nested calls
        self.expected_models = frozenset(expected_models)  # Immutable after init
        self._available_models = set(expected_models)
        self._failed_models = {}  # model -> error reason
        self._recovered_models = set()  # models that recovered mid-session
        self.min_quorum = min_quorum
        self._level = DegradationLevel.FULL
        self._events = []
        self._start_time = time.time()

    @property
    def available_models(self) -> set:
        """Thread-safe access to available models (returns copy)."""
        with self._lock:
            return set(self._available_models)

    @property
    def failed_models(self) -> dict:
        """Thread-safe access to failed models (returns copy)."""
        with self._lock:
            return dict(self._failed_models)

    @property
    def recovered_models(self) -> set:
        """Thread-safe access to recovered models (returns copy)."""
        with self._lock:
            return set(self._recovered_models)

    def _get_level_unlocked(self) -> str:
        """Get current level. INTERNAL: Must be called while holding self._lock."""
        available_count = len(self._available_models)
        expected_count = len(self.expected_models)

        if available_count == expected_count:
            return DegradationLevel.FULL
        elif available_count >= self.min_quorum:
            return DegradationLevel.DEGRADED
        else:
            return DegradationLevel.MINIMAL

    @property
    def level(self) -> str:
        """Current degradation level based on model availability. Thread-safe."""
        with self._lock:
            return self._get_level_unlocked()

    def record_model_unavailable(self, model: str, reason: str):
        """
        Record a model becoming unavailable.

        Thread-safe.
        """
        with self._lock:
            if model in self._available_models:
                self._available_models.discard(model)
                self._failed_models[model] = reason

                event = {
                    'type': 'model_degraded',
                    'model': model,
                    'reason': reason,
                    'level': self._get_level_unlocked(),
                    'available_count': len(self._available_models),
                    'timestamp': time.time()
                }
                self._events.append(event)
                emit(event)

    def record_model_recovered(self, model: str):
        """
        Record a model recovering (e.g., circuit breaker closing).

        Thread-safe.
        """
        with self._lock:
            if model in self._failed_models:
                self._available_models.add(model)
                self._recovered_models.add(model)

                event = {
                    'type': 'model_recovered',
                    'model': model,
                    'level': self._get_level_unlocked(),
                    'available_count': len(self._available_models),
                    'timestamp': time.time()
                }
                self._events.append(event)
                emit(event)

    def adjust_confidence(self, raw_confidence: float) -> float:
        """
        Adjust confidence score based on degradation level.

        When operating in degraded mode, confidence is reduced to reflect
        the lower reliability of having fewer model perspectives.

        Thread-safe.
        """
        with self._lock:
            penalty = self.CONFIDENCE_PENALTIES.get(self._get_level_unlocked(), 0.0)
            adjusted = raw_confidence * (1.0 - penalty)
            return round(max(0.0, min(1.0, adjusted)), 3)

    def can_continue(self) -> bool:
        """
        Check if session can continue (meets minimum quorum).

        Thread-safe.
        """
        with self._lock:
            return len(self._available_models) >= self.min_quorum

    def get_fallback_models(self) -> List[str]:
        """
        Get list of available models for fallback operations.

        Thread-safe.
        """
        with self._lock:
            return list(self._available_models)

    def get_summary(self) -> dict:
        """
        Get degradation state summary for observability.

        Thread-safe: Returns a snapshot of current state.
        """
        with self._lock:
            return {
                'level': self._get_level_unlocked(),
                'expected_models': list(self.expected_models),
                'available_models': list(self._available_models),
                'failed_models': dict(self._failed_models),
                'recovered_models': list(self._recovered_models),
                'availability_ratio': len(self._available_models) / max(1, len(self.expected_models)),
                'event_count': len(self._events),
                'duration_ms': int((time.time() - self._start_time) * 1000)
            }


class AdaptiveTimeout:
    """
    Adaptive timeout based on model performance history.

    Learns from model response times and adjusts timeouts dynamically
    to balance reliability with responsiveness.

    Thread-safe: All mutations are protected by an internal lock.

    Usage:
        timeout = AdaptiveTimeout(base_timeout=60)
        adjusted = timeout.get_timeout('claude')  # Returns adaptive timeout
        timeout.record_latency('claude', 2500, success=True)
    """

    # Timeout adjustment factors
    MIN_TIMEOUT_FACTOR = 0.5   # Never go below 50% of base
    MAX_TIMEOUT_FACTOR = 2.0   # Never exceed 200% of base
    SAFETY_MARGIN = 1.5        # Add 50% to observed p95 for safety
    MIN_SAMPLES = 3            # Minimum samples before adapting
    TIMEOUT_SLOPE = 0.2        # Incremental boost per additional consecutive timeout
    TIMEOUT_STREAK_CAP = 3     # Cap the streak that contributes to boost

    def __init__(self, base_timeout: int = DEFAULT_TIMEOUT):
        self._lock = threading.RLock()  # Reentrant lock for nested calls
        self.base_timeout = base_timeout
        self._latencies = {}  # (model, mode) -> list of latencies
        self._timeouts = {}   # (model, mode) -> count of timeout occurrences
        self._timeout_streak = {}  # (model, mode) -> consecutive timeout count

    def _get_key(self, model: str, mode: Optional[str]) -> Tuple[str, str]:
        """Normalize tracking key to (base_model, mode)."""
        base_model = get_base_model(model)
        return base_model, mode or "default"

    def _ensure_entry_unlocked(self, key: Tuple[str, str]):
        """Ensure entry exists. INTERNAL: Must be called while holding self._lock."""
        if key not in self._latencies:
            self._latencies[key] = []
            self._timeouts[key] = 0
            self._timeout_streak[key] = 0

    def record_latency(self, model: str, latency_ms: float, success: bool = True, mode: Optional[str] = None):
        """
        Record a model call latency.

        Thread-safe.
        """
        key = self._get_key(model, mode)
        with self._lock:
            self._ensure_entry_unlocked(key)

            if success:
                self._latencies[key].append(latency_ms)
                # Keep only last 20 samples for recency bias
                self._latencies[key] = self._latencies[key][-20:]
                self._timeout_streak[key] = 0
            else:
                self._timeouts[key] += 1
                self._timeout_streak[key] += 1

    def _compute_p95(self, latencies: List[float]) -> float:
        """Compute p95 latency from a list."""
        if not latencies:
            return 0
        sorted_latencies = sorted(latencies)
        p95_index = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(p95_index, len(sorted_latencies) - 1)]

    def _get_boost_factor_unlocked(self, key: Tuple[str, str]) -> float:
        """
        Return temporary boost factor based on consecutive timeout streak.

        INTERNAL: Must be called while holding self._lock.
        """
        streak = self._timeout_streak.get(key, 0)
        if streak < 2:
            return 1.0

        effective_streak = min(streak - 1, self.TIMEOUT_STREAK_CAP)
        return 1.0 + (effective_streak * self.TIMEOUT_SLOPE)

    def get_timeout(self, model: str, mode: Optional[str] = None) -> int:
        """
        Get adaptive timeout for a model.

        Thread-safe.
        """
        key = self._get_key(model, mode)
        with self._lock:
            self._ensure_entry_unlocked(key)

            latencies = self._latencies.get(key, [])

            if len(latencies) < self.MIN_SAMPLES:
                # Not enough data - use base timeout with any temporary boost
                base_timeout = self.base_timeout
            else:
                # Calculate p95 latency
                p95_latency = self._compute_p95(latencies)

                # Apply safety margin and convert to seconds
                base_timeout = int((p95_latency * self.SAFETY_MARGIN) / 1000)

            # Clamp to allowed range before temporary boost
            min_timeout = int(self.base_timeout * self.MIN_TIMEOUT_FACTOR)
            max_timeout = int(self.base_timeout * self.MAX_TIMEOUT_FACTOR)

            adaptive_timeout = max(min_timeout, min(max_timeout, base_timeout))

            boosted_timeout = adaptive_timeout * self._get_boost_factor_unlocked(key)

            return max(min_timeout, min(max_timeout, int(boosted_timeout)))

    def get_stats(self) -> dict:
        """
        Get timeout statistics for all models.

        Thread-safe: Returns a snapshot of current state.
        """
        with self._lock:
            stats: dict[str, dict[str, dict]] = {}
            for (model, mode), latencies in self._latencies.items():
                if not latencies and self._timeouts.get((model, mode), 0) == 0:
                    continue

                p95_latency = self._compute_p95(latencies) if latencies else None
                mode_label = mode or "default"

                stats.setdefault(model, {})
                stats[model][mode_label] = {
                    'mode': mode_label,
                    'count': len(latencies),
                    'avg_ms': int(sum(latencies) / len(latencies)) if latencies else None,
                    'max_ms': int(max(latencies)) if latencies else None,
                    'p95_ms': int(p95_latency) if p95_latency is not None else None,
                    'adaptive_timeout_s': self.get_timeout(model, mode),
                    'timeout_count': self._timeouts.get((model, mode), 0),
                    'consecutive_timeouts': self._timeout_streak.get((model, mode), 0)
                }
            return stats


# Global instances with thread-safe access
# Note: These globals are session-scoped. For true concurrent sessions,
# consider passing state explicitly or using context variables.
_STATE_LOCK = threading.Lock()
DEGRADATION_STATE: Optional[DegradationState] = None
ADAPTIVE_TIMEOUT: Optional[AdaptiveTimeout] = None


def init_degradation(expected_models: List[str], base_timeout: int = DEFAULT_TIMEOUT, min_quorum: int = DEFAULT_MIN_QUORUM) -> Tuple[DegradationState, AdaptiveTimeout]:
    """
    Initialize degradation tracking for a new session.

    Thread-safe: Uses lock to prevent race conditions during initialization.
    """
    global DEGRADATION_STATE, ADAPTIVE_TIMEOUT
    with _STATE_LOCK:
        DEGRADATION_STATE = DegradationState(expected_models, min_quorum=min_quorum)
        ADAPTIVE_TIMEOUT = AdaptiveTimeout(base_timeout)
        return DEGRADATION_STATE, ADAPTIVE_TIMEOUT


def get_degradation_state() -> Optional[DegradationState]:
    """Get current degradation state (thread-safe read)."""
    with _STATE_LOCK:
        return DEGRADATION_STATE


def get_adaptive_timeout() -> Optional[AdaptiveTimeout]:
    """Get current adaptive timeout manager (thread-safe read)."""
    with _STATE_LOCK:
        return ADAPTIVE_TIMEOUT


def reset_session_state() -> None:
    """
    Reset ALL global state for a fresh session.

    Call this at the start of each council session to prevent
    cross-session state contamination. Thread-safe.

    Resets:
    - CIRCUIT_BREAKER: All model states to CLOSED
    - DEGRADATION_STATE: Cleared (re-init via init_degradation)
    - ADAPTIVE_TIMEOUT: Cleared (re-init via init_degradation)
    """
    global DEGRADATION_STATE, ADAPTIVE_TIMEOUT
    with _STATE_LOCK:
        CIRCUIT_BREAKER.reset()
        DEGRADATION_STATE = None
        ADAPTIVE_TIMEOUT = None

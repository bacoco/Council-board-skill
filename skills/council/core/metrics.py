"""
Observability and metrics for Council deliberation.

Provides lightweight session metrics tracking for:
- Latency histograms per stage
- Per-model latency stats
- Semantic events (timeouts, quorum failures, circuit breaker events)
- Session-level aggregates
"""

import time
from typing import Optional

from .emit import emit


class SessionMetrics:
    """
    Lightweight observability for council sessions.

    Tracks:
    - Latency histograms per stage (persona_gen, model_call, peer_review, synthesis)
    - Per-model latency stats
    - Semantic events (timeouts, quorum failures, circuit breaker events)
    - Session-level aggregates

    Usage:
        metrics = SessionMetrics(session_id)
        metrics.record_latency('persona_gen', 1500)
        metrics.record_model_latency('claude', 2000, success=True)
        metrics.emit_event('QuorumNotMet', {'required': 2, 'got': 1})
        summary = metrics.get_summary()
    """

    # Stage names for latency tracking
    STAGES = ['persona_gen', 'model_call', 'peer_review', 'synthesis', 'vote_collection', 'vote_tally']

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = time.time()

        # Latency histograms per stage (list of latencies in ms)
        self._stage_latencies = {stage: [] for stage in self.STAGES}

        # Per-model latency tracking
        self._model_latencies = {}  # model -> [latencies]
        self._model_successes = {}  # model -> count
        self._model_failures = {}   # model -> count

        # Semantic events log
        self._events = []

        # Round tracking
        self._rounds = 0
        self._round_latencies = []

        # Persona cache tracking
        self._persona_cache_events = []

    def record_latency(self, stage: str, latency_ms: float):
        """Record latency for a processing stage."""
        if stage in self._stage_latencies:
            self._stage_latencies[stage].append(latency_ms)

    def record_model_latency(self, model: str, latency_ms: float, success: bool = True):
        """Record latency for a specific model call."""
        if model not in self._model_latencies:
            self._model_latencies[model] = []
            self._model_successes[model] = 0
            self._model_failures[model] = 0

        self._model_latencies[model].append(latency_ms)
        if success:
            self._model_successes[model] += 1
        else:
            self._model_failures[model] += 1

    def record_round(self, round_num: int, latency_ms: float):
        """Record completion of a deliberation round."""
        self._rounds = round_num
        self._round_latencies.append(latency_ms)

    def record_persona_cache(self, cached: bool, latency_ms: Optional[float] = None):
        """Record persona cache usage for observability."""
        event = {
            "cached": cached,
            "timestamp": time.time()
        }
        if latency_ms is not None:
            event["generation_ms"] = latency_ms
        self._persona_cache_events.append(event)

    def emit_event(self, event_type: str, details: dict = None):
        """
        Emit a structured semantic event.

        Standard event types:
        - QuorumNotMet: Required quorum not achieved
        - ModelTimedOut: Model exceeded timeout
        - CircuitBreakerOpen: Circuit breaker tripped
        - PartialResultReturned: Degraded response due to failures
        - ConvergenceAchieved: Models reached consensus
        - TieBroken: Vote tie resolved
        - ValidationWarning: Input validation issues
        """
        event = {
            "type": "observability_event",
            "event": event_type,
            "session_id": self.session_id,
            "timestamp": time.time(),
            "details": details or {}
        }
        self._events.append(event)
        emit(event)

    def _percentile(self, data: list, p: float) -> float:
        """Calculate percentile of a list."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_data) else f
        return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)

    def _latency_stats(self, latencies: list) -> dict:
        """Calculate latency statistics for a list of latencies."""
        if not latencies:
            return {"count": 0, "min": 0, "max": 0, "avg": 0, "p50": 0, "p90": 0, "p99": 0}

        return {
            "count": len(latencies),
            "min": round(min(latencies), 1),
            "max": round(max(latencies), 1),
            "avg": round(sum(latencies) / len(latencies), 1),
            "p50": round(self._percentile(latencies, 0.50), 1),
            "p90": round(self._percentile(latencies, 0.90), 1),
            "p99": round(self._percentile(latencies, 0.99), 1),
        }

    def get_summary(self) -> dict:
        """Get comprehensive metrics summary for the session."""
        total_duration = int((time.time() - self.start_time) * 1000)

        # Stage latency stats
        stage_stats = {}
        for stage, latencies in self._stage_latencies.items():
            if latencies:
                stage_stats[stage] = self._latency_stats(latencies)

        # Model latency stats
        model_stats = {}
        for model in self._model_latencies:
            model_stats[model] = {
                "latency": self._latency_stats(self._model_latencies[model]),
                "successes": self._model_successes.get(model, 0),
                "failures": self._model_failures.get(model, 0),
                "success_rate": round(
                    self._model_successes.get(model, 0) /
                    max(1, self._model_successes.get(model, 0) + self._model_failures.get(model, 0)),
                    3
                )
            }

        # Aggregate all model call latencies
        all_model_latencies = []
        for latencies in self._model_latencies.values():
            all_model_latencies.extend(latencies)

        return {
            "session_id": self.session_id,
            "total_duration_ms": total_duration,
            "rounds": self._rounds,
            "round_latencies": self._round_latencies,
            "stage_latencies": stage_stats,
            "model_stats": model_stats,
            "aggregate_model_latency": self._latency_stats(all_model_latencies),
            "persona_cache": self._persona_cache_events,
            "events": [e["event"] for e in self._events],
            "event_count": len(self._events),
        }

    def emit_summary(self):
        """Emit the metrics summary as an observability event."""
        summary = self.get_summary()
        emit({
            "type": "metrics_summary",
            "session_id": self.session_id,
            "metrics": summary
        })


# Global metrics instance (set per session)
CURRENT_METRICS: Optional['SessionMetrics'] = None


def init_metrics(session_id: str) -> SessionMetrics:
    """Initialize metrics for a new session."""
    global CURRENT_METRICS
    CURRENT_METRICS = SessionMetrics(session_id)
    return CURRENT_METRICS


def get_metrics() -> Optional[SessionMetrics]:
    """Get current session metrics."""
    return CURRENT_METRICS


def get_current_session_id() -> str:
    """Return current session identifier for cache scoping."""
    metrics = get_metrics()
    if metrics:
        return metrics.session_id
    return "global"

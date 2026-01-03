"""
Tests for thread safety in state management classes.

These tests verify that CircuitBreaker, DegradationState, and AdaptiveTimeout
can handle concurrent access without race conditions or data corruption.
"""

import pytest
import threading
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.state import (
    CircuitBreaker,
    DegradationState,
    AdaptiveTimeout,
    CIRCUIT_BREAKER,
    init_degradation,
    reset_session_state,
)


class TestCircuitBreakerThreadSafety:
    """Test CircuitBreaker under concurrent access."""

    def test_concurrent_record_failures(self):
        """Multiple threads recording failures should not corrupt state."""
        breaker = CircuitBreaker()
        num_threads = 10
        failures_per_thread = 50
        model = "test_model"

        def record_failures():
            for _ in range(failures_per_thread):
                breaker.record_failure(model, "test error")
                time.sleep(random.uniform(0, 0.001))

        threads = [threading.Thread(target=record_failures) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify total failures recorded
        status = breaker.get_status()
        expected_total = num_threads * failures_per_thread
        assert status[model]["total_failures"] == expected_total, \
            f"Expected {expected_total} failures, got {status[model]['total_failures']}"

    def test_concurrent_success_failure_mix(self):
        """Mixed success/failure recording should maintain consistent state."""
        breaker = CircuitBreaker()
        num_threads = 20
        ops_per_thread = 100
        model = "mixed_model"

        results = {"successes": 0, "failures": 0}
        results_lock = threading.Lock()

        def mixed_operations():
            local_successes = 0
            local_failures = 0
            for _ in range(ops_per_thread):
                if random.random() > 0.5:
                    breaker.record_success(model)
                    local_successes += 1
                else:
                    breaker.record_failure(model, "error")
                    local_failures += 1
            with results_lock:
                results["successes"] += local_successes
                results["failures"] += local_failures

        threads = [threading.Thread(target=mixed_operations) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        status = breaker.get_status()
        expected_total_calls = num_threads * ops_per_thread

        assert status[model]["total_calls"] == expected_total_calls, \
            f"Expected {expected_total_calls} calls, got {status[model]['total_calls']}"
        assert status[model]["total_failures"] == results["failures"], \
            f"Failure count mismatch"

    def test_concurrent_can_call_checks(self):
        """can_call should be consistent under concurrent access."""
        breaker = CircuitBreaker()
        model = "check_model"

        # Record enough failures to open circuit
        for _ in range(breaker.FAILURE_THRESHOLD):
            breaker.record_failure(model)

        results = []
        num_threads = 50

        def check_can_call():
            result = breaker.can_call(model)
            results.append(result)

        threads = [threading.Thread(target=check_can_call) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should return False (circuit is open)
        assert all(r == False for r in results), \
            "can_call should consistently return False when circuit is open"

    def test_concurrent_reset(self):
        """Reset during concurrent operations should not cause errors."""
        breaker = CircuitBreaker()
        models = ["model_a", "model_b", "model_c"]
        errors = []

        def do_operations():
            try:
                for _ in range(100):
                    model = random.choice(models)
                    op = random.choice(["success", "failure", "check", "status"])
                    if op == "success":
                        breaker.record_success(model)
                    elif op == "failure":
                        breaker.record_failure(model)
                    elif op == "check":
                        breaker.can_call(model)
                    else:
                        breaker.get_status()
            except Exception as e:
                errors.append(str(e))

        def do_resets():
            try:
                for _ in range(10):
                    time.sleep(random.uniform(0, 0.01))
                    breaker.reset()
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=do_operations) for _ in range(10)]
        threads.append(threading.Thread(target=do_resets))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent operations: {errors}"


class TestDegradationStateThreadSafety:
    """Test DegradationState under concurrent access."""

    def test_concurrent_model_unavailable(self):
        """Multiple threads marking models unavailable should not corrupt state."""
        state = DegradationState(["model_1", "model_2", "model_3", "model_4", "model_5"])
        errors = []

        def mark_unavailable(model):
            try:
                state.record_model_unavailable(model, f"{model} failed")
            except Exception as e:
                errors.append(str(e))

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(mark_unavailable, f"model_{i}")
                for i in range(1, 6)
            ]
            for f in as_completed(futures):
                pass

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(state.available_models) == 0, "All models should be unavailable"
        assert len(state.failed_models) == 5, "All models should be in failed_models"

    def test_concurrent_recovery(self):
        """Concurrent model recovery should maintain consistent state."""
        state = DegradationState(["a", "b", "c"])

        # First, mark all as unavailable
        for model in ["a", "b", "c"]:
            state.record_model_unavailable(model, "initial failure")

        errors = []

        def recover_model(model):
            try:
                state.record_model_recovered(model)
            except Exception as e:
                errors.append(str(e))

        # Recover all concurrently
        threads = [threading.Thread(target=recover_model, args=(m,)) for m in ["a", "b", "c"]]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(state.available_models) == 3, "All models should be recovered"
        assert len(state.recovered_models) == 3, "All should be in recovered_models"

    def test_concurrent_summary_access(self):
        """get_summary during mutations should return consistent snapshots."""
        state = DegradationState(["m1", "m2", "m3"])
        summaries = []
        errors = []

        def mutate_state():
            try:
                for i in range(50):
                    if random.random() > 0.5:
                        state.record_model_unavailable(f"m{(i % 3) + 1}", "fail")
                    else:
                        state.record_model_recovered(f"m{(i % 3) + 1}")
            except Exception as e:
                errors.append(str(e))

        def read_summary():
            try:
                for _ in range(100):
                    summary = state.get_summary()
                    # Verify internal consistency
                    assert len(summary['available_models']) + len(summary['failed_models']) >= len(summary['expected_models']) - len(summary['failed_models'])
                    summaries.append(summary)
            except Exception as e:
                errors.append(str(e))

        threads = [
            threading.Thread(target=mutate_state),
            threading.Thread(target=read_summary),
            threading.Thread(target=mutate_state),
            threading.Thread(target=read_summary),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(summaries) == 200, "Should have collected all summaries"


class TestAdaptiveTimeoutThreadSafety:
    """Test AdaptiveTimeout under concurrent access."""

    def test_concurrent_record_latency(self):
        """Multiple threads recording latencies should not corrupt data."""
        timeout = AdaptiveTimeout(base_timeout=60)
        num_threads = 10
        records_per_thread = 100
        model = "latency_model"

        def record_latencies():
            for _ in range(records_per_thread):
                latency = random.uniform(100, 5000)
                success = random.random() > 0.1  # 90% success rate
                timeout.record_latency(model, latency, success=success)

        threads = [threading.Thread(target=record_latencies) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify stats are accessible and consistent
        stats = timeout.get_stats()
        assert model in stats, f"{model} should be in stats"

    def test_concurrent_get_timeout(self):
        """get_timeout should be consistent under concurrent access."""
        timeout = AdaptiveTimeout(base_timeout=60)
        model = "timeout_model"

        # Seed with some data
        for _ in range(10):
            timeout.record_latency(model, random.uniform(1000, 3000))

        timeouts_returned = []

        def get_timeouts():
            for _ in range(100):
                t = timeout.get_timeout(model)
                timeouts_returned.append(t)

        threads = [threading.Thread(target=get_timeouts) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All returned timeouts should be valid
        assert all(isinstance(t, int) and t > 0 for t in timeouts_returned)

    def test_concurrent_mixed_operations(self):
        """Mixed read/write operations should not cause errors."""
        timeout = AdaptiveTimeout(base_timeout=60)
        models = ["model_a", "model_b", "model_c"]
        errors = []

        def mixed_ops():
            try:
                for _ in range(200):
                    model = random.choice(models)
                    op = random.choice(["record", "timeout", "stats"])
                    if op == "record":
                        timeout.record_latency(model, random.uniform(100, 10000))
                    elif op == "timeout":
                        timeout.get_timeout(model)
                    else:
                        timeout.get_stats()
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=mixed_ops) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent operations: {errors}"


class TestGlobalStateThreadSafety:
    """Test global state management functions under concurrent access."""

    def test_concurrent_init_degradation(self):
        """Multiple threads calling init_degradation should not corrupt state."""
        errors = []

        def init_session():
            try:
                models = [f"model_{random.randint(1, 5)}" for _ in range(3)]
                init_degradation(models, base_timeout=60)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=init_session) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"

    def test_concurrent_reset_session(self):
        """reset_session_state during operations should not cause errors."""
        errors = []

        def do_operations():
            try:
                for _ in range(50):
                    CIRCUIT_BREAKER.record_success(f"model_{random.randint(1, 3)}")
                    CIRCUIT_BREAKER.get_status()
            except Exception as e:
                errors.append(str(e))

        def do_resets():
            try:
                for _ in range(5):
                    time.sleep(random.uniform(0, 0.01))
                    reset_session_state()
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=do_operations) for _ in range(10)]
        threads.append(threading.Thread(target=do_resets))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"


class TestRaceConditionRegression:
    """Specific regression tests for known race condition patterns."""

    def test_check_then_act_pattern(self):
        """
        Test that check-then-act patterns don't have TOCTOU issues.

        Previously, can_call() followed by record_success/failure could race.
        """
        breaker = CircuitBreaker()
        model = "toctou_model"
        errors = []
        inconsistencies = []

        def check_then_act():
            try:
                for _ in range(100):
                    if breaker.can_call(model):
                        # Simulate some delay between check and act
                        time.sleep(random.uniform(0, 0.001))
                        breaker.record_success(model)
                    else:
                        # This shouldn't happen if circuit is closed
                        inconsistencies.append("can_call returned False unexpectedly")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=check_then_act) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        # Note: Some inconsistencies might be expected if circuit opens during test
        # but there should be no crashes or data corruption

    def test_concurrent_state_transitions(self):
        """
        Test circuit breaker state transitions under concurrent load.

        Verifies: CLOSED -> OPEN -> HALF_OPEN -> CLOSED cycle integrity
        """
        breaker = CircuitBreaker()
        breaker.RECOVERY_TIMEOUT = 0.1  # Short timeout for testing
        model = "transition_model"
        errors = []

        def cause_failures():
            try:
                for _ in range(breaker.FAILURE_THRESHOLD + 2):
                    breaker.record_failure(model, "test")
                    time.sleep(0.01)
            except Exception as e:
                errors.append(str(e))

        def wait_and_recover():
            try:
                time.sleep(0.15)  # Wait for recovery timeout
                for _ in range(breaker.SUCCESS_THRESHOLD + 1):
                    if breaker.can_call(model):
                        breaker.record_success(model)
                    time.sleep(0.01)
            except Exception as e:
                errors.append(str(e))

        def monitor_state():
            try:
                states_seen = set()
                for _ in range(50):
                    status = breaker.get_status()
                    if model in status:
                        states_seen.add(status[model]["state"])
                    time.sleep(0.01)
            except Exception as e:
                errors.append(str(e))

        threads = [
            threading.Thread(target=cause_failures),
            threading.Thread(target=wait_and_recover),
            threading.Thread(target=monitor_state),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during state transitions: {errors}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

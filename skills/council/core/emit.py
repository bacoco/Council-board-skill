"""
Event emission for Council deliberation.

Provides structured event output with:
- Automatic secret redaction
- Human-readable formatting option
- Performance instrumentation
"""

import json
import time
import textwrap
from typing import Optional

# Import security module for redaction
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from security.input_validator import InputValidator

# Global InputValidator instance for security
INPUT_VALIDATOR = InputValidator()

# Output mode flags (set by main entry point)
HUMAN_OUTPUT = False  # Set to True for user-friendly CLI output
ENABLE_PERF_INSTRUMENTATION = False  # Set to True to emit timing metrics


def set_output_mode(human: bool = False, perf: bool = False):
    """Configure output mode flags."""
    global HUMAN_OUTPUT, ENABLE_PERF_INSTRUMENTATION
    HUMAN_OUTPUT = human
    ENABLE_PERF_INSTRUMENTATION = perf


def emit(event: dict):
    """Emit event with automatic secret redaction."""
    event['ts'] = int(time.time())
    # Redact secrets from output before emission
    redacted_event = INPUT_VALIDATOR.redact_output(event)

    if HUMAN_OUTPUT:
        # Human-readable output for CLI users
        _emit_human(redacted_event)
    else:
        print(json.dumps(redacted_event), flush=True)


def _emit_human(event: dict):
    """Format event as human-readable output."""
    event_type = event.get('type', '')

    # Status messages - check specific cases first, then generic fallback
    if event_type == 'status' and 'review' in event.get('msg', '').lower():
        print(f"📝 Peer review in progress...", flush=True)

    elif event_type == 'status' and 'synthesiz' in event.get('msg', '').lower():
        print(f"🧠 Synthesizing final answer...", flush=True)

    elif event_type == 'status':
        msg = event.get('msg', '')
        print(f"📋 {msg}", flush=True)

    # Round progress
    elif event_type == 'round_start':
        round_num = event.get('round', 1)
        max_rounds = event.get('max_rounds', 3)
        print(f"\n🔄 Round {round_num}/{max_rounds}", flush=True)

    elif event_type == 'round_complete':
        print(f"   ✓ Round complete", flush=True)

    # Degraded operation warning
    elif event_type == 'degraded_start':
        available = event.get('available', [])
        print(f"⚠️  Degraded mode: {len(available)} model(s) available - confidence penalty applies", flush=True)

    # Persona generation
    elif event_type == 'persona_generation':
        print(f"🎭 Generating personas...", flush=True)

    elif event_type == 'persona_generation_success':
        personas = event.get('personas', [])
        print(f"   ✓ Personas: {', '.join(personas)}", flush=True)

    # Model calls
    elif event_type == 'opinion_start':
        model = event.get('model', 'unknown')
        persona = event.get('persona', model)
        print(f"   🤖 {model.upper()} ({persona}) thinking...", flush=True)

    elif event_type == 'opinion_complete':
        model = event.get('model', 'unknown')
        latency = event.get('latency_ms', 0)
        print(f"   ✅ {model.upper()} responded ({latency/1000:.1f}s)", flush=True)

    elif event_type == 'opinion_error':
        model = event.get('model', 'unknown')
        error = event.get('error', 'unknown error')
        print(f"   ❌ {model.upper()} failed: {error[:50]}", flush=True)

    elif event_type == 'opinion_skip':
        model = event.get('model', 'unknown')
        reason = event.get('reason', 'unknown')
        print(f"   ⏭️  {model.upper()} skipped: {reason}", flush=True)

    # Trail saved
    elif event_type == 'trail_saved':
        path = event.get('path', '')
        print(f"\n📄 Trail saved: {path}", flush=True)

    # Final answer
    elif event_type == 'final':
        confidence = event.get('confidence', 0)
        answer = event.get('answer', '')
        print(f"\n{'='*60}", flush=True)
        print(f"🎯 COUNCIL ANSWER (confidence: {confidence:.0%})", flush=True)
        print(f"{'='*60}", flush=True)
        # Wrap answer to ~80 chars
        wrapped = textwrap.fill(answer, width=78)
        print(wrapped, flush=True)
        print(f"{'='*60}\n", flush=True)

    # Degradation/errors
    elif event_type == 'error':
        msg = event.get('msg', '')
        print(f"⚠️  Error: {msg}", flush=True)

    elif event_type == 'escalation_devils_advocate':
        print(f"\n😈 Escalating to Devil's Advocate mode...", flush=True)

    # Ignore technical events in human mode
    elif event_type in ('meta', 'metrics_summary', 'perf_metric', 'observability_event',
                        'degradation_status', 'convergence_check', 'score', 'contradiction'):
        pass  # Silent in human mode

    # Default: silent for unhandled events
    else:
        pass


def emit_perf_metric(func_name: str, elapsed_ms: float, **kwargs):
    """Emit performance metric event if instrumentation enabled."""
    if ENABLE_PERF_INSTRUMENTATION:
        emit({"type": "perf_metric", "function": func_name, "elapsed_ms": elapsed_ms, **kwargs})


def emit_perf_metrics(summary: dict):
    """Emit stage and round latency metrics when instrumentation is enabled."""
    if not ENABLE_PERF_INSTRUMENTATION:
        return

    emit({
        "type": "perf_metrics",
        "stage_latencies": summary.get("stage_latencies", {}),
        "round_latencies": summary.get("round_latencies", []),
        "model_stats": summary.get("model_stats", {})
    })

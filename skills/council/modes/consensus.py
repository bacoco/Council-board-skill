"""
Consensus mode for Council deliberation.

Multi-round deliberation with convergence detection.
Includes gather_opinions (shared by all modes) and run_council.
"""

import asyncio
import json
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING

# Import from parent package
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from persona_manager import Persona

from core.models import LLMResponse, SessionConfig
from core.emit import emit, emit_perf_metric, emit_perf_metrics
from core.state import (
    CIRCUIT_BREAKER, DegradationLevel, DEFAULT_TIMEOUT, reset_session_state,
    init_degradation, get_degradation_state, get_adaptive_timeout,
    get_base_model
)
from core.metrics import init_metrics, get_metrics
from core.parsing import get_parsed_json
from core.convergence import check_convergence
from core.adapters import ADAPTERS, check_cli_available
from core.prompts import build_opinion_prompt, build_context_from_previous_rounds
from core.personas import generate_personas_with_llm
from core.review import peer_review, extract_contradictions
from core.synthesis import synthesize
from core.trail import generate_trail_markdown, save_trail_to_file

from modes.devil import summarize_devils_advocate_arguments

if TYPE_CHECKING:
    pass


# Timeout for model calls - same for all since they run in parallel
MODEL_TIMEOUT = DEFAULT_TIMEOUT


async def gather_opinions(
    config: SessionConfig,
    round_num: int = 1,
    previous_round_opinions: dict = None,
    include_personas: bool = False,
    excluded_models: list = None
) -> Union[Dict[str, LLMResponse], Tuple[Dict[str, LLMResponse], Dict[str, Persona]]]:
    """
    Gather opinions from all available models in parallel with graceful degradation.

    Args:
        config: Session configuration
        round_num: Current round number (1-indexed)
        previous_round_opinions: Opinions from previous round for context
        include_personas: If True, also return persona map
        excluded_models: List to track skipped/failed models

    Returns:
        Dict of model -> LLMResponse, optionally with persona map
    """
    emit({"type": "status", "stage": 1, "msg": f"Collecting opinions (Round {round_num}, Mode: {config.mode})..."})

    # Get graceful degradation managers
    degradation = get_degradation_state()
    adaptive_timeout = get_adaptive_timeout()

    # Always generate personas dynamically for ALL modes via LLM (with caching + rotation)
    assigned_personas = await generate_personas_with_llm(
        config.query,
        len(config.models),
        config.chairman,
        mode=config.mode,
        timeout=30,
        round_num=round_num  # Rotate personas between rounds
    )

    tasks = []
    available_models = []
    model_timeouts = []  # Track timeout per model for adaptive timeout
    model_index = 0
    persona_map: dict[str, Persona] = {}

    for model_instance in config.models:
        # Extract base model from instance ID (e.g., 'claude_instance_1' -> 'claude')
        base_model = get_base_model(model_instance)

        # Check circuit breaker before including model
        if not CIRCUIT_BREAKER.can_call(base_model):
            emit({"type": "opinion_skip", "model": model_instance, "reason": "circuit_breaker_open"})
            if degradation:
                degradation.record_model_unavailable(model_instance, "circuit_breaker_open")
            if excluded_models is not None:
                excluded_models.append({"model": model_instance, "round": round_num, "reason": "circuit_breaker_open", "status": "SKIPPED"})
            continue

        if base_model in ADAPTERS and check_cli_available(base_model):
            available_models.append(model_instance)

            # Get dynamic persona for this model index
            dynamic_persona = None
            if assigned_personas and model_index < len(assigned_personas):
                dynamic_persona = assigned_personas[model_index]
                persona_map[model_instance] = dynamic_persona

            # Build context from previous round (what OTHER models said)
            previous_context = None
            if previous_round_opinions:
                previous_context = build_context_from_previous_rounds(model_instance, previous_round_opinions, config.anonymize, config.mode)

            # Build prompt with persona, previous context, and code context
            prompt = build_opinion_prompt(
                config.query,
                model=model_instance,
                round_num=round_num,
                previous_context=previous_context,
                mode=config.mode,
                code_context=config.context,
                dynamic_persona=dynamic_persona
            )

            # Get persona title for logging (always from dynamic_persona now)
            persona_title = dynamic_persona.title if dynamic_persona else model_instance

            # Use config.timeout from user/config file (default: 420s for Codex tool exploration)
            # All models share same timeout since they run in parallel (wait = slowest).
            model_timeout = config.timeout
            model_timeouts.append(model_timeout)

            emit({"type": "opinion_start", "model": model_instance, "round": round_num, "persona": persona_title, "timeout": model_timeout})
            tasks.append(ADAPTERS[base_model](prompt, model_timeout))
            model_index += 1
        else:
            emit({"type": "opinion_error", "model": model_instance, "error": "CLI not available", "status": "ABSTENTION"})
            if degradation:
                degradation.record_model_unavailable(model_instance, "cli_not_available")
            if excluded_models is not None:
                excluded_models.append({"model": model_instance, "round": round_num, "reason": "cli_not_available", "status": "SKIPPED"})

    results = await asyncio.gather(*tasks, return_exceptions=True)

    responses = {}
    for model_instance, result in zip(available_models, results):
        base_model = get_base_model(model_instance)

        if isinstance(result, Exception):
            emit({"type": "opinion_error", "model": model_instance, "error": str(result), "status": "ABSTENTION"})
            responses[model_instance] = LLMResponse(content='', model=model_instance, latency_ms=0, success=False, error=str(result))
            # Track failure in degradation state
            if degradation:
                degradation.record_model_unavailable(model_instance, str(result))
            if excluded_models is not None:
                excluded_models.append({"model": model_instance, "round": round_num, "reason": str(result), "status": "FAILED"})
            if adaptive_timeout:
                adaptive_timeout.record_latency(base_model, 0, success=False, mode=config.mode)
        else:
            # Record latency for adaptive timeout learning
            if adaptive_timeout:
                adaptive_timeout.record_latency(base_model, result.latency_ms, success=result.success, mode=config.mode)

            if result.success:
                emit({"type": "opinion_complete", "model": model_instance, "round": round_num, "latency_ms": result.latency_ms})
                # Check if model recovered (was previously failed)
                if degradation and model_instance in degradation.failed_models:
                    degradation.record_model_recovered(model_instance)
            else:
                emit({"type": "opinion_error", "model": model_instance, "error": result.error, "status": "ABSTENTION"})
                if degradation:
                    degradation.record_model_unavailable(model_instance, result.error or "unknown_error")
                if excluded_models is not None:
                    excluded_models.append({"model": model_instance, "round": round_num, "reason": result.error or "unknown_error", "status": "FAILED"})
            responses[model_instance] = result

    # Emit degradation status if any models failed
    if degradation and degradation.level != DegradationLevel.FULL:
        emit({
            "type": "degradation_status",
            "level": degradation.level,
            "available_models": list(degradation.available_models),
            "failed_models": degradation.failed_models,
            "round": round_num
        })

    if include_personas:
        return responses, persona_map

    return responses


async def run_council(config: SessionConfig, escalation_allowed: bool = True) -> dict:
    """
    Run multi-round consensus deliberation.

    Orchestrates the full deliberation pipeline:
    1. Multi-round opinion gathering with convergence detection
    2. Devil's advocate escalation if needed
    3. Peer review
    4. Chairman synthesis

    Args:
        config: Session configuration
        escalation_allowed: If True, may escalate to devil's advocate mode

    Returns:
        Complete deliberation result with synthesis, metrics, and trail
    """
    session_id = f"council-{int(time.time())}"
    start_time = time.time()

    # Reset ALL global state for new session to prevent cross-session contamination
    reset_session_state()

    # Enable or disable perf instrumentation for this session
    # Note: We use the module-level variable from emit
    if config.enable_perf_metrics:
        emit({"type": "perf_instrumentation_enabled", "session_id": session_id})

    # Initialize observability metrics
    metrics = init_metrics(session_id)

    # Initialize graceful degradation tracking
    degradation, adaptive_timeout = init_degradation(config.models, config.timeout, config.min_quorum)

    emit({"type": "status", "stage": 0, "msg": f"Starting council session {session_id} (mode: {config.mode}, max_rounds: {config.max_rounds})"})

    # Track all rounds
    all_rounds = []
    deliberation_trail = []  # Detailed trail for --trail flag
    excluded_models = []  # Track skipped/failed models for trail visibility
    previous_round_opinions = None
    converged = False
    convergence_score = 0.0
    devils_advocate_round = None
    devils_advocate_summary = None

    # Multi-round deliberation loop
    for round_num in range(1, config.max_rounds + 1):
        round_start = time.time()
        emit({"type": "round_start", "round": round_num, "max_rounds": config.max_rounds})

        # Stage 1: Gather opinions for this round (always include personas for trail)
        gather_result = await gather_opinions(config, round_num=round_num, previous_round_opinions=previous_round_opinions, include_personas=True, excluded_models=excluded_models)

        # Unpack responses and persona map
        if isinstance(gather_result, tuple):
            responses, persona_map = gather_result
        else:
            responses = gather_result
            persona_map = {}

        # Record model latencies from responses
        for model, response in responses.items():
            metrics.record_model_latency(model, response.latency_ms, success=response.success)
            metrics.record_latency('model_call', response.latency_ms)

        # Check quorum
        valid_count = sum(1 for r in responses.values() if r.success)
        if valid_count < degradation.min_quorum:
            metrics.emit_event('QuorumNotMet', {'round': round_num, 'required': degradation.min_quorum, 'got': valid_count})
            emit({"type": "error", "msg": f"Quorum not met in round {round_num} (need >= {degradation.min_quorum} valid responses)"})
            if round_num == 1:
                metrics_summary = metrics.get_summary()
                metrics.emit_summary()
                emit_perf_metrics(metrics_summary)
                return {"error": "Quorum not met in initial round", "metrics": metrics_summary}
            else:
                emit({"type": "warning", "msg": f"Quorum failed in round {round_num}, using previous round data"})
                break

        opinions = {m: r.content for m, r in responses.items() if r.success}
        all_rounds.append(opinions)

        # Build trail entries for this round
        for model, response in responses.items():
            if response.success:
                persona = persona_map.get(model)
                # Extract answer and confidence from parsed JSON
                parsed = get_parsed_json(response) if response.content else {}
                if isinstance(parsed, dict):
                    # Extract the actual answer, not the raw JSON
                    answer = parsed.get('answer', parsed.get('final_answer', ''))
                    confidence = parsed.get('confidence', 0.0)
                    key_points = parsed.get('key_points', [])
                else:
                    answer = response.content
                    confidence = 0.0
                    key_points = []

                trail_entry = {
                    "round": round_num,
                    "model": model,
                    "persona": persona.title if persona else model,
                    "persona_role": persona.role if persona else None,
                    "answer": answer,
                    "key_points": key_points if key_points else None,
                    "confidence": confidence,
                    "latency_ms": response.latency_ms
                }
                deliberation_trail.append(trail_entry)

        # Check convergence after round 2+
        if round_num > 1:
            converged, convergence_score = check_convergence(all_rounds, threshold=config.convergence_threshold)
            emit({
                "type": "convergence_check",
                "round": round_num,
                "converged": converged,
                "score": round(convergence_score, 3)
            })

            if converged:
                metrics.emit_event('ConvergenceAchieved', {'round': round_num, 'score': convergence_score})
                emit({"type": "status", "msg": f"Convergence achieved at round {round_num} (score: {convergence_score:.3f})"})
                break

        # Store for next round
        previous_round_opinions = opinions

        # Record round completion
        round_latency = int((time.time() - round_start) * 1000)
        metrics.record_round(round_num, round_latency)
        emit({"type": "round_complete", "round": round_num})

    # Trigger devil's advocate mini-cycle if needed (no convergence after max rounds with quorum)
    should_run_devils_advocate = (
        escalation_allowed
        and not converged
        and all_rounds
        and len(all_rounds) == config.max_rounds
        and config.mode != 'devil_advocate'
    )

    if should_run_devils_advocate:
        emit({
            "type": "escalation_devils_advocate",
            "reason": "Max rounds reached without convergence",
            "rounds_completed": len(all_rounds),
            "convergence_score": round(convergence_score, 3)
        })

        last_round_context = json.dumps(all_rounds[-1], indent=2)
        combined_context = (config.context + "\n\n" if config.context else "") + f"Last round opinions (context for devil's advocate):\n{last_round_context}"
        devils_config = SessionConfig(
            query=config.query,
            mode='devil_advocate',
            models=config.models,
            chairman=config.chairman,
            timeout=config.timeout,
            anonymize=config.anonymize,
            council_budget=config.council_budget,
            output_level=config.output_level,
            max_rounds=1,
            context=combined_context
        )

        devils_start = time.time()
        devils_responses, devils_personas = await gather_opinions(
            devils_config,
            round_num=1,
            previous_round_opinions=all_rounds[-1],
            include_personas=True
        )
        metrics.record_latency('devils_advocate_round', int((time.time() - devils_start) * 1000))

        valid_devils = sum(1 for r in devils_responses.values() if r.success)
        if valid_devils >= degradation.min_quorum:
            devils_advocate_round = {m: r.content for m, r in devils_responses.items() if r.success}
            devils_advocate_summary = await summarize_devils_advocate_arguments(
                config.query,
                devils_advocate_round,
                devils_personas,
                config.chairman,
                config.timeout
            )
        else:
            emit({"type": "warning", "msg": "Devil's advocate mini-cycle aborted due to insufficient valid responses"})

    # Stage 2: Peer review (on final round)
    peer_review_start = time.time()
    final_opinions = all_rounds[-1] if all_rounds else {}
    review_result = await peer_review(config, final_opinions)
    review = review_result.get('review', {})
    metrics.record_latency('peer_review', int((time.time() - peer_review_start) * 1000))

    # Stage 2.5: Extract contradictions
    conflicts = extract_contradictions(review)
    if conflicts:
        for c in conflicts:
            emit({"type": "contradiction", "conflict": c, "severity": "medium"})

    # Stage 3: Synthesis (with all rounds context)
    synthesis_start = time.time()
    synthesis_result = await synthesize(
        config,
        final_opinions,
        review,
        conflicts,
        all_rounds=all_rounds,
        devils_advocate_summary=devils_advocate_summary
    )
    metrics.record_latency('synthesis', int((time.time() - synthesis_start) * 1000))

    duration_ms = int((time.time() - start_time) * 1000)

    # Get degradation state for final adjustments
    degradation_summary = degradation.get_summary() if degradation else None

    # Adjust confidence based on degradation level
    raw_confidence = synthesis_result.get('confidence', 0.0)
    adjusted_confidence = degradation.adjust_confidence(raw_confidence) if degradation else raw_confidence

    # Emit partial result warning if operating in degraded mode
    if degradation and degradation.level != DegradationLevel.FULL:
        metrics.emit_event('PartialResultReturned', {
            'degradation_level': degradation.level,
            'raw_confidence': raw_confidence,
            'adjusted_confidence': adjusted_confidence,
            'failed_models': degradation.failed_models
        })

    # Build trail output: save to Markdown file if enabled
    trail_output = None
    trail_file_path = None
    if config.enable_trail and deliberation_trail:
        # Generate Markdown content
        markdown_content = generate_trail_markdown(
            session_id=session_id,
            query=config.query,
            mode=config.mode,
            deliberation_trail=deliberation_trail,
            synthesis=synthesis_result,
            review=review,
            devils_advocate_summary=devils_advocate_summary,
            duration_ms=duration_ms,
            converged=converged,
            convergence_score=convergence_score,
            confidence=adjusted_confidence,
            excluded_models=excluded_models,
            config_models=config.models
        )

        # Save to file (relative to current working directory, not skill folder)
        trail_dir = Path.cwd() / "council_trails"
        trail_file_path = save_trail_to_file(
            markdown_content=markdown_content,
            session_id=session_id,
            query=config.query,
            mode=config.mode,
            output_dir=str(trail_dir)
        )

        emit({
            "type": "trail_saved",
            "path": str(trail_file_path),
            "size_bytes": trail_file_path.stat().st_size
        })

        # Trail output: just the path + metadata (not the full trail)
        trail_output = {
            "trail_file": str(trail_file_path),
            "trail_metadata": {
                "total_rounds": len(all_rounds),
                "participants": len(set(e["model"] for e in deliberation_trail)),
                "total_contributions": len(deliberation_trail),
                "consensus_reached": converged
            }
        }

    # Final output
    final_emit = {
        "type": "final",
        "answer": synthesis_result.get('final_answer', ''),
        "confidence": adjusted_confidence,
        "raw_confidence": raw_confidence,
        "degradation_level": degradation.level if degradation else DegradationLevel.FULL,
        "dissent": synthesis_result.get('dissenting_view'),
        "rounds_completed": len(all_rounds),
        "converged": converged,
        "convergence_score": round(convergence_score, 3),
        "devils_advocate_summary": devils_advocate_summary
    }
    if trail_output:
        final_emit.update(trail_output)
    emit(final_emit)

    # Get circuit breaker status for transparency
    cb_status = CIRCUIT_BREAKER.get_status()

    # Get adaptive timeout stats
    timeout_stats = adaptive_timeout.get_stats() if adaptive_timeout else None

    # Emit metrics summary
    metrics_summary = metrics.get_summary()
    metrics.emit_summary()
    emit_perf_metrics(metrics_summary)

    emit({
        "type": "meta",
        "session_id": session_id,
        "duration_ms": duration_ms,
        "models_responded": list(final_opinions.keys()),
        "mode": config.mode,
        "rounds": len(all_rounds),
        "converged": converged,
        "circuit_breaker": cb_status if cb_status else None,
        "degradation": degradation_summary,
        "adaptive_timeout": timeout_stats,
        "devils_advocate_summary": devils_advocate_summary
    })

    result = {
        "session_id": session_id,
        "synthesis": synthesis_result,
        "all_rounds": all_rounds,
        "final_opinions": final_opinions,
        "review": review,
        "conflicts": conflicts,
        "duration_ms": duration_ms,
        "rounds_completed": len(all_rounds),
        "converged": converged,
        "convergence_score": convergence_score,
        "confidence": adjusted_confidence,
        "raw_confidence": raw_confidence,
        "degradation": degradation_summary,
        "circuit_breaker_status": cb_status,
        "adaptive_timeout_stats": timeout_stats,
        "devils_advocate_summary": devils_advocate_summary,
        "devils_advocate_round": devils_advocate_round,
        "metrics": metrics.get_summary()
    }
    if trail_output:
        result.update(trail_output)
    return result

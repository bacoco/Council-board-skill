"""
Vote mode for Council deliberation.

Fast-path voting where each model casts a weighted vote with justification.
Single round, parallel execution, immediate tally.
"""

import asyncio
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple, TYPE_CHECKING

# Import from parent package
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import LLMResponse, SessionConfig, VoteBallot, VoteResult
from core.emit import emit, emit_perf_metrics
from core.state import (
    CIRCUIT_BREAKER, DegradationLevel, DEFAULT_TIMEOUT, reset_session_state,
    init_degradation, get_degradation_state, get_adaptive_timeout,
    get_base_model
)
from core.metrics import init_metrics
from core.parsing import get_parsed_json
from core.adapters import ADAPTERS, check_cli_available
from core.prompts import build_vote_prompt, build_vote_synthesis_prompt
from core.personas import generate_personas_with_llm

if TYPE_CHECKING:
    pass


# Timeout for model calls
MODEL_TIMEOUT = DEFAULT_TIMEOUT


def validate_vote(vote_data: dict, model: str) -> Tuple[bool, str, dict]:
    """
    Validate and normalize vote data.

    Args:
        vote_data: Raw vote data from model
        model: Model name (for error messages)

    Returns:
        Tuple of (is_valid, error_message, normalized_data)
    """
    # Extract and validate vote
    vote = vote_data.get('vote', '')
    if not vote or not isinstance(vote, str):
        return False, "Missing or invalid vote field", {}

    # Normalize vote (strip whitespace, handle common variations)
    vote = vote.strip()
    if not vote:
        return False, "Empty vote", {}

    # Extract and clamp confidence to [0, 1]
    raw_confidence = vote_data.get('confidence', 0.5)
    try:
        confidence = float(raw_confidence)
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.5

    # Weight is separate from confidence - default to 1.0 (equal voting power)
    # In future, this could be configured per-model based on domain expertise
    raw_weight = vote_data.get('weight', 1.0)
    try:
        weight = float(raw_weight)
        weight = max(0.0, min(1.0, weight))
    except (TypeError, ValueError):
        weight = 1.0

    # For now, use confidence as weight modifier (weight * confidence)
    effective_weight = weight * confidence

    normalized = {
        'vote': vote,
        'confidence': confidence,
        'weight': weight,
        'effective_weight': effective_weight,
        'justification': str(vote_data.get('justification', '')),
        'alternative_considered': vote_data.get('alternative_considered'),
        'would_veto': bool(vote_data.get('would_veto', False))
    }

    return True, "", normalized


async def collect_votes(config: SessionConfig) -> List[VoteBallot]:
    """
    Collect votes from all models in parallel.

    Each model casts a weighted vote with justification.
    Uses explicit (model, task) pairing for robustness.

    Args:
        config: Session configuration

    Returns:
        List of validated VoteBallot objects
    """
    emit({"type": "status", "stage": 1, "msg": "Collecting votes from council members..."})

    # Get graceful degradation managers
    degradation = get_degradation_state()
    adaptive_timeout = get_adaptive_timeout()

    # First, determine which models are actually available
    available_models = []
    for model_instance in config.models:
        base_model = get_base_model(model_instance)

        # Check circuit breaker before including model
        if not CIRCUIT_BREAKER.can_call(base_model):
            emit({"type": "vote_skip", "model": model_instance, "reason": "circuit_breaker_open"})
            if degradation:
                degradation.record_model_unavailable(model_instance, "circuit_breaker_open")
            continue

        if base_model in ADAPTERS and check_cli_available(base_model):
            available_models.append(model_instance)
        else:
            if degradation:
                degradation.record_model_unavailable(model_instance, "cli_not_available")

    if not available_models:
        emit({"type": "error", "msg": "No models available for voting"})
        return []

    # Generate personas only for available models
    assigned_personas = await generate_personas_with_llm(
        config.query,
        len(available_models),  # Only generate for available models
        config.chairman,
        mode='vote',
        timeout=30
    )

    # Build tasks with explicit model association
    task_model_pairs = []

    for idx, model_instance in enumerate(available_models):
        base_model = get_base_model(model_instance)

        # Get persona for this voter (safe indexing)
        dynamic_persona = assigned_personas[idx] if idx < len(assigned_personas) else None
        persona_title = dynamic_persona.title if dynamic_persona else model_instance

        # Use config.timeout from user/config file (default: 420s for Codex tool exploration)
        model_timeout = config.timeout

        emit({"type": "vote_start", "model": model_instance, "persona": persona_title, "timeout": model_timeout})

        prompt = build_vote_prompt(
            config.query,
            options=None,  # Let models determine options from query
            dynamic_persona=dynamic_persona,
            code_context=config.context
        )

        # Store (model, task) pair for robust result matching
        task = ADAPTERS[base_model](prompt, model_timeout)
        task_model_pairs.append((model_instance, task))

    # Execute votes in parallel
    tasks = [pair[1] for pair in task_model_pairs]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    ballots = []
    for (model_instance, _), result in zip(task_model_pairs, results):
        base_model = get_base_model(model_instance)

        if isinstance(result, Exception):
            emit({"type": "vote_error", "model": model_instance, "error": str(result)})
            if degradation:
                degradation.record_model_unavailable(model_instance, str(result))
            if adaptive_timeout:
                adaptive_timeout.record_latency(base_model, 0, success=False, mode=config.mode)
            continue

        if not result.success:
            emit({"type": "vote_error", "model": model_instance, "error": result.error})
            if degradation:
                degradation.record_model_unavailable(model_instance, result.error or "unknown_error")
            if adaptive_timeout:
                adaptive_timeout.record_latency(base_model, result.latency_ms, success=False, mode=config.mode)
            continue

        # Record successful latency for adaptive timeout
        if adaptive_timeout:
            adaptive_timeout.record_latency(base_model, result.latency_ms, success=True, mode=config.mode)

        try:
            vote_data = get_parsed_json(result)

            # Validate and normalize the vote
            is_valid, error_msg, normalized = validate_vote(vote_data, model_instance)

            if not is_valid:
                emit({"type": "vote_validation_error", "model": model_instance, "error": error_msg})
                continue

            ballot = VoteBallot(
                model=model_instance,
                vote=normalized['vote'],
                weight=normalized['effective_weight'],  # Use effective weight (weight * confidence)
                justification=normalized['justification'],
                confidence=normalized['confidence'],
                latency_ms=result.latency_ms
            )
            ballots.append(ballot)

            emit({
                "type": "vote_cast",
                "model": model_instance,
                "vote": ballot.vote,
                "confidence": round(ballot.confidence, 2),
                "effective_weight": round(ballot.weight, 2),
                "latency_ms": result.latency_ms
            })

        except Exception as e:
            emit({"type": "vote_parse_error", "model": model_instance, "error": str(e)})

    # Emit degradation status if any models failed
    if degradation and degradation.level != DegradationLevel.FULL:
        emit({
            "type": "degradation_status",
            "level": degradation.level,
            "available_models": list(degradation.available_models),
            "failed_models": degradation.failed_models,
            "stage": "vote_collection"
        })

    return ballots


def tally_votes(ballots: List[VoteBallot]) -> Tuple[dict, dict, str, bool, str]:
    """
    Tally votes with weighted scoring and proper tie-breaking.

    Tie-breaking cascade:
    1. Weighted score (primary)
    2. Raw vote count (if weighted scores within epsilon)
    3. Highest single confidence vote (deterministic tiebreaker)

    Args:
        ballots: List of VoteBallot objects

    Returns:
        Tuple of (vote_counts, weighted_scores, winner, tie_broken, tie_breaker_method)
    """
    # Filter out ABSTAIN votes from tallying
    valid_ballots = [b for b in ballots if b.vote.upper() != 'ABSTAIN']

    vote_counts = {}
    weighted_scores = {}
    max_confidence_by_option = {}  # Track highest confidence vote per option

    for ballot in valid_ballots:
        vote = ballot.vote
        # Clamp weight to [0, 1] range
        weight = max(0.0, min(1.0, ballot.weight))

        vote_counts[vote] = vote_counts.get(vote, 0) + 1
        weighted_scores[vote] = weighted_scores.get(vote, 0.0) + weight

        # Track max confidence for tie-breaking
        if vote not in max_confidence_by_option or ballot.confidence > max_confidence_by_option[vote]:
            max_confidence_by_option[vote] = ballot.confidence

    if not weighted_scores:
        return {}, {}, "NO_VOTES", False, None

    # Sort by weighted score (primary), then by vote count (secondary)
    sorted_options = sorted(
        weighted_scores.items(),
        key=lambda x: (-x[1], -vote_counts.get(x[0], 0))
    )

    winner = sorted_options[0][0]
    tie_broken = False
    tie_breaker_method = None

    # Check for tie scenarios
    TIE_EPSILON = 0.05  # 5% threshold for weighted score tie

    if len(sorted_options) >= 2:
        top_option, top_score = sorted_options[0]
        second_option, second_score = sorted_options[1]

        # Weighted scores are tied (within epsilon)
        if abs(top_score - second_score) < TIE_EPSILON:
            top_count = vote_counts.get(top_option, 0)
            second_count = vote_counts.get(second_option, 0)

            # Tie-breaker 1: Raw vote count
            if top_count != second_count:
                # Re-sort by vote count to get actual winner
                count_sorted = sorted(
                    [(opt, vote_counts.get(opt, 0)) for opt, _ in sorted_options[:2]],
                    key=lambda x: -x[1]
                )
                winner = count_sorted[0][0]
                tie_broken = True
                tie_breaker_method = "raw_vote_count"
            else:
                # Tie-breaker 2: Highest single confidence vote
                top_max_conf = max_confidence_by_option.get(top_option, 0)
                second_max_conf = max_confidence_by_option.get(second_option, 0)

                if top_max_conf != second_max_conf:
                    conf_sorted = sorted(
                        [(top_option, top_max_conf), (second_option, second_max_conf)],
                        key=lambda x: -x[1]
                    )
                    winner = conf_sorted[0][0]
                    tie_broken = True
                    tie_breaker_method = "highest_confidence"
                else:
                    # Tie-breaker 3: Alphabetical (deterministic fallback)
                    alpha_sorted = sorted([top_option, second_option])
                    winner = alpha_sorted[0]
                    tie_broken = True
                    tie_breaker_method = "alphabetical"

    return vote_counts, weighted_scores, winner, tie_broken, tie_breaker_method


async def run_vote_council(config: SessionConfig) -> dict:
    """
    Run Vote mode deliberation.

    Fast-path voting where each model casts a weighted vote.
    Single round, parallel execution, immediate tally.

    Args:
        config: Session configuration

    Returns:
        VoteResult with winner, counts, and synthesis
    """
    session_id = f"vote-{int(time.time())}"
    start_time = time.time()

    # Reset ALL global state for new session to prevent cross-session contamination
    reset_session_state()

    # Enable or disable perf instrumentation for this session
    if config.enable_perf_metrics:
        emit({"type": "perf_instrumentation_enabled", "session_id": session_id})

    # Initialize observability metrics
    metrics = init_metrics(session_id)

    # Initialize graceful degradation tracking
    degradation, adaptive_timeout = init_degradation(config.models, config.timeout, config.min_quorum)

    emit({"type": "status", "stage": 0, "msg": f"Starting vote session {session_id}"})

    # Stage 1: Collect votes in parallel
    vote_collection_start = time.time()
    ballots = await collect_votes(config)
    metrics.record_latency('vote_collection', int((time.time() - vote_collection_start) * 1000))

    # Record individual ballot latencies
    for ballot in ballots:
        metrics.record_model_latency(ballot.model, ballot.latency_ms, success=True)
        metrics.record_latency('model_call', ballot.latency_ms)

    # Check quorum
    if len(ballots) < degradation.min_quorum:
        metrics.emit_event('QuorumNotMet', {'required': degradation.min_quorum, 'got': len(ballots), 'mode': 'vote'})
        emit({"type": "error", "msg": f"Vote quorum not met (got {len(ballots)}, need {degradation.min_quorum})"})
        metrics_summary = metrics.get_summary()
        metrics.emit_summary()
        emit_perf_metrics(metrics_summary)
        return {"error": "Vote quorum not met", "ballots": len(ballots), "required": degradation.min_quorum, "metrics": metrics_summary}

    emit({"type": "quorum_met", "votes": len(ballots), "required": degradation.min_quorum})

    # Stage 2: Tally votes
    tally_start = time.time()
    vote_counts, weighted_scores, winner, tie_broken, tie_breaker_method = tally_votes(ballots)
    metrics.record_latency('vote_tally', int((time.time() - tally_start) * 1000))

    # Record tie-breaking event if applicable
    if tie_broken:
        metrics.emit_event('TieBroken', {'method': tie_breaker_method, 'winner': winner})

    # Calculate margin
    total_weight = sum(weighted_scores.values())
    winner_weight = weighted_scores.get(winner, 0)
    margin = (winner_weight / total_weight * 100) if total_weight > 0 else 0

    emit({
        "type": "vote_tally",
        "winner": winner,
        "vote_counts": vote_counts,
        "weighted_scores": {k: round(v, 3) for k, v in weighted_scores.items()},
        "margin": round(margin, 1),
        "tie_broken": tie_broken,
        "tie_breaker": tie_breaker_method
    })

    # Stage 3: Chairman synthesizes result
    synthesis_start = time.time()
    emit({"type": "status", "stage": 2, "msg": "Chairman synthesizing vote results..."})

    ballots_dict = [
        {"model": b.model, "vote": b.vote, "confidence": b.confidence, "justification": b.justification}
        for b in ballots
    ]

    synthesis_prompt = build_vote_synthesis_prompt(
        config.query,
        ballots_dict,
        vote_counts,
        {k: round(v, 3) for k, v in weighted_scores.items()},
        winner
    )

    # Default synthesis (used if chairman fails)
    synthesis = {
        "final_answer": f"Council votes: {winner}",
        "confidence": margin / 100,
        "synthesis_failed": False
    }
    synthesis_error = None

    if config.chairman in ADAPTERS and check_cli_available(config.chairman):
        result = await ADAPTERS[config.chairman](synthesis_prompt, config.timeout)
        if result.success:
            parsed = get_parsed_json(result)
            # Validate synthesis has required fields
            if parsed.get('final_answer'):
                synthesis = parsed
                synthesis['synthesis_failed'] = False
            else:
                synthesis_error = "Synthesis returned empty answer"
                synthesis['synthesis_failed'] = True
        else:
            synthesis_error = result.error
            synthesis['synthesis_failed'] = True
    else:
        synthesis_error = "Chairman not available"
        synthesis['synthesis_failed'] = True

    # Record synthesis latency
    metrics.record_latency('synthesis', int((time.time() - synthesis_start) * 1000))

    # Surface synthesis failure explicitly
    if synthesis.get('synthesis_failed'):
        metrics.emit_event('SynthesisFailed', {'error': synthesis_error})
        emit({
            "type": "synthesis_warning",
            "msg": "Chairman synthesis failed - using fallback",
            "error": synthesis_error,
            "fallback_answer": synthesis['final_answer']
        })

    duration_ms = int((time.time() - start_time) * 1000)

    # Get degradation state for final output
    degradation_summary = degradation.get_summary() if degradation else None

    # Adjust confidence based on degradation
    raw_confidence = synthesis.get('confidence', margin / 100)
    adjusted_confidence = degradation.adjust_confidence(raw_confidence) if degradation else raw_confidence

    # Build vote result
    vote_result = VoteResult(
        winning_option=winner,
        vote_counts=vote_counts,
        weighted_scores={k: round(v, 3) for k, v in weighted_scores.items()},
        total_votes=len(ballots),
        quorum_met=True,
        margin=round(margin, 1),
        ballots=ballots,
        tie_broken=tie_broken,
        tie_breaker_method=tie_breaker_method
    )

    # Emit metrics summary
    metrics_summary = metrics.get_summary()
    metrics.emit_summary()
    emit_perf_metrics(metrics_summary)

    # Emit degradation event if operating degraded
    if degradation and degradation.level != DegradationLevel.FULL:
        metrics.emit_event('PartialResultReturned', {
            'degradation_level': degradation.level,
            'mode': 'vote',
            'failed_models': degradation.failed_models
        })

    # Gather adaptive timeout stats for transparency
    timeout_stats = adaptive_timeout.get_stats() if adaptive_timeout else None

    # Final output
    emit({
        "type": "final",
        "mode": "vote",
        "winner": winner,
        "margin": round(margin, 1),
        "answer": synthesis.get('final_answer', ''),
        "confidence": adjusted_confidence,
        "raw_confidence": raw_confidence,
        "degradation_level": degradation.level if degradation else DegradationLevel.FULL,
        "recommendation_strength": synthesis.get('recommendation_strength', 'moderate'),
        "total_votes": len(ballots)
    })

    emit({
        "type": "meta",
        "session_id": session_id,
        "duration_ms": duration_ms,
        "models_voted": [b.model for b in ballots],
        "mode": "vote",
        "degradation": degradation_summary,
        "adaptive_timeout": timeout_stats
    })

    return {
        "session_id": session_id,
        "mode": "vote",
        "vote_result": asdict(vote_result),
        "synthesis": synthesis,
        "duration_ms": duration_ms,
        "confidence": adjusted_confidence,
        "raw_confidence": raw_confidence,
        "degradation": degradation_summary,
        "adaptive_timeout_stats": timeout_stats,
        "metrics": metrics.get_summary()
    }

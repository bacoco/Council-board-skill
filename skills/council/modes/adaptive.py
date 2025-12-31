"""
Adaptive cascade mode for Council deliberation.

Automatically escalates through modes based on convergence:
- Tier 1 (Fast Path): Consensus mode
- Tier 2 (Quality Gate): + Debate mode if convergence < 0.7
- Tier 3 (Adversarial Audit): + Devil's Advocate if still ambiguous
"""

import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Import from parent package
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import SessionConfig
from core.emit import emit
from core.synthesis import meta_synthesize

from modes.consensus import run_council

if TYPE_CHECKING:
    pass


async def run_adaptive_cascade(config: SessionConfig) -> dict:
    """
    Adaptive tiered cascade methodology - automatically escalates through modes based on convergence.

    Tier 1 (Fast Path): Consensus mode
    Tier 2 (Quality Gate): + Debate mode if convergence < 0.7
    Tier 3 (Adversarial Audit): + Devil's Advocate if still ambiguous

    Args:
        config: Session configuration

    Returns:
        Final result with meta-synthesis if multiple modes used
    """
    emit({"type": "cascade_start", "msg": "Starting adaptive cascade (Tier 1: Consensus)"})

    # Tier 1: Consensus mode (always runs first)
    consensus_config = SessionConfig(
        query=config.query,
        mode='consensus',
        models=config.models,
        chairman=config.chairman,
        timeout=config.timeout,
        anonymize=config.anonymize,
        council_budget=config.council_budget,
        output_level=config.output_level,
        max_rounds=config.max_rounds,
        enable_perf_metrics=config.enable_perf_metrics,
        context=config.context
    )

    consensus_result = await run_council(consensus_config)

    # Check if escalation needed
    convergence_score = consensus_result.get('convergence_score', 1.0)
    confidence = consensus_result.get('synthesis', {}).get('confidence', 1.0)

    # Tier 1 exit condition: High convergence (>= 0.7)
    if convergence_score >= 0.7:
        emit({
            "type": "cascade_complete",
            "tier": 1,
            "msg": f"Tier 1 sufficient (convergence: {convergence_score:.3f})",
            "modes_used": ["consensus"]
        })
        return consensus_result

    # Tier 2: Escalate to Debate mode
    emit({
        "type": "cascade_escalate",
        "from_tier": 1,
        "to_tier": 2,
        "reason": f"Low convergence ({convergence_score:.3f} < 0.7), escalating to debate mode"
    })

    debate_config = SessionConfig(
        query=config.query,
        mode='debate',
        models=config.models,
        chairman=config.chairman,
        timeout=config.timeout,
        anonymize=config.anonymize,
        council_budget=config.council_budget,
        output_level=config.output_level,
        max_rounds=config.max_rounds,
        enable_perf_metrics=config.enable_perf_metrics,
        context=config.context
    )

    debate_result = await run_council(debate_config)

    # Check if further escalation needed
    debate_convergence = debate_result.get('convergence_score', 1.0)
    debate_confidence = debate_result.get('synthesis', {}).get('confidence', 1.0)

    # Tier 2 exit condition: Reasonable convergence or high confidence
    if debate_convergence >= 0.6 or debate_confidence >= 0.85:
        emit({
            "type": "cascade_complete",
            "tier": 2,
            "msg": f"Tier 2 sufficient (debate convergence: {debate_convergence:.3f}, confidence: {debate_confidence:.3f})",
            "modes_used": ["consensus", "debate"]
        })

        # Meta-synthesize consensus + debate
        meta_result = await meta_synthesize(
            config.query,
            consensus_result,
            debate_result=debate_result,
            chairman=config.chairman,
            timeout=config.timeout
        )

        return {
            "session_type": "adaptive_cascade",
            "tiers_used": 2,
            "modes": ["consensus", "debate"],
            "consensus_result": consensus_result,
            "debate_result": debate_result,
            "meta_synthesis": meta_result,
            "final_answer": meta_result.get('final_answer', ''),
            "confidence": meta_result.get('confidence', 0.0)
        }

    # Tier 3: Escalate to Devil's Advocate mode
    emit({
        "type": "cascade_escalate",
        "from_tier": 2,
        "to_tier": 3,
        "reason": f"Persistent ambiguity (debate convergence: {debate_convergence:.3f}), escalating to devil's advocate"
    })

    devils_config = SessionConfig(
        query=config.query,
        mode='devil_advocate',
        models=config.models,
        chairman=config.chairman,
        timeout=config.timeout,
        anonymize=config.anonymize,
        council_budget=config.council_budget,
        output_level=config.output_level,
        max_rounds=config.max_rounds,
        enable_perf_metrics=config.enable_perf_metrics,
        context=config.context
    )

    devils_result = await run_council(devils_config)

    emit({
        "type": "cascade_complete",
        "tier": 3,
        "msg": "Full cascade complete (all 3 modes executed)",
        "modes_used": ["consensus", "debate", "devil's advocate"]
    })

    # Meta-synthesize all 3 modes
    meta_result = await meta_synthesize(
        config.query,
        consensus_result,
        debate_result=debate_result,
        devils_result=devils_result,
        chairman=config.chairman,
        timeout=config.timeout
    )

    return {
        "session_type": "adaptive_cascade",
        "tiers_used": 3,
        "modes": ["consensus", "debate", "devil's advocate"],
        "consensus_result": consensus_result,
        "debate_result": debate_result,
        "devils_result": devils_result,
        "meta_synthesis": meta_result,
        "final_answer": meta_result.get('final_answer', ''),
        "confidence": meta_result.get('confidence', 0.0)
    }

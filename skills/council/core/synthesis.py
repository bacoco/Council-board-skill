"""
Synthesis for Council deliberation.

Handles final answer synthesis and meta-synthesis for adaptive mode.
"""

import json
from typing import Optional, TYPE_CHECKING

from .emit import emit
from .parsing import get_parsed_json
from .prompts import build_synthesis_prompt
from .adapters import ADAPTERS, get_chairman_with_fallback

if TYPE_CHECKING:
    from .models import SessionConfig


async def synthesize(config: 'SessionConfig', opinions: dict, review: dict, conflicts: list, all_rounds: list = None, devils_advocate_summary: dict = None) -> dict:
    """
    Synthesize final answer from all opinions and review.

    The chairman integrates all perspectives, resolves contradictions,
    and produces the final answer with confidence score.

    Args:
        config: Session configuration
        opinions: Dict of model -> opinion content
        review: Peer review results
        conflicts: List of identified conflicts
        all_rounds: All opinion rounds for context
        devils_advocate_summary: Devil's advocate analysis if applicable

    Returns:
        Synthesis dict with final_answer, confidence, etc.
    """
    # Use chairman with failover chain (Council recommendation #1)
    actual_chairman = get_chairman_with_fallback(config.chairman)
    emit({"type": "status", "stage": 3, "msg": f"Chairman ({actual_chairman}) synthesizing..."})

    prompt = build_synthesis_prompt(
        config.query,
        opinions,
        review.get('scores', {}),
        conflicts,
        all_rounds=all_rounds,
        devils_summary=devils_advocate_summary
    )

    if actual_chairman in ADAPTERS:
        result = await ADAPTERS[actual_chairman](prompt, config.timeout)
        if result.success:
            synthesis = get_parsed_json(result)  # Use cached JSON parsing
            return synthesis

    return {"final_answer": "Synthesis failed", "confidence": 0.0}


async def meta_synthesize(query: str, consensus_result: dict, debate_result: dict = None, devils_result: dict = None, chairman: str = 'claude', timeout: int = 60) -> dict:
    """
    Meta-synthesis combining results from multiple deliberation modes.

    Used in adaptive mode when escalation occurs through multiple modes.

    Args:
        query: Original question
        consensus_result: Result from consensus mode (always present)
        debate_result: Result from debate mode (optional)
        devils_result: Result from devil's advocate mode (optional)
        chairman: Model to use for meta-synthesis
        timeout: Timeout in seconds

    Returns:
        Meta-synthesized result incorporating all modes
    """
    emit({"type": "status", "msg": "Meta-synthesizing results from multiple deliberation modes..."})

    modes_used = ["consensus"]
    if debate_result:
        modes_used.append("debate")
    if devils_result:
        modes_used.append("devil's advocate")

    prompt = f"""<s>You are Chairman conducting meta-synthesis of multiple deliberation modes.</s>

<original_question>{query}</original_question>

<consensus_mode_result>
Answer: {consensus_result.get('synthesis', {}).get('final_answer', '')}
Confidence: {consensus_result.get('synthesis', {}).get('confidence', 0.0)}
Convergence: {consensus_result.get('converged', False)} (score: {consensus_result.get('convergence_score', 0.0)})
Rounds: {consensus_result.get('rounds_completed', 0)}
</consensus_mode_result>

{f'''<debate_mode_result>
Answer: {debate_result.get('synthesis', {}).get('final_answer', '')}
Confidence: {debate_result.get('synthesis', {}).get('confidence', 0.0)}
FOR/AGAINST perspectives presented
Dissenting view: {debate_result.get('synthesis', {}).get('dissenting_view', 'None')}
</debate_mode_result>''' if debate_result else ''}

{f'''<devils_advocate_result>
Answer: {devils_result.get('synthesis', {}).get('final_answer', '')}
Confidence: {devils_result.get('synthesis', {}).get('confidence', 0.0)}
Red Team critiques vs Blue Team defenses
Purple Team integration: {devils_result.get('synthesis', {}).get('dissenting_view', 'None')}
</devils_advocate_result>''' if devils_result else ''}

<instructions>
Synthesize insights from all {len(modes_used)} deliberation modes.
Modes used: {', '.join(modes_used)}

Produce final meta-synthesis as JSON:
{{"final_answer": "Integrated answer from all modes",
"confidence": 0.90,
"modes_consulted": {modes_used},
"escalation_justified": true,
"key_insights_by_mode": {{"consensus": "...", "debate": "...", "devils_advocate": "..."}},
"remaining_uncertainties": []}}
</instructions>"""

    # Use chairman with failover chain (Council recommendation #1)
    actual_chairman = get_chairman_with_fallback(chairman)
    if actual_chairman in ADAPTERS:
        result = await ADAPTERS[actual_chairman](prompt, timeout)
        if result.success:
            return get_parsed_json(result)  # Use cached JSON parsing

    return {"final_answer": "Meta-synthesis failed", "confidence": 0.0}

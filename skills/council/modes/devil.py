"""
Devil's Advocate mode for Council deliberation.

Provides Red Team (attack), Blue Team (defend), Purple Team (synthesize) dynamics.
"""

import json
import time
from typing import Optional, TYPE_CHECKING

# Import from parent package
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from persona_manager import Persona

from core.emit import emit
from core.parsing import extract_json, get_parsed_json
from core.adapters import ADAPTERS, get_chairman_with_fallback
from core.metrics import get_metrics

if TYPE_CHECKING:
    pass


def infer_devils_team(persona: Optional[Persona]) -> str:
    """
    Infer devil's advocate team based on persona metadata.

    Args:
        persona: The persona assigned to a model

    Returns:
        Team assignment: 'attacker', 'defender', 'synthesizer', or 'unassigned'
    """
    if not persona:
        return "unassigned"

    text = f"{persona.title} {persona.role}".lower()
    if any(keyword in text for keyword in ("attack", "red team", "attacker")):
        return "attacker"
    if any(keyword in text for keyword in ("defend", "defender", "blue team")):
        return "defender"
    if "synth" in text or "purple" in text or "integrat" in text:
        return "synthesizer"
    return "unassigned"


def fallback_devils_summary(opinions: dict[str, str], persona_map: dict[str, Persona]) -> dict:
    """
    Build a lightweight summary of devil's advocate arguments without extra LLM calls.

    Used as fallback when chairman synthesis fails.

    Args:
        opinions: Dict of model -> opinion content
        persona_map: Dict of model -> assigned persona

    Returns:
        Summary dict with attacker/defender/synthesizer/unassigned lists
    """
    summary = {"attacker": [], "defender": [], "synthesizer": [], "unassigned": []}
    for model, opinion in opinions.items():
        persona = persona_map.get(model)
        team = infer_devils_team(persona)
        try:
            parsed = extract_json(opinion)
            key_points = parsed.get('key_points') or []
            answer = parsed.get('answer')
            collected = key_points if key_points else [answer] if answer else []
            summary[team].extend([str(point) for point in collected if point])
        except Exception:
            summary[team].append(opinion)
    return summary


async def summarize_devils_advocate_arguments(
    query: str,
    opinions: dict[str, str],
    persona_map: dict[str, Persona],
    chairman: str,
    timeout: int
) -> dict:
    """
    Summarize key devil's advocate arguments by team (attacker/defender/synthesizer).

    Uses the chairman model to synthesize arguments from each team perspective.
    Falls back to deterministic summary if LLM call fails.

    Args:
        query: Original question
        opinions: Dict of model -> opinion content
        persona_map: Dict of model -> assigned persona
        chairman: Chairman model for synthesis
        timeout: Timeout in seconds

    Returns:
        Summary dict with team arguments and headline takeaways
    """
    if not opinions:
        return {}

    persona_lines = []
    for model, persona in persona_map.items():
        team_hint = infer_devils_team(persona)
        persona_lines.append(f"- {model}: {persona.title} ({persona.role}) | inferred_team: {team_hint}")

    prompt = f"""<s>Summarize a devil's advocate mini-cycle.</s>

<question>{query}</question>

<participants>
{chr(10).join(persona_lines)}
</participants>

<round_arguments>
{json.dumps(opinions, indent=2)}
</round_arguments>

<instructions>
Group the major arguments by team.
Return ONLY JSON:
{{"attacker": ["key critiques"], "defender": ["defenses"], "synthesizer": ["integrations or meta points"], "unassigned": [], "headline_takeaways": ["2-3 bullet summary"]}}
</instructions>"""

    actual_chairman = get_chairman_with_fallback(chairman)
    metrics = get_metrics()
    start = time.time()

    if actual_chairman in ADAPTERS:
        result = await ADAPTERS[actual_chairman](prompt, timeout)
        if metrics:
            metrics.record_latency('devils_advocate_summary', int((time.time() - start) * 1000))
        if result.success:
            try:
                parsed = get_parsed_json(result)
                # Ensure expected keys exist
                for key in ("attacker", "defender", "synthesizer", "unassigned", "headline_takeaways"):
                    parsed.setdefault(key, [])
                return parsed
            except Exception:
                emit({"type": "devils_advocate_summary_parse_error", "error": "Failed to parse chairman summary"})

    # Fallback to deterministic summary
    fallback = fallback_devils_summary(opinions, persona_map)
    fallback["headline_takeaways"] = []
    return fallback

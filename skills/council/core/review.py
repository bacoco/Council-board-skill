"""
Peer review for Council deliberation.

Handles anonymized scoring and contradiction extraction.
"""

from typing import TYPE_CHECKING

from .emit import emit
from .parsing import anonymize_responses, get_parsed_json
from .prompts import build_review_prompt
from .adapters import ADAPTERS, get_chairman_with_fallback

if TYPE_CHECKING:
    from .models import SessionConfig


async def peer_review(config: 'SessionConfig', opinions: dict[str, str]) -> dict:
    """
    Conduct peer review of model opinions.

    The chairman reviews and scores all opinions anonymously,
    identifying conflicts and ranking responses.

    Args:
        config: Session configuration
        opinions: Dict of model -> opinion content

    Returns:
        Dict with review scores and label->model mapping
    """
    emit({"type": "status", "stage": 2, "msg": "Peer review in progress..."})

    if config.anonymize:
        anon_responses, mapping = anonymize_responses(opinions)
    else:
        anon_responses = {m: opinions[m] for m in opinions}
        mapping = {m: m for m in opinions}

    prompt = build_review_prompt(config.query, anon_responses)

    # Use chairman with failover chain (Council recommendation #1)
    actual_chairman = get_chairman_with_fallback(config.chairman)
    if actual_chairman in ADAPTERS:
        result = await ADAPTERS[actual_chairman](prompt, config.timeout)
        if result.success:
            review = get_parsed_json(result)  # Use cached JSON parsing
            # Emit scores
            for resp_id, scores in review.get('scores', {}).items():
                emit({"type": "score", "reviewer": actual_chairman, "target": resp_id, "scores": scores})
            return {"review": review, "mapping": mapping}

    return {"review": {}, "mapping": mapping}


def extract_contradictions(review: dict) -> list[str]:
    """Extract key conflicts from peer review."""
    return review.get('key_conflicts', [])

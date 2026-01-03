"""
STORM Prompt Templates - Structured prompts for workflow agents.

Provides prompt templates for:
- Moderator decisions
- Decision workflow nodes
- Research workflow nodes
- Code review workflow nodes
- Evidence evaluation
"""

from .storm_prompts import (
    # Moderator prompts
    MODERATOR_ROUTING_PROMPT,
    MODERATOR_SHALLOW_CONSENSUS_PROMPT,

    # Decision workflow prompts
    DECISION_GENERATE_OPTIONS_PROMPT,
    DECISION_RUBRIC_SCORING_PROMPT,
    DECISION_RED_TEAM_PROMPT,
    DECISION_RECOMMENDATION_PROMPT,

    # Research workflow prompts
    RESEARCH_PERSPECTIVES_PROMPT,
    RESEARCH_QUESTIONS_PROMPT,
    RESEARCH_DRAFT_PROMPT,
    RESEARCH_CRITIQUE_PROMPT,

    # Code review workflow prompts
    REVIEW_STATIC_SCAN_PROMPT,
    REVIEW_THREAT_MODEL_PROMPT,
    REVIEW_QUALITY_PROMPT,
    REVIEW_PATCHES_PROMPT,

    # Evidence prompts
    EVIDENCE_JUDGE_PROMPT,
    CLAIM_EXTRACTION_PROMPT,

    # Helpers
    format_prompt,
    format_kb_context,
)

__all__ = [
    'MODERATOR_ROUTING_PROMPT',
    'MODERATOR_SHALLOW_CONSENSUS_PROMPT',
    'DECISION_GENERATE_OPTIONS_PROMPT',
    'DECISION_RUBRIC_SCORING_PROMPT',
    'DECISION_RED_TEAM_PROMPT',
    'DECISION_RECOMMENDATION_PROMPT',
    'RESEARCH_PERSPECTIVES_PROMPT',
    'RESEARCH_QUESTIONS_PROMPT',
    'RESEARCH_DRAFT_PROMPT',
    'RESEARCH_CRITIQUE_PROMPT',
    'REVIEW_STATIC_SCAN_PROMPT',
    'REVIEW_THREAT_MODEL_PROMPT',
    'REVIEW_QUALITY_PROMPT',
    'REVIEW_PATCHES_PROMPT',
    'EVIDENCE_JUDGE_PROMPT',
    'CLAIM_EXTRACTION_PROMPT',
    'format_prompt',
    'format_kb_context',
]

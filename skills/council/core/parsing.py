"""
JSON extraction and response parsing utilities.
"""

import json
import re
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import LLMResponse

# Pre-compiled regex patterns (performance optimization)
JSON_PATTERN = re.compile(r'\{[\s\S]*\}')  # Extract JSON from text


def get_base_model(model_instance: str) -> str:
    """
    Extract base model name from instance ID.

    Examples:
        'claude' -> 'claude'
        'claude_instance_1' -> 'claude'
        'gemini_instance_2' -> 'gemini'

    Args:
        model_instance: Model instance ID (base model or instance like 'claude_instance_1')

    Returns:
        Base model name (claude, gemini, or codex)
    """
    if '_instance_' in model_instance:
        return model_instance.split('_instance_')[0]
    return model_instance


def extract_json(text: str):
    """
    Extract JSON from text response, with fallback to raw text.

    Handles both JSON objects {...} and arrays [...].
    Uses pre-compiled regex.
    """
    text = text.strip()

    # Try parsing as-is (works for both objects and arrays)
    if text.startswith('{') or text.startswith('['):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Look for JSON block using pre-compiled pattern (objects)
    match = JSON_PATTERN.search(text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Look for JSON array
    array_match = re.search(r'\[[\s\S]*\]', text)
    if array_match:
        try:
            return json.loads(array_match.group())
        except json.JSONDecodeError:
            pass

    return {"raw": text}


def get_parsed_json(response: 'LLMResponse') -> dict:
    """
    Get parsed JSON from LLMResponse, using cached value if available.

    This avoids re-parsing the same JSON multiple times (performance optimization).
    """
    if response.parsed_json is None:
        response.parsed_json = extract_json(response.content)
    return response.parsed_json


def anonymize_responses(responses: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """
    Anonymize model responses by assigning random labels (A, B, C, ...).

    Returns:
        Tuple of (anonymized_responses, label_to_model_mapping)
    """
    labels = ['A', 'B', 'C', 'D', 'E']
    models = list(responses.keys())
    random.shuffle(models)
    anonymized = {}
    mapping = {}
    for label, model in zip(labels, models):
        anonymized[label] = responses[model]
        mapping[label] = model
    return anonymized, mapping

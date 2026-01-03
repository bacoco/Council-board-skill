"""
Model Query Helper - Simplifies querying models from workflow nodes.

Provides a clean interface for workflow nodes to query models without
dealing with adapter details directly.
"""

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.adapters import ADAPTERS, check_cli_available, get_chairman_with_fallback
from core.emit import emit


@dataclass
class QueryResult:
    """Result from querying a model."""
    success: bool
    content: str
    model: str
    latency_ms: int
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'model': self.model,
            'latency_ms': self.latency_ms,
            'content_length': len(self.content),
            'error': self.error
        }


async def query_model(model: str, prompt: str, timeout: int = 120) -> QueryResult:
    """
    Query a single model with a prompt.

    Args:
        model: Model name ('claude', 'gemini', 'codex')
        prompt: The prompt to send
        timeout: Timeout in seconds

    Returns:
        QueryResult with response or error
    """
    if not check_cli_available(model):
        return QueryResult(
            success=False,
            content='',
            model=model,
            latency_ms=0,
            error=f"Model {model} CLI not available"
        )

    adapter = ADAPTERS.get(model)
    if not adapter:
        return QueryResult(
            success=False,
            content='',
            model=model,
            latency_ms=0,
            error=f"No adapter for model {model}"
        )

    try:
        response = await adapter(prompt, timeout)
        return QueryResult(
            success=response.success,
            content=response.content if response.success else '',
            model=model,
            latency_ms=response.latency_ms,
            error=response.error if not response.success else None
        )
    except Exception as e:
        return QueryResult(
            success=False,
            content='',
            model=model,
            latency_ms=0,
            error=str(e)
        )


async def query_models_parallel(
    models: List[str],
    prompt: str,
    timeout: int = 120
) -> List[QueryResult]:
    """
    Query multiple models in parallel.

    Args:
        models: List of model names
        prompt: The prompt to send to all models
        timeout: Timeout per model

    Returns:
        List of QueryResult, one per model
    """
    tasks = [query_model(m, prompt, timeout) for m in models]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    processed = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed.append(QueryResult(
                success=False,
                content='',
                model=models[i],
                latency_ms=0,
                error=str(result)
            ))
        else:
            processed.append(result)

    return processed


async def query_chairman(
    prompt: str,
    preferred: str = 'claude',
    timeout: int = 120
) -> QueryResult:
    """
    Query the chairman model with automatic failover.

    Args:
        prompt: The prompt to send
        preferred: Preferred chairman model
        timeout: Timeout in seconds

    Returns:
        QueryResult from available chairman
    """
    chairman = get_chairman_with_fallback(preferred)
    return await query_model(chairman, prompt, timeout)


# =============================================================================
# Response Parsing Helpers
# =============================================================================

def parse_options(response: str) -> List[Dict[str, Any]]:
    """
    Parse options from a model response.

    Expected format:
    ## Option 1: [Name]
    Description: [description]
    Pros:
    - [pro 1]
    Cons:
    - [con 1]
    """
    options = []
    option_pattern = r'##\s*Option\s*\d+:\s*(.+?)(?=##\s*Option|\Z)'
    matches = re.findall(option_pattern, response, re.DOTALL | re.IGNORECASE)

    for i, match in enumerate(matches):
        lines = match.strip().split('\n')
        name = lines[0].strip() if lines else f"Option {i+1}"

        # Extract description
        desc_match = re.search(r'Description:\s*(.+?)(?=Pros:|Cons:|\Z)', match, re.DOTALL | re.IGNORECASE)
        description = desc_match.group(1).strip() if desc_match else ""

        # Extract pros
        pros_match = re.search(r'Pros:\s*(.+?)(?=Cons:|\Z)', match, re.DOTALL | re.IGNORECASE)
        pros = []
        if pros_match:
            pros = [p.strip().lstrip('- ') for p in pros_match.group(1).strip().split('\n') if p.strip().startswith('-')]

        # Extract cons
        cons_match = re.search(r'Cons:\s*(.+?)(?=##|\Z)', match, re.DOTALL | re.IGNORECASE)
        cons = []
        if cons_match:
            cons = [c.strip().lstrip('- ') for c in cons_match.group(1).strip().split('\n') if c.strip().startswith('-')]

        options.append({
            'id': f'opt_{i+1}',
            'name': name,
            'description': description,
            'pros': pros,
            'cons': cons
        })

    return options


def parse_scores(response: str) -> Dict[str, Dict[str, float]]:
    """
    Parse rubric scores from a model response.

    Returns dict mapping option name to criterion scores.
    """
    scores = {}
    current_option = None

    for line in response.split('\n'):
        # Detect option header
        if line.strip().startswith('##'):
            current_option = line.replace('#', '').strip()
            scores[current_option] = {}
        elif current_option and ':' in line:
            # Parse score line like "- feasibility: 75 - rationale"
            parts = line.split(':')
            if len(parts) >= 2:
                criterion = parts[0].strip().lstrip('- ')
                score_part = parts[1].split('-')[0].strip()
                try:
                    score = float(score_part) / 100.0  # Normalize to 0-1
                    scores[current_option][criterion.lower()] = score
                except ValueError:
                    pass

    return scores


def parse_threats(response: str) -> List[Dict[str, Any]]:
    """
    Parse security threats from a model response.
    """
    threats = []
    threat_pattern = r'##\s*Threat\s*\d+:\s*(.+?)(?=##\s*Threat|\Z)'
    matches = re.findall(threat_pattern, response, re.DOTALL | re.IGNORECASE)

    for i, match in enumerate(matches):
        name_match = re.match(r'(.+?)(?:\n|$)', match)
        name = name_match.group(1).strip() if name_match else f"Threat {i+1}"

        # Extract fields
        attack = _extract_field(match, 'Attack Vector')
        impact = _extract_field(match, 'Impact')
        likelihood = _extract_field(match, 'Likelihood', 'medium')
        severity = _extract_field(match, 'Severity', 'medium')
        mitigation = _extract_field(match, 'Mitigation')

        threats.append({
            'id': f'threat_{i+1}',
            'name': name,
            'attack_vector': attack,
            'impact': impact,
            'likelihood': likelihood.lower(),
            'severity': severity.lower(),
            'mitigation': mitigation
        })

    return threats


def parse_issues(response: str) -> List[Dict[str, Any]]:
    """
    Parse code issues from a model response.
    """
    issues = []
    issue_pattern = r'##\s*Issue\s*\d+(.+?)(?=##\s*Issue|\Z)'
    matches = re.findall(issue_pattern, response, re.DOTALL | re.IGNORECASE)

    for i, match in enumerate(matches):
        severity = _extract_field(match, 'Severity', 'medium')
        category = _extract_field(match, 'Category', 'correctness')
        location = _extract_field(match, 'Location', '')
        description = _extract_field(match, 'Description', '')
        suggestion = _extract_field(match, 'Suggestion', '')

        issues.append({
            'id': f'issue_{i+1}',
            'severity': severity.lower(),
            'category': category.lower(),
            'location': location,
            'description': description,
            'suggestion': suggestion
        })

    return issues


def parse_claims(response: str) -> List[Dict[str, Any]]:
    """
    Parse extracted claims from a model response.
    """
    claims = []
    lines = response.strip().split('\n')

    for line in lines:
        # Match format: 1. [claim] | Confidence: [level] | Speculative: [yes/no]
        match = re.match(r'\d+\.\s*(.+?)\s*\|\s*Confidence:\s*(\w+)\s*\|\s*Speculative:\s*(\w+)', line)
        if match:
            text = match.group(1).strip()
            confidence = match.group(2).lower()
            speculative = match.group(3).lower() == 'yes'

            conf_map = {'low': 0.3, 'medium': 0.6, 'high': 0.85}
            claims.append({
                'text': text,
                'confidence': conf_map.get(confidence, 0.5),
                'speculative': speculative
            })

    return claims


def _extract_field(text: str, field: str, default: str = '') -> str:
    """Extract a field value from text."""
    pattern = rf'{field}:\s*(.+?)(?:\n|$)'
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else default


def parse_recommendation(response: str) -> Dict[str, Any]:
    """
    Parse recommendation from a model response.
    """
    result = {
        'recommended_option': '',
        'rationale': '',
        'tradeoffs': [],
        'tripwires': [],
        'confidence': 0.5,
        'confidence_rationale': ''
    }

    # Extract recommendation
    rec_match = re.search(r'RECOMMENDATION:\s*(.+?)(?:\n|RATIONALE)', response, re.IGNORECASE | re.DOTALL)
    if rec_match:
        result['recommended_option'] = rec_match.group(1).strip()

    # Extract rationale
    rat_match = re.search(r'RATIONALE:\s*(.+?)(?:\n\n|TRADEOFFS)', response, re.IGNORECASE | re.DOTALL)
    if rat_match:
        result['rationale'] = rat_match.group(1).strip()

    # Extract tradeoffs
    trade_match = re.search(r'TRADEOFFS:\s*(.+?)(?:\n\n|TRIPWIRES)', response, re.IGNORECASE | re.DOTALL)
    if trade_match:
        result['tradeoffs'] = [t.strip().lstrip('- ') for t in trade_match.group(1).split('\n') if t.strip().startswith('-')]

    # Extract tripwires
    trip_match = re.search(r'TRIPWIRES[^:]*:\s*(.+?)(?:\n\n|CONFIDENCE)', response, re.IGNORECASE | re.DOTALL)
    if trip_match:
        result['tripwires'] = [t.strip().lstrip('- ') for t in trip_match.group(1).split('\n') if t.strip().startswith('-')]

    # Extract confidence
    conf_match = re.search(r'CONFIDENCE:\s*(\d+)', response, re.IGNORECASE)
    if conf_match:
        result['confidence'] = int(conf_match.group(1)) / 100.0

    # Extract confidence rationale
    confrat_match = re.search(r'CONFIDENCE_RATIONALE:\s*(.+?)(?:\n\n|\Z)', response, re.IGNORECASE | re.DOTALL)
    if confrat_match:
        result['confidence_rationale'] = confrat_match.group(1).strip()

    return result

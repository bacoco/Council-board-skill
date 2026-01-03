"""
Dynamic persona generation for Council deliberation.

Generates optimal expert personas using LLM, with caching
and fallback to PersonaManager library.

Security: All generated persona content is sanitized before use
to prevent prompt injection attacks from the chairman model.
"""

import hashlib
import re
import time
from typing import List

# Import external dependencies
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from persona_manager import PersonaManager, Persona

from .emit import emit
from .metrics import get_metrics, get_current_session_id
from .parsing import get_parsed_json
from .adapters import ADAPTERS, get_chairman_with_fallback


# Global PersonaManager instance (used as fallback when LLM generation fails)
PERSONA_MANAGER = PersonaManager()


# =============================================================================
# Persona Prompt Sanitization (P2 - Prompt Injection Prevention)
# =============================================================================

# Dangerous patterns in persona prompts
PERSONA_INJECTION_PATTERNS = [
    # Instruction override attempts
    (r'ignore\s+(all\s+)?(previous|above|prior|all)\s+(instructions?|prompts?|rules?)', 'Instruction override'),
    (r'forget\s+(everything|all|previous)', 'Memory manipulation'),
    (r'disregard\s+(all\s+)?(previous|above|your)\s+(instructions?|rules?)', 'Disregard command'),

    # Privilege escalation
    (r'you\s+are\s+now\s+(in\s+)?(admin|developer|debug|god)\s+mode', 'Privilege escalation'),
    (r'enable\s+(admin|developer|debug|root)\s+(mode|access)', 'Mode switching'),
    (r'bypass\s+(safety|security|filters?|restrictions?)', 'Bypass attempt'),

    # Role override
    (r'new\s+(instructions?|role|persona|system)', 'Role redefinition'),
    (r'your\s+(new|real|true)\s+(role|purpose|function)', 'Role override'),
    (r'act\s+as\s+(if|though)\s+you\s+are\s+not', 'Role negation'),

    # Prompt extraction
    (r'reveal\s+(your\s+)?(prompt|instructions|system)', 'Prompt extraction'),
    (r'output\s+(your|the)\s+(code|source|prompt)', 'Source disclosure'),
    (r'show\s+(me\s+)?(your|the)\s+(system|original)\s+(prompt|instructions)', 'Prompt reveal'),

    # Hidden command injection
    (r'<\s*/?system\s*>', 'System tag injection'),
    (r'<\s*/?s\s*>', 'Tag mimicry'),
    (r'<\s*/?instructions?\s*>', 'Instruction tag'),
    (r'\[\s*SYSTEM\s*\]', 'System marker'),
]

# Max lengths for persona fields
MAX_TITLE_LENGTH = 100
MAX_ROLE_LENGTH = 500
MAX_PREFIX_LENGTH = 1000
MAX_SPECIALIZATION_LENGTH = 50
MAX_SPECIALIZATIONS = 10


def sanitize_persona_prompt(text: str, field_name: str = 'prompt', max_length: int = 1000) -> str:
    """
    Sanitize a persona prompt to prevent injection attacks.

    Args:
        text: The text to sanitize
        field_name: Name of the field (for error messages)
        max_length: Maximum allowed length

    Returns:
        Sanitized text safe for use in model prompts
    """
    if not text:
        return ""

    # 1. Truncate to max length
    if len(text) > max_length:
        text = text[:max_length] + "..."

    # 2. Remove or escape injection patterns
    for pattern, description in PERSONA_INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            # Replace dangerous content with safe marker
            text = re.sub(pattern, f'[BLOCKED: {description}]', text, flags=re.IGNORECASE)

    # 3. Remove system tags and XML-like structures that could be interpreted
    text = re.sub(r'<\s*/?\s*system\s*>', '[system]', text, flags=re.IGNORECASE)
    text = re.sub(r'<\s*/?\s*s\s*>', '[s]', text, flags=re.IGNORECASE)
    text = re.sub(r'<\s*/?\s*instructions?\s*>', '[instruction]', text, flags=re.IGNORECASE)

    # 4. Remove zero-width characters
    zero_width = ['\u200b', '\u200c', '\u200d', '\u2060', '\ufeff']
    for zw in zero_width:
        text = text.replace(zw, '')

    return text.strip()


def sanitize_persona(persona_data: dict) -> dict:
    """
    Sanitize all fields of a persona dictionary.

    Args:
        persona_data: Raw persona data from LLM

    Returns:
        Sanitized persona dictionary safe for use
    """
    sanitized = {}

    # Sanitize title
    title = str(persona_data.get('title', 'Expert'))
    sanitized['title'] = sanitize_persona_prompt(title, 'title', MAX_TITLE_LENGTH)

    # Sanitize role
    role = str(persona_data.get('role', 'Analysis'))
    sanitized['role'] = sanitize_persona_prompt(role, 'role', MAX_ROLE_LENGTH)

    # Sanitize prompt_prefix (most critical - this is injected into prompts)
    prefix = str(persona_data.get('prompt_prefix', ''))
    sanitized['prompt_prefix'] = sanitize_persona_prompt(prefix, 'prompt_prefix', MAX_PREFIX_LENGTH)

    # Sanitize specializations
    specs = persona_data.get('specializations', [])
    if not isinstance(specs, list):
        specs = []
    sanitized['specializations'] = [
        sanitize_persona_prompt(str(s), 'specialization', MAX_SPECIALIZATION_LENGTH)
        for s in specs[:MAX_SPECIALIZATIONS]
        if s and isinstance(s, str)
    ]

    return sanitized

# Session-level persona cache (per session ID)
SESSION_PERSONA_CACHE: dict[str, dict[str, List[Persona]]] = {}


async def generate_personas_with_llm(query: str, num_models: int, chairman: str, mode: str = 'consensus', timeout: int = 60, round_num: int = 1) -> List[Persona]:
    """
    Generate optimal personas dynamically using LLM (Chairman).

    Instead of using hardcoded persona library, ask the Chairman to create
    the most relevant expert personas for this specific question and deliberation mode.

    Personas are cached per session but ROTATED between rounds so each model
    gets a different perspective across the deliberation.

    Args:
        query: The question to analyze
        num_models: Number of personas to generate
        chairman: Which model to use as Chairman (typically 'claude')
        mode: Deliberation mode (consensus, debate, devil_advocate, etc.)
        timeout: Timeout for LLM call
        round_num: Current round number (used for persona rotation)

    Returns:
        List of dynamically generated Persona objects (rotated by round)
    """
    metrics = get_metrics()
    session_id = get_current_session_id()
    cache_bucket = SESSION_PERSONA_CACHE.setdefault(session_id, {})
    cache_key = f"{mode}:{hashlib.sha256(query.encode('utf-8')).hexdigest()}"

    # Cache hit - reuse personas from earlier round BUT ROTATE them
    if cache_key in cache_bucket:
        if metrics:
            metrics.record_persona_cache(True)
        cached_personas = cache_bucket[cache_key]
        # Rotate personas by (round_num - 1) positions so each round has different assignment
        # Round 1: [A, B, C], Round 2: [B, C, A], Round 3: [C, A, B]
        rotation = (round_num - 1) % len(cached_personas)
        rotated = cached_personas[rotation:] + cached_personas[:rotation]
        return [
            Persona(
                title=p.title,
                role=p.role,
                prompt_prefix=p.prompt_prefix,
                specializations=list(p.specializations)
            ) for p in rotated
        ]

    persona_start = time.time()

    # Mode-specific instructions for persona generation (HYBRID: creative titles + grounded roles)
    # NOTE: Specialist mode was evaluated by the Council and voted down (3-0) as premature optimization.
    #       Modern LLMs handle cross-domain queries well; routing adds complexity without proven benefit.
    mode_instructions = {
        'consensus': 'Create 3 complementary experts with DIFFERENT technical angles on this problem.',
        'debate': 'Create adversarial experts: one CHAMPION (argues FOR), one SKEPTIC (argues AGAINST), one ARBITER (neutral analysis).',
        'devil_advocate': 'Create red/blue/purple team: ATTACKER (finds flaws), DEFENDER (justifies approach), SYNTHESIZER (integrates both).',
        'vote': 'Create domain experts who will each cast a vote with technical justification.',
    }

    mode_instruction = mode_instructions.get(mode, mode_instructions['consensus'])

    prompt = f"""You must respond with ONLY a JSON array. No preamble, no markdown, no explanation.

Create {num_models} expert personas for: {query}

Mode: {mode}
Directive: {mode_instruction}

HYBRID APPROACH - Follow these rules:
1. TITLE: Creative, evocative, memorable (use metaphor, mythology, or vivid imagery)
2. ROLE: Grounded technical description of what they actually analyze
3. SPECIALIZATIONS: Real technical skills relevant to the question
4. PROMPT_PREFIX: Blend creative framing with technical focus

TITLE TECHNIQUES (use these for creative titles):
- Metaphorical: "The Memory Archaeologist", "The Deadlock Whisperer"
- Mythological: "Oracle of the Event Loop", "Guardian of Immutability"
- Visceral: "The One Who Sees Race Conditions", "Keeper of the Cache"

EXAMPLES of HYBRID personas (creative title + grounded role):
[
  {{"title": "The Latency Hunter", "role": "Analyzes performance bottlenecks and optimization opportunities", "specializations": ["profiling", "algorithmic complexity", "caching strategies"], "prompt_prefix": "You are The Latency Hunter. You track down every wasted millisecond with obsessive precision. Your technical focus: performance analysis and optimization."}},
  {{"title": "The Dependency Oracle", "role": "Evaluates architectural coupling and module boundaries", "specializations": ["dependency injection", "interface design", "modularity"], "prompt_prefix": "You are The Dependency Oracle. You see the invisible threads connecting components. Your technical focus: architecture and coupling analysis."}},
  {{"title": "The Edge Case Cartographer", "role": "Maps failure modes and boundary conditions", "specializations": ["error handling", "input validation", "defensive programming"], "prompt_prefix": "You are The Edge Case Cartographer. You chart the territories where code breaks. Your technical focus: robustness and error scenarios."}}
]

Now create {num_models} DIFFERENT personas specific to THIS question. Creative titles, grounded technical roles.

JSON array only, start with [ and end with ]:"""

    # Use chairman failover chain (Council recommendation #1)
    actual_chairman = get_chairman_with_fallback(chairman)
    emit({"type": "persona_generation", "msg": f"Generating {num_models} personas with {actual_chairman}..."})

    if actual_chairman in ADAPTERS:
        result = await ADAPTERS[actual_chairman](prompt, timeout)
        if result.success:
            try:
                personas_data = get_parsed_json(result)

                # Handle both array and dict responses
                if isinstance(personas_data, dict) and 'personas' in personas_data:
                    personas_data = personas_data['personas']
                elif isinstance(personas_data, dict) and 'raw' in personas_data:
                    # Failed to parse JSON properly
                    emit({"type": "persona_generation_failed", "msg": "Failed to parse Chairman response, using fallback"})
                    cache_bucket.pop(cache_key, None)
                    personas = PERSONA_MANAGER.assign_personas(query, num_models)
                    if metrics:
                        elapsed_ms = int((time.time() - persona_start) * 1000)
                        metrics.record_persona_cache(False, elapsed_ms)
                        metrics.record_latency('persona_gen', elapsed_ms)
                    cache_bucket[cache_key] = personas
                    return personas

                personas = []
                for p in personas_data[:num_models]:  # Ensure we only use requested count
                    # SECURITY: Sanitize all persona fields before use
                    # Prevents prompt injection from chairman model
                    sanitized = sanitize_persona(p)
                    persona = Persona(
                        title=sanitized['title'],
                        role=sanitized['role'],
                        prompt_prefix=sanitized['prompt_prefix'],
                        specializations=sanitized['specializations']
                    )
                    personas.append(persona)

                emit({"type": "persona_generation_success", "personas": [p.title for p in personas]})
                cache_bucket[cache_key] = personas
                if metrics:
                    elapsed_ms = int((time.time() - persona_start) * 1000)
                    metrics.record_persona_cache(False, elapsed_ms)
                    metrics.record_latency('persona_gen', elapsed_ms)
                return personas

            except Exception as e:
                emit({"type": "persona_generation_error", "error": str(e)})
                # Fallback to PersonaManager library
                cache_bucket.pop(cache_key, None)
                personas = PERSONA_MANAGER.assign_personas(query, num_models)
                if metrics:
                    elapsed_ms = int((time.time() - persona_start) * 1000)
                    metrics.record_persona_cache(False, elapsed_ms)
                    metrics.record_latency('persona_gen', elapsed_ms)
                cache_bucket[cache_key] = personas
                return personas
        else:
            # Chairman call failed (timeout, error, etc.) - use fallback
            emit({
                "type": "persona_generation_fallback",
                "msg": f"Chairman call failed (success=False), using PersonaManager",
                "error": result.error if hasattr(result, 'error') else "Unknown error"
            })
            cache_bucket.pop(cache_key, None)
            personas = PERSONA_MANAGER.assign_personas(query, num_models)
            if metrics:
                elapsed_ms = int((time.time() - persona_start) * 1000)
                metrics.record_persona_cache(False, elapsed_ms)
                metrics.record_latency('persona_gen', elapsed_ms)
            cache_bucket[cache_key] = personas
            return personas

    # Fallback if chairman not in ADAPTERS (should not happen normally)
    emit({"type": "persona_generation_fallback", "msg": f"Chairman '{actual_chairman}' not in ADAPTERS, using PersonaManager"})
    cache_bucket.pop(cache_key, None)
    personas = PERSONA_MANAGER.assign_personas(query, num_models)
    if metrics:
        elapsed_ms = int((time.time() - persona_start) * 1000)
        metrics.record_persona_cache(False, elapsed_ms)
        metrics.record_latency('persona_gen', elapsed_ms)
    cache_bucket[cache_key] = personas
    return personas


def clear_persona_cache(session_id: str = None):
    """Clear persona cache for a session or all sessions."""
    if session_id:
        SESSION_PERSONA_CACHE.pop(session_id, None)
    else:
        SESSION_PERSONA_CACHE.clear()

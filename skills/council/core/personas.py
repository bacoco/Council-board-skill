"""
Dynamic persona generation for Council deliberation.

Generates optimal expert personas using LLM, with caching
and fallback to PersonaManager library.
"""

import hashlib
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

# Session-level persona cache (per session ID)
SESSION_PERSONA_CACHE: dict[str, dict[str, List[Persona]]] = {}


async def generate_personas_with_llm(query: str, num_models: int, chairman: str, mode: str = 'consensus', timeout: int = 60) -> List[Persona]:
    """
    Generate optimal personas dynamically using LLM (Chairman).

    Instead of using hardcoded persona library, ask the Chairman to create
    the most relevant expert personas for this specific question and deliberation mode.

    Args:
        query: The question to analyze
        num_models: Number of personas to generate
        chairman: Which model to use as Chairman (typically 'claude')
        mode: Deliberation mode (consensus, debate, devil_advocate, etc.)
        timeout: Timeout for LLM call

    Returns:
        List of dynamically generated Persona objects
    """
    metrics = get_metrics()
    session_id = get_current_session_id()
    cache_bucket = SESSION_PERSONA_CACHE.setdefault(session_id, {})
    cache_key = f"{mode}:{hashlib.sha256(query.encode('utf-8')).hexdigest()}"

    # Cache hit - reuse personas from earlier round
    if cache_key in cache_bucket:
        if metrics:
            metrics.record_persona_cache(True)
        return [
            Persona(
                title=p.title,
                role=p.role,
                prompt_prefix=p.prompt_prefix,
                specializations=list(p.specializations)
            ) for p in cache_bucket[cache_key]
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
                    persona = Persona(
                        title=p.get('title', 'Expert'),
                        role=p.get('role', 'Analysis'),
                        prompt_prefix=p.get('prompt_prefix', ''),
                        specializations=p.get('specializations', [])
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

    # Fallback if chairman unavailable
    emit({"type": "persona_generation_fallback", "msg": "Chairman unavailable, using PersonaManager"})
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

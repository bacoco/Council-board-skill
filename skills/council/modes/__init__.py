"""
Council Modes - Deliberation mode implementations.

Modes:
- consensus: Multi-round deliberation with convergence detection
- debate: FOR/AGAINST personas via run_council(mode='debate') - uses consensus.py with debate prompts
- devil_advocate: Red/Blue/Purple team via run_council(mode='devil_advocate')
- vote: Fast-path voting with weighted ballots
- adaptive: Adaptive cascade through multiple modes

Note: debate and devil_advocate modes are handled by run_council with different
persona generation and prompt templates. They don't have separate modules.
"""

import sys
from pathlib import Path

# Add parent to path for core imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from modes.consensus import run_council, gather_opinions
from modes.devil import summarize_devils_advocate_arguments, infer_devils_team, fallback_devils_summary
from modes.vote import run_vote_council, collect_votes, tally_votes, validate_vote
from modes.adaptive import run_adaptive_cascade

__all__ = [
    'run_council',
    'gather_opinions',
    'run_vote_council',
    'collect_votes',
    'tally_votes',
    'validate_vote',
    'run_adaptive_cascade',
    'summarize_devils_advocate_arguments',
    'infer_devils_team',
    'fallback_devils_summary',
]

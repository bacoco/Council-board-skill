# Council Board Skill

Multi-model deliberation orchestrator for Claude Code.

## Quick Start

```bash
python3 scripts/council.py \
  --query "Your question" \
  --mode consensus \
  --models claude,gemini,codex
```

## Modes

- `consensus` - Factual questions (default)
- `debate` - Multi-round argumentation
- `vote` - Binary/multiple choice
- `specialist` - Domain expert routing
- `devil_advocate` - Systematic challenge

## Output

NDJSON events with:
- Opinion gathering (parallel)
- Peer review (anonymized scoring)
- Contradiction detection
- Final synthesis with confidence

See [SKILL.md](SKILL.md) for details.

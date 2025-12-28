---
name: llm-council
description: Orchestrate multi-LLM deliberation via CLI-native approach. Use when user requests collective AI opinions, debates, votes, or validation across multiple models (Claude, Gemini, Codex). Triggers include "ask the council", "debate this", "vote on", "peer review", "multi-model", "deliberate", "what do other AIs think", "challenge my design", "specialist review".
---

# LLM Council Skill

Transform Claude into an orchestrator of LLM deliberation. Solicit opinions from multiple models via their CLIs, conduct anonymized peer review, and synthesize superior responses.

## When to Use

1. **Explicit requests**: "Ask the council", "debate this", "vote on", "peer review"
2. **Complex decisions**: Architecture choices, controversial topics, multi-valid-answer problems
3. **Validation needs**: Code review, design challenges, fact verification

## Deliberation Modes

| Mode | Use Case | Process |
|------|----------|---------|
| `consensus` (default) | Factual questions, single-solution problems | 3-stage Karpathy |
| `debate` | Open questions, controversial topics | Multi-round argumentation |
| `vote` | Binary/multiple choice decisions | Weighted voting |
| `specialist` | Technical expertise needed | Route to expert model |
| `devil_advocate` | Stress-test ideas | Systematic challenge |

## Quick Start

```bash
python scripts/council.py --query "Question" --mode consensus --models claude,gemini,codex
```

## Pipeline Stages

```
Stage 0: ROUTING → Classify task, estimate complexity, set council size
Stage 1: OPINIONS → Collect responses in parallel, normalize style
Stage 2: PEER REVIEW → Anonymized scoring (accuracy, completeness, reasoning, clarity)
Stage 2.5: CONTRADICTIONS → Extract conflicts, flag for resolution
Stage 3: SYNTHESIS → Chairman resolves conflicts, produces final answer
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `models` | claude,gemini,codex | Participating models |
| `chairman` | claude | Synthesizer model |
| `mode` | consensus | Deliberation mode |
| `council_budget` | balanced | fast/balanced/thorough |
| `timeout` | 60 | Per-model timeout (seconds) |
| `anonymize` | true | Anonymize responses in peer review |
| `redact_secrets` | true | Mask API keys, tokens |
| `output_level` | standard | minimal/standard/audit |

## Reference Files

- **Prompt templates**: See [references/prompts.md](references/prompts.md)
- **JSON schemas**: See [references/schemas.md](references/schemas.md)
- **Mode details**: See [references/modes.md](references/modes.md)
- **Security guidelines**: See [references/security.md](references/security.md)

## Error Handling

- **Timeout**: Model marked as abstention, continue with others
- **Invalid JSON**: Retry up to 2 times, then skip
- **Quorum failure**: Require minimum 2 valid responses

## Security

- User queries treated as DATA via XML sandwich prompts
- Secrets auto-redacted before sending to models
- Responses anonymized to prevent brand bias

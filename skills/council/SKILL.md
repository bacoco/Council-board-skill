---
name: council
description: Orchestrates multi-model deliberation from Claude, Gemini, and Codex. Use when user asks to "ask the council", "debate this", "vote on", "get multiple opinions", "peer review", "challenge my design", or requests collective AI intelligence. Also use for direct model queries like "ask Claude directly", "use Gemini", "ask Codex", "ask only Claude", "query Gemini", or comparing specific models.
allowed-tools:
  - "Bash(python3 ${SKILL_ROOT}/scripts/council.py:*)"
  - "Read(**/*.py)"
  - "Read(**/*.ts)"
  - "Read(**/*.js)"
  - "Read(**/*.tsx)"
  - "Read(**/*.go)"
  - "Read(**/*.rs)"
  - "Read(**/*.java)"
---

# Council - Multi-Model Deliberation

Orchestrate 3 AI models through multi-round deliberation, anonymous peer review, and synthesis.

## Quick Start

```bash
python3 ${SKILL_ROOT}/scripts/council.py --query "[question]" --mode consensus
```

**With code context** (recommended for reviews):

```bash
python3 ${SKILL_ROOT}/scripts/council.py \
  --query "Review this auth module" \
  --context-file /path/to/manifest.md \
  --mode devil_advocate
```

## Modes

| Mode | Use When | Process |
|------|----------|---------|
| `consensus` | Factual questions, design decisions | Multi-round with convergence detection |
| `debate` | Controversial topics, binary decisions | FOR vs AGAINST personas |
| `devil_advocate` | Stress-testing, security reviews | Red/Blue/Purple team analysis |
| `vote` | Multiple choice decisions | Weighted vote tally |
| `adaptive` | Uncertain complexity | Auto-escalates based on convergence |

**Mode details**: See [references/modes.md](references/modes.md)

## Direct Mode (Skip Deliberation)

Query individual models directly without multi-round deliberation:

```bash
# Single model
python3 ${SKILL_ROOT}/scripts/council.py --direct --models claude --query "[question]" --human

# Multiple models (sequential, no synthesis)
python3 ${SKILL_ROOT}/scripts/council.py --direct --models claude,gemini --query "[question]" --human
```

**When user says** → **Use this command**:
- "Ask Claude directly: X" → `--direct --models claude`
- "Use Gemini to explain Y" → `--direct --models gemini`
- "Ask Codex to write Z" → `--direct --models codex`
- "Ask Claude and Gemini: X" → `--direct --models claude,gemini`
- "Query all models: X" → `--direct --models claude,gemini,codex`
- "Ask the council: X" → No `--direct` flag (full deliberation)

## Key Options

```bash
--mode MODE          # consensus, debate, devil_advocate, vote, adaptive
--max-rounds N       # Max deliberation rounds (default: 3)
--trail / --no-trail # Save full reasoning to Markdown (default: on)
--human              # Human-readable output instead of JSON
--context-file PATH  # Load code files via manifest
```

## Output

JSON with answer, confidence, and optional trail file path:

```json
{
  "answer": "Council recommends...",
  "confidence": 0.91,
  "trail_file": "./council_trails/council_2025-12-31_consensus_query.md"
}
```

## Context Manifest Format

Create a manifest file listing code files to analyze:

```markdown
# Council Context

## Question
Review auth module for security issues

## Files to Analyze

### src/auth.py
- Main authentication logic

### src/config.py
- JWT configuration
```

Lines starting with `### filename.ext` are loaded automatically.

## Reference Documentation

- [references/modes.md](references/modes.md) - Deliberation mode details
- [references/security.md](references/security.md) - Security features, input validation
- [references/resilience.md](references/resilience.md) - Graceful degradation, circuit breaker
- [references/failure-modes.md](references/failure-modes.md) - Error handling and recovery
- [references/output-format.md](references/output-format.md) - Response templates
- [references/examples.md](references/examples.md) - Usage examples

## Architecture Note

Council uses `sys.path.insert()` for imports to enable direct CLI invocation (`python3 scripts/council.py`). This is intentional - relative imports would require running as a module (`python -m council.scripts.council`), breaking the user-friendly CLI interface. The root `__init__.py` provides package organization while maintaining CLI usability.

## Configuration

Persistent settings via `council.config.yaml`:

```yaml
providers: [claude, gemini, codex]
chairman: claude
timeout: 420
max_rounds: 3
convergence_threshold: 0.8
min_quorum: 2
```

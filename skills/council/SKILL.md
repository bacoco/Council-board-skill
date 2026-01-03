---
name: council
description: |
  Orchestrates multi-model deliberation from Claude, Gemini, and Codex.

  INVOKE THIS SKILL when user wants:
  - Multiple AI perspectives: "ask the council", "get opinions", "what do the models think"
  - Debate/both sides: "both sides of", "pros and cons", "I'm torn between", "arguments for and against"
  - Stress-testing: "poke holes", "what could go wrong", "find flaws", "what am I missing", "blind spots"
  - Choosing between options: "help me choose", "which should I pick", "A vs B vs C"
  - Deep understanding: "deeply understand", "thorough research", "comprehensive analysis"
  - Direct model query: "ask Claude directly", "just Gemini", "Codex only", "skip the council"
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

## Mode Detection from Natural Language

**IMPORTANT**: Detect the user's intent and use the correct `--mode` flag.

### Detect DEBATE mode when user says:
- "both sides", "pros and cons", "arguments for and against"
- "I'm torn between", "help me see both perspectives"
- "what's the case for/against"
- Binary choices with genuine controversy (no clear right answer)

→ Use `--mode debate`

### Detect DEVIL_ADVOCATE mode when user says:
- "poke holes", "tear apart", "what could go wrong"
- "find flaws", "stress test", "attack my idea"
- "what am I missing", "blind spots", "risks"
- "challenge", "critique", "red team"

→ Use `--mode devil_advocate`

### Detect VOTE mode when user says:
- "help me choose between", "which should I pick"
- "A vs B vs C" (3+ options)
- "rank these options", "which is best"

→ Use `--mode vote`

### Detect STORM modes when user says:
- "deeply understand", "thorough research", "comprehensive analysis" → `--mode storm_research`
- "rigorous decision", "structured evaluation", "compare options formally" → `--mode storm_decision`
- "audit", "review thoroughly", "check for issues" → `--mode storm_review`

### Default to CONSENSUS when:
- User asks a question expecting a clear answer
- "What should I do?", "Is this a good idea?", "How should I approach..."
- No explicit signals for other modes

→ Use `--mode consensus` (or omit, it's the default)

## Modes Reference

### Classic Modes
| Mode | Use When | Process |
|------|----------|---------|
| `consensus` | Factual questions, design decisions | Multi-round with convergence detection |
| `debate` | Controversial topics, binary decisions | FOR vs AGAINST personas |
| `devil_advocate` | Stress-testing, security reviews | Red/Blue/Purple team analysis |
| `vote` | Multiple choice decisions | Weighted vote tally |
| `adaptive` | Uncertain complexity | Auto-escalates based on convergence |

### STORM Modes (Evidence-Grounded)
| Mode | Use When | Workflow |
|------|----------|----------|
| `storm_decision` | Architecture choices, technology selection | Options → Rubric → Red-team → Evidence → Recommendation |
| `storm_research` | Deep dives, technical analysis | Perspectives → Questions → Retrieve → Draft → Critique |
| `storm_review` | Code review, security audit | Static scan → Threat model → Quality → Patches → Checklist |

STORM modes use KnowledgeBase tracking, semantic evidence matching, and Moderator-driven routing.

**Mode details**: See [references/modes.md](references/modes.md)

## Direct Mode (Skip Deliberation)

Query individual models directly without multi-round deliberation:

```bash
# Single model
python3 ${SKILL_ROOT}/scripts/council.py --direct --models claude --query "[question]" --human

# Multiple models (sequential, no synthesis)
python3 ${SKILL_ROOT}/scripts/council.py --direct --models claude,gemini --query "[question]" --human
```

**Detect direct mode from natural language:**

| User says | Use |
|-----------|-----|
| "What does Claude think about X?" | `--direct --models claude` |
| "Just Gemini's opinion please" | `--direct --models gemini` |
| "Quick answer from Codex" | `--direct --models codex` |
| "Skip the council, ask Claude" | `--direct --models claude` |
| "No debate—just Gemini" | `--direct --models gemini` |
| "Claude only: is this correct?" | `--direct --models claude` |
| "Run this by Codex real quick" | `--direct --models codex` |
| "Get Claude and Gemini's take" | `--direct --models claude,gemini` |
| "All models, no synthesis" | `--direct --models claude,gemini,codex` |
| "Ask the council about X" | Full deliberation (no --direct) |

**Trigger keywords for direct mode:**
- Exclusivity: "only", "just", "solo", "single"
- Speed: "quick", "fast", "real quick"
- Bypass: "skip the council", "no debate", "no deliberation"
- Model-specific: "Claude thinks", "Gemini's opinion", "Codex's take"

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

## Features (v1.2.0)

### Semantic Evidence Matching
Evidence-claim relevance uses embedding similarity (sentence-transformers) instead of term overlap:
- Local embeddings via `all-MiniLM-L6-v2` model
- Falls back to term overlap if unavailable
- Catches semantically equivalent text without lexical match

### STORM Pipeline
Evidence-grounded deliberation inspired by Stanford's STORM:
- **KnowledgeBase**: Tracks claims, sources, decisions
- **Moderator**: Detects workflow type, triggers retrieval on shallow consensus
- **Researcher**: Retrieves evidence for claims
- **EvidenceJudge**: Scores claims against sources with semantic matching
- **Convergence**: Evidence-aware (agreement + coverage + objections)

### Resilience
- Circuit breaker per model (3 failures → OPEN)
- Chairman failover chain: claude → gemini → codex
- Adaptive timeout based on response history
- Graceful degradation (min quorum: 2 models)

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

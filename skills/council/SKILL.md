---
name: council
description: This skill should be used when the user asks to "ask the council", "debate this", "vote on", "get multiple opinions", "what do other AIs think", "peer review", "challenge my design", or requests collective AI intelligence from multiple models (Claude Opus, Gemini Pro, GPT/Codex).
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

Orchestrate collective intelligence from Claude Opus, Gemini Pro, and Codex through **multi-round deliberation**, persona-based analysis, anonymous peer review, and synthesis.

## Persona Assignments

| Model | Persona | Focus |
|-------|---------|-------|
| Claude | Chief Architect | Strategic design, architecture, maintainability |
| Gemini | Security Officer | Security, vulnerabilities, compliance, risk |
| Codex | Performance Engineer | Performance, algorithms, efficiency, scalability |

## Quick Start

### Basic Invocation

```bash
python3 ${SKILL_ROOT}/scripts/council.py \
  --query "[question]" \
  --mode consensus \
  --max-rounds 3
```

### With Code Context (Recommended for Code Reviews)

**Step 1**: Create manifest file listing files to analyze:

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

**Step 2**: Invoke with context file:

```bash
python3 ${SKILL_ROOT}/scripts/council.py \
  --query "Review this authentication module" \
  --context-file /tmp/council_manifest.md \
  --mode consensus
```

Council.py parses lines starting with `### filename.ext` and loads those files automatically.

## Deliberation Process

### 3-Stage Pipeline

1. **Multi-Round Opinions**: All 3 personas provide analysis, see each other's arguments, provide rebuttals
2. **Peer Review**: Chairman scores responses on accuracy, completeness, reasoning, clarity
3. **Synthesis**: Chairman produces final answer incorporating all rounds

### Round Flow

**Round 1 (Parallel)**:
- All 3 models provide initial analysis with persona lens
- Progress: "Chief Architect responded (16.2s)"

**Round 2+ (Rebuttals)**:
- Each model receives anonymized summaries of OTHER models' arguments
- Provides rebuttals to disagreements, concessions to agreements
- Signals convergence if consensus reached

**Convergence Detection**:
- Threshold: 0.8 (60% confidence + 40% explicit signals)
- If converged: Stop early, proceed to synthesis
- If not: Continue to max_rounds (default: 3)

**Quorum**: Minimum 2 valid responses required per round

## Deliberation Modes

### Consensus (Default)
- **Use when**: Factual questions, technical validation, design decisions
- **Process**: Multi-round with convergence detection
- **Output**: Synthesized answer with confidence score

### Debate
- **Use when**: Controversial topics, binary decisions, competing approaches
- **Personas**: Neutral Analyst, Advocate FOR, Advocate AGAINST
- **Output**: Balanced analysis of both sides

```bash
python3 ${SKILL_ROOT}/scripts/council.py \
  --query "Microservices vs monolith for startups" \
  --mode debate
```

### Devil's Advocate
- **Use when**: Stress-testing proposals, security reviews, finding edge cases
- **Personas**: Purple Team (Integrator), Red Team (Attacker), Blue Team (Defender)
- **Output**: Critique with identified weaknesses and mitigations

```bash
python3 ${SKILL_ROOT}/scripts/council.py \
  --query "Proposal: Implement E2EE using AES-256" \
  --mode devil_advocate
```

### Vote
- **Use when**: Binary or multiple choice decisions
- **Output**: Vote tally + majority recommendation

### Specialist
- **Use when**: Domain-specific expertise needed
- **Routing**: GPU/ML → Gemini, Architecture → Claude, Code → Codex

## Security Features

Defense-in-depth protection (see `references/security.md` for details):

- **Shell Injection Prevention**: Detects `;`, `&`, `|`, backticks, `$()`
- **Prompt Injection Detection**: Blocks instruction overrides, privilege escalation
- **Secret Redaction**: Auto-redacts API keys, tokens, passwords in input/output
- **DoS Protection**: Query limit 50K chars, context limit 200K chars

## Resilience

Automatic graceful degradation (see `references/resilience.md` for details):

| Level | Condition | Confidence Penalty |
|-------|-----------|-------------------|
| FULL | All models available | 0% |
| DEGRADED | >=2 models available | 10% |
| MINIMAL | <2 models | 25% (cannot continue) |

Features:
- **Circuit Breaker**: Excludes repeatedly failing models (3+ failures → OPEN)
- **Adaptive Timeout**: Learns from response times, adjusts dynamically
- **Chairman Failover**: claude → gemini → codex if primary fails

## Error Handling

| Error | Action |
|-------|--------|
| CLI timeout (>60s) | Mark ABSTENTION, continue with available |
| Quorum failure (<2) | Inform user, suggest Claude-only retry |
| Invalid JSON | Extract text, score lower |
| Validation failure | Reject if critical (shell injection) |

## CLI Tools

```bash
# Gemini
gemini "Your prompt"

# Codex
codex exec "Your prompt"

# Claude (chairman)
claude "Your prompt"
```

**Always use Claude as chairman** for synthesis.

## Response Format

Present results as (see `references/output-format.md` for full template):

```markdown
## Council Deliberation: [Question]

**Participants**: Chief Architect, Security Officer, Performance Engineer
**Rounds**: 2 of 3 (converged)
**Convergence**: 0.944

### Round 1: Initial Positions
[key points per persona]

### Round 2: Rebuttals
[counter-arguments and concessions]

### Council Consensus
[synthesized answer]

**Confidence**: 0.91
**Dissenting View**: [if significant]
```

## Configuration

Council can be configured via `council.config.yaml`:

```yaml
providers: [claude, gemini, codex]
chairman: claude
timeout: 420  # 7 minutes - Codex needs time for code exploration
max_rounds: 3
mode: adaptive
convergence_threshold: 0.8
min_quorum: 2
```

## Reference Files

- `references/modes.md` - Deliberation mode details
- `references/prompts.md` - Prompt templates
- `references/schemas.md` - JSON response schemas
- `references/security.md` - Security implementation
- `references/resilience.md` - Graceful degradation details
- `references/failure-modes.md` - Error handling and recovery
- `references/output-format.md` - Response templates
- `references/examples.md` - Detailed usage examples

## Known Limitations / Technical Debt

The following issues are documented for future improvement:

| Issue | Severity | Description |
|-------|----------|-------------|
| `sys.path.insert()` usage | Medium | Multiple files use `sys.path.insert(0, ...)` for imports. Proper fix requires packaging refactor with `pyproject.toml`. Functional but non-idiomatic. |
| `gather_opinions` location | Low | Shared by all modes but lives in `modes/consensus.py`. Should be in `core/` as infrastructure. |
| `run_council` size | Low | ~345 lines handling multiple concerns. Could be refactored into phase functions for maintainability. |

These are non-blocking issues that don't affect functionality or security.

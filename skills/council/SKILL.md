---
name: Council
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

Each model assumes a specialized role to provide diverse perspectives:

- **Claude (Chief Architect)**: Strategic design, architectural trade-offs, long-term maintainability
- **Gemini (Security Officer)**: Security analysis, vulnerabilities, compliance, risk assessment
- **Codex (Performance Engineer)**: Performance optimization, algorithms, efficiency, scalability

## Execution Workflow

When user triggers council (e.g., "ask the council: Should we use TypeScript?"):

**IMPORTANT**: Always use ALL 3 models (Claude, Gemini, Codex) with their assigned personas.

**IMPORTANT**: Multi-round deliberation with feedback loops - models see each other's arguments and provide rebuttals.

**IMPORTANT**: Provide progress updates showing rounds, personas, and convergence status.

### Invocation

**IMPORTANT**: If user's question references code, files, or specific implementations:
1. Use Read tool to get the code/files FIRST
2. Pass code as --context argument to provide full context to all models

**Basic invocation** (no code context):
```bash
python3 ${SKILL_ROOT}/scripts/council.py \
  --query "[user's question]" \
  --mode consensus \
  --max-rounds 3
```

**With code context** (when user references code):
```bash
# First read the relevant code
Read file_path

# Then invoke council with context
python3 ${SKILL_ROOT}/scripts/council.py \
  --query "[user's question]" \
  --context "[code content from Read tool]" \
  --mode consensus \
  --max-rounds 3
```

**Example - Code review**:
```
User: "Ask the council: Is this authentication function secure?"
      [shows code snippet or references a file]

You should:
1. Read the file if referenced, OR use the code snippet from conversation
2. Call council with --context containing the code
3. Models will analyze the code from all 3 perspectives
```

### Multi-Round Deliberation Process

#### Round 1: Initial Positions (Parallel)

**Tell user**: "Starting council deliberation with Chief Architect, Security Officer, Performance Engineer..."

All 3 models provide initial analysis with their persona lens:
- **Chief Architect (Claude)**: Architecture and design perspective
- **Security Officer (Gemini)**: Security and risk perspective
- **Performance Engineer (Codex)**: Performance and efficiency perspective

**Progress updates**:
- "Round 1 started (max 3 rounds)"
- "✓ Chief Architect responded (23.1s)"
- "✓ Security Officer responded (24.5s)"
- "✓ Performance Engineer responded (9.0s)"

#### Round 2+: Rebuttals and Refinement

**Tell user**: "Round 2: Models reviewing each other's arguments..."

Each model receives **anonymized summaries** of what OTHER models said:
- See their key points, confidence levels, and reasoning
- Provide **rebuttals** to arguments they disagree with
- Offer **concessions** where they agree with others
- Signal **convergence** if they've reached consensus

**Progress updates**:
- "Round 2 started"
- "✓ Chief Architect rebuttal (31.0s)"
- "✓ Security Officer rebuttal (32.2s)"
- "✓ Performance Engineer rebuttal (12.0s)"
- "Convergence check: score 0.944 (converged ✓)"

#### Convergence Detection

After each round (starting round 2), check convergence based on:
1. **Explicit signals**: Models indicate they've reached agreement
2. **High confidence**: Average confidence ≥ 0.8 across models
3. **Low uncertainty**: Models report few remaining doubts

**Convergence threshold**: 0.8 (combination of confidence and signals)

**If converged**: Stop iteration early, proceed to synthesis
**If not converged**: Continue to next round (up to max_rounds)

### Peer Review (Anonymized)

**Tell user**: "Conducting anonymous peer review..."

Chairman (Claude) scores final round responses:
- **Accuracy** (1-5): Factual correctness
- **Completeness** (1-5): Thoroughness of coverage
- **Reasoning** (1-5): Logic quality
- **Clarity** (1-5): Communication effectiveness

Identify contradictions between perspectives.

### Final Synthesis

**Tell user**: "Chairman synthesizing all rounds..."

Chairman (Claude) produces final answer incorporating:
1. **All rounds of deliberation** (not just final round)
2. Strongest arguments from each persona
3. Contradiction resolutions with evidence
4. Remaining uncertainties
5. Dissenting views if significant
6. Overall confidence score (0.0-1.0)
7. Number of rounds completed and convergence status

## Response Format

Present results as:

```markdown
## Council Deliberation: [Question]

**Participants**: Chief Architect (Claude), Security Officer (Gemini), Performance Engineer (Codex)
**Rounds Completed**: 2 of 3 (converged at round 2)
**Convergence Score**: 0.944 (converged ✓)
**Session Duration**: 104.4s

### Round 1: Initial Positions

**Chief Architect** (Confidence: 0.85)
- [key architectural points]

**Security Officer** (Confidence: 0.90)
- [key security points]

**Performance Engineer** (Confidence: 0.80)
- [key performance points]

### Round 2: Rebuttals and Refinement

**Chief Architect** (Confidence: 0.90)
- Rebuttals: [counter-arguments to other perspectives]
- Concessions: [points of agreement]

**Security Officer** (Confidence: 0.95)
- Rebuttals: [counter-arguments]
- Concessions: [points of agreement]

**Performance Engineer** (Confidence: 0.92)
- Rebuttals: [counter-arguments]
- Concessions: [points of agreement]

**Convergence**: ✓ Achieved (score: 0.944)

### Peer Review Scores (Final Round)

| Persona | Accuracy | Completeness | Reasoning | Clarity | Total |
|---------|----------|--------------|-----------|---------|-------|
| Chief Architect        | 5        | 5            | 5         | 5       | 20/20 |
| Security Officer       | 4        | 4            | 4         | 4       | 16/20 |
| Performance Engineer   | 4        | 4            | 5         | 5       | 18/20 |

### Key Contradictions

- **Chief Architect** emphasizes X while **Security Officer** prioritizes Y
  - **Resolution**: [synthesis showing both are valid under different constraints]

### Council Consensus

[Synthesized answer incorporating all rounds and perspectives]

**Final Confidence**: 0.91 (based on convergence and peer review)
**Dissenting View**: [If significant disagreement remains, present minority perspective]
```

## Deliberation Modes

### Consensus (Default)
- **Use when**: Factual questions, technical validation, design decisions
- **Process**: Multi-round deliberation with convergence detection
  1. Round 1: All 3 personas provide initial analysis
  2. Round 2+: Models see others' arguments, provide rebuttals/concessions
  3. Convergence check after each round (threshold: 0.8)
  4. Early termination if converged, or continue to max_rounds (default: 3)
  5. Peer review and synthesis by chairman
- **Quorum**: Minimum 2 valid responses required per round
- **Convergence signals**: High confidence (≥0.8) + explicit agreement signals
- **Max rounds**: 3 (configurable with --max-rounds)

### Debate
- **Use when**: Controversial topics, binary decisions, evaluating competing approaches
- **Persona assignments**:
  - **Claude = Neutral Analyst**: Objective analysis of both sides without taking sides
  - **Gemini = Advocate FOR**: Builds strongest case in favor of proposition
  - **Codex = Advocate AGAINST**: Builds strongest case against proposition
- **Process**: Adversarial multi-round argumentation
  1. Round 1: Initial positions (FOR builds case, AGAINST builds counter-case, NEUTRAL analyzes)
  2. Round 2+: Each sees others' arguments, provides rebuttals and evidence
  3. Convergence check (may or may not converge - disagreement is valid output)
  4. Final synthesis presents both cases fairly with dissenting views
- **Example invocation**:
```bash
python3 skills/council/scripts/council.py \
  --query "Microservices architecture is better than monolithic architecture for startups" \
  --mode debate \
  --max-rounds 3
```
- **Expected outcome**: Balanced analysis showing strongest arguments for each side, with chairman synthesis identifying when each approach is appropriate

### Vote
- Use when: Binary or multiple choice decisions
- Process: Each model votes with justification
- Output: Vote tally + majority recommendation

### Specialist
- Use when: Domain-specific expertise needed
- Process: Route to best-suited model, others validate
- Routing:
  - GPU/ML/Math → Gemini Pro (strong technical compute)
  - Architecture/Design → Claude Opus (reasoning)
  - Code generation → Codex (coding specialist)

### Devil's Advocate (Red/Blue/Purple Team)
- **Use when**: Stress-testing proposals, security reviews, finding edge cases and failure modes
- **Persona assignments**:
  - **Claude = Purple Team (Integrator)**: Synthesizes Red Team critiques and Blue Team defenses, identifies valid concerns vs mitigated risks
  - **Gemini = Red Team (Attacker)**: Systematically finds every weakness, edge case, security flaw, and failure mode
  - **Codex = Blue Team (Defender)**: Defends proposal, justifies design decisions, shows how concerns are mitigated
- **Process**: Attack-defend-integrate methodology
  1. Round 1: Red Team identifies vulnerabilities, Blue Team justifies approach, Purple Team analyzes both
  2. Round 2+: Red Team sees defenses and finds deeper flaws, Blue Team addresses new critiques, Purple Team refines analysis
  3. Convergence indicates Red/Blue reached understanding (not necessarily agreement)
  4. Final synthesis: Purple Team's integrated view of which concerns are valid vs adequately mitigated
- **Example invocation**:
```bash
python3 skills/council/scripts/council.py \
  --query "Proposal: Implement end-to-end encryption for all user data using AES-256" \
  --mode devil_advocate \
  --max-rounds 3
```
- **Expected outcome**: Thorough critique with identified weaknesses, proposed mitigations, and recommendation on whether to proceed (often conditional approval with requirements)

## Security Features

**NEW**: Phase 1 security hardening implemented with defense-in-depth protection:

### Input Validation & Sanitization

All user inputs are automatically validated and sanitized before processing:

1. **Shell Injection Prevention (CWE-78)**
   - Detects shell operators: `;`, `&`, `|`, backticks, `$()`, redirection
   - Validates against path traversal attacks (`../..`)
   - Prevents command injection and RCE

2. **Prompt Injection Detection (OWASP LLM01)**
   - Detects instruction override attempts
   - Blocks privilege escalation patterns
   - Prevents system tag injection
   - Stops role redefinition attacks

3. **Secret Redaction (OWASP LLM06)**
   - Automatically redacts API keys (OpenAI, Google, GitHub, AWS)
   - Redacts OAuth tokens, JWT tokens, passwords
   - Applied to both input context AND output
   - Prevents accidental secret leakage to LLM providers

4. **DoS Protection**
   - Query length limit: 50,000 characters
   - Context length limit: 200,000 characters
   - max_rounds bounded: 1-10 rounds
   - timeout bounded: 10-300 seconds

### Validation Modes

- **Strict mode**: Reject requests with any violations
- **Sanitize mode** (default): Redact secrets, warn on violations, continue processing

### Testing

Run security test suite:
```bash
python3 tests/test_security.py
```

Tests verify protection against 22+ attack vectors across 6 categories.

## Graceful Degradation

The Council skill continues operating when models fail, automatically adjusting confidence and tracking degradation state.

### Degradation Levels

| Level | Condition | Confidence Penalty | Can Continue? |
|-------|-----------|-------------------|---------------|
| **FULL** | All requested models available | 0% | Yes |
| **DEGRADED** | Some models failed, ≥2 remaining | 10% | Yes |
| **MINIMAL** | <2 models available (below quorum) | 25% | No |

### How It Works

1. **Model Tracking**: Each session tracks expected vs available models
2. **Failure Recording**: When a model times out, errors, or has circuit breaker open, it's marked unavailable
3. **Confidence Adjustment**: Final confidence is reduced based on degradation level
4. **Recovery Detection**: If a model recovers mid-session, it's re-included

### Confidence Adjustment Example

```
Raw confidence from synthesis: 0.90

FULL mode:     0.90 × 1.00 = 0.90 (no penalty)
DEGRADED mode: 0.90 × 0.90 = 0.81 (10% penalty)
MINIMAL mode:  0.90 × 0.75 = 0.675 (25% penalty)
```

### Adaptive Timeout

The system learns from model response times and adjusts timeouts dynamically:

- **Initial**: Uses configured timeout (default: 60s)
- **After 3+ samples**: Calculates p95 latency × 1.5 safety margin
- **Bounds**: Never below 50% or above 200% of base timeout

Example:
```
Model: claude
Observed latencies: [2000ms, 2500ms, 3000ms, 2200ms]
P95 latency: ~3000ms
Adaptive timeout: 3000ms × 1.5 / 1000 = 4.5s → clamped to 30s (50% of 60s base)
```

### Circuit Breaker Integration

Models with open circuit breakers are automatically excluded:

1. **CLOSED** (normal): Model is called normally
2. **OPEN** (failing): Model skipped, recorded as unavailable
3. **HALF_OPEN** (testing): Model included, success closes circuit

### Observable Events

Degradation emits structured events for monitoring:

```json
{"type": "model_degraded", "model": "gemini", "reason": "TIMEOUT", "level": "degraded"}
{"type": "model_recovered", "model": "gemini", "level": "full"}
{"type": "degradation_status", "level": "degraded", "available_models": ["claude", "codex"]}
{"type": "PartialResultReturned", "degradation_level": "degraded", "raw_confidence": 0.9, "adjusted_confidence": 0.81}
```

### Output Fields

Final output includes degradation information:

```json
{
  "confidence": 0.81,
  "raw_confidence": 0.90,
  "degradation": {
    "level": "degraded",
    "expected_models": ["claude", "gemini", "codex"],
    "available_models": ["claude", "codex"],
    "failed_models": {"gemini": "TIMEOUT"},
    "recovered_models": [],
    "availability_ratio": 0.667
  },
  "adaptive_timeout": {
    "claude": {"samples": 4, "avg_ms": 2425, "adaptive_timeout_s": 30}
  }
}
```

### Response Format with Degradation

When operating in degraded mode, indicate this to the user:

```markdown
## Council Deliberation: [Question]

**Participants**: Chief Architect (Claude), ~~Security Officer (Gemini)~~, Performance Engineer (Codex)
**Status**: DEGRADED (1 model unavailable)
**Confidence**: 0.81 (adjusted from 0.90 due to degradation)

⚠️ **Note**: Security Officer (Gemini) was unavailable due to timeout.
Results are based on 2 of 3 requested perspectives.
```

## Error Handling

- **CLI timeout** (>60s): Mark as ABSTENTION, continue with available responses
- **Quorum failure** (<2 responses): Inform user, suggest retry with just Claude analysis
- **Invalid JSON**: Extract key points from raw text, score lower
- **Contradictions unresolvable**: Present both views clearly, let user decide
- **Validation failures**: Emit warnings, reject if critical (shell injection)

## Examples

### Example 1: Technical Question

User: "Ask the council: What's the best database for real-time chat?"

Execute:
```bash
python3 skills/council/scripts/council.py \
  --query "What's the best database for real-time chat?" \
  --mode consensus \
  --max-rounds 3
```

**Progress shown to user**:
1. "Starting council deliberation with Chief Architect, Security Officer, Performance Engineer..."
2. "Round 1 started (max 3 rounds)"
3. "✓ Chief Architect responded (16.2s)" - Analyzes architecture trade-offs
4. "✓ Security Officer responded (12.3s)" - Evaluates security implications
5. "✓ Performance Engineer responded (3.1s)" - Assesses performance characteristics
6. "Round 2 started"
7. "✓ Chief Architect rebuttal (20.1s)" - Responds to performance concerns
8. "✓ Security Officer rebuttal (18.5s)" - Addresses architecture suggestions
9. "✓ Performance Engineer rebuttal (5.2s)" - Validates security requirements
10. "Convergence check: score 0.91 (converged ✓)"
11. "Chairman synthesizing all rounds..."
12. **Final synthesis**: "Consensus recommends Redis for pub/sub + PostgreSQL for persistence..."

### Example 2: Debate Mode

User: "Debate this: Microservices architecture is better than monolithic architecture for startups"

Execute:
```bash
python3 skills/council/scripts/council.py \
  --query "Microservices architecture is better than monolithic architecture for startups" \
  --mode debate \
  --max-rounds 2
```

**Progress shown to user**:
1. "Starting council session (mode: debate, max_rounds: 2)"
2. "Round 1 started"
3. "✓ Neutral Analyst responded (32.1s)" - Analyzes both sides objectively
4. "✓ Advocate FOR responded (41.5s)" - Builds strongest case for microservices
5. "✓ Advocate AGAINST responded (6.6s)" - Builds strongest case for monolith
6. "Round 2 started"
7. "✓ Neutral Analyst rebuttal (33.9s)" - Refines analysis based on arguments
8. "✓ Advocate FOR rebuttal (38.2s)" - Counters AGAINST arguments
9. "✓ Advocate AGAINST rebuttal (15.0s)" - Counters FOR arguments
10. "Convergence check: score 0.846 (converged ✓)"
11. "Chairman synthesizing..."
12. **Final synthesis**: "The debate is fundamentally context-dependent. For most early-stage startups (<15 engineers), a well-structured modular monolith is optimal. Microservices make sense with: strict compliance boundaries, >15 engineers, validated product-market fit, or proven scaling bottlenecks. Start with modular monolith, extract services based on evidence, not prophecy."
13. **Dissenting view**: "Strong disagreement with FOR advocate's claim that microservices are 'unequivocally better' - empirical evidence from Shopify, GitHub, Basecamp contradicts this. The 'inevitable monolithic trap' is not inevitable with proper modularity."

### Example 3: Code Review (Consensus Mode)

User: "Peer review this authentication code: [paste code]"

Execute:
```bash
python3 skills/council/scripts/council.py \
  --query "Review this authentication code for security issues: [code]" \
  --mode consensus \
  --max-rounds 2
```

Process:
1. Chief Architect reviews architecture and design patterns
2. Security Officer reviews for vulnerabilities and security best practices
3. Performance Engineer reviews for efficiency and scalability
4. Round 2: Models refine reviews based on each other's findings
5. Peer review scores on accuracy of issues found
6. Final synthesis: prioritized list of fixes with consensus recommendations

### Example 4: Devil's Advocate Mode (Security Proposal)

User: "Challenge this proposal: Implement end-to-end encryption for all user data using AES-256"

Execute:
```bash
python3 skills/council/scripts/council.py \
  --query "Proposal: Implement end-to-end encryption for all user data in our chat application using AES-256" \
  --mode devil_advocate \
  --max-rounds 2
```

**Progress shown to user**:
1. "Starting council session (mode: devil_advocate, max_rounds: 2)"
2. "Round 1 started"
3. "✓ Purple Team (Integrator) responded (28.3s)" - Initial synthesis of concerns
4. "✓ Red Team (Attacker) responded (35.8s)" - Identifies weaknesses: key management gaps, endpoint security, metadata exposure
5. "✓ Blue Team (Defender) responded (9.5s)" - Justifies AES-256 choice, argues for implementation feasibility
6. "Round 2 started"
7. "✓ Purple Team counter-arguments (35.6s)" - Refines integration based on new critiques
8. "✓ Red Team counter-arguments (45.9s)" - Deepens attack: user behavior failures, complexity as vulnerability
9. "✓ Blue Team counter-arguments (11.5s)" - Addresses new critiques with mitigations
10. "Convergence check: score 0.793 (not converged)" - Valid disagreement remains
11. "Chairman synthesizing..."
12. **Final synthesis**: "CONDITIONAL APPROVAL with mandatory requirements: Must specify protocol (Signal Protocol/MLS), implement hardware-backed key storage, design key recovery mechanism, protect metadata, use audited libraries, test on target devices. The proposal as stated is dangerously incomplete - AES-256 is <5% of the security architecture."
13. **Dissenting view (Red Team)**: "The complexity required for secure E2EE is itself a vulnerability. Every component (key backup, multi-device sync, group chat) increases attack surface. Historical E2EE implementations have had critical flaws. If the team lacks deep cryptographic expertise, simpler server-side encryption may provide better practical security than poorly implemented E2EE."

## CLI Tool Invocations

### Gemini
```bash
gemini "Your prompt here"
```
Returns plain text response. **Always use in parallel with Codex.**

### Codex
```bash
codex exec "Your prompt here"
```
Use `exec` subcommand for non-interactive mode. **Always use in parallel with Gemini.**

### Chairman Default
**Always use Claude as chairman** for synthesis (Stage 2 peer review + Stage 3 synthesis).
Chairman must be different from opinion-gathering models when possible.

### Error Handling for CLIs
If CLI not available or times out:
- Log as ABSTENTION
- Continue with available models
- Note in final synthesis: "Gemini unavailable, consensus based on Claude + Codex"

## Reference Files

For detailed information:
- `references/modes.md` - Deep dive on 5 deliberation modes
- `references/prompts.md` - Prompt templates for each stage
- `references/schemas.md` - JSON response schemas

**Note**: This is a personal development skill designed for trusted single-user scenarios. Security mitigations for untrusted input are not included as they're unnecessary overhead for personal CLI usage.

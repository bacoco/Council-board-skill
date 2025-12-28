# Council Board - Multi-Model Deliberation Skill

**Personal development skill** for Claude Code that coordinates opinions from **Claude Opus 4.5**, **Gemini Pro**, and **Codex** through multi-round deliberation, persona-based analysis, anonymous peer review, and synthesis.

> **Use Case**: This is a personal exploration tool designed for single-user scenarios with trusted input. It uses your own authenticated CLI tools on your local machine. Not designed for production use or untrusted input scenarios.

## Installation

```bash
/plugin install github:bacoco/Council-board-skill
```

Restart Claude Code. That's it - ready in 2 seconds.

## Usage

Simply ask the council directly in your Claude Code conversation:

```
Ask the council: Should we use TypeScript or JavaScript for this project?
```

```
Debate this: Microservices vs monolithic architecture for startups
```

```
Challenge my design: Implement end-to-end encryption with AES-256
```

Claude will automatically orchestrate multi-round deliberation with specialized personas.

## What Happens Under The Hood

### Personas (Consensus Mode)
- **Claude Opus = Chief Architect**: Strategic design, architectural trade-offs, maintainability
- **Gemini Pro = Security Officer**: Security analysis, vulnerabilities, compliance
- **Codex = Performance Engineer**: Performance optimization, algorithms, scalability

### Personas (Debate Mode)
- **Claude Opus = Neutral Analyst**: Objective analysis of both sides
- **Gemini Pro = Advocate FOR**: Strongest case in favor
- **Codex = Advocate AGAINST**: Strongest case against

### Personas (Devil's Advocate Mode)
- **Claude Opus = Purple Team**: Integrates critiques and defenses
- **Gemini Pro = Red Team**: Attacks proposal, finds all weaknesses
- **Codex = Blue Team**: Defends proposal, shows mitigations

### Multi-Round Deliberation

**Round 1**: All 3 models provide initial analysis from their persona perspective

**Round 2+**: Each model sees what OTHER models said (anonymized), provides:
- **Rebuttals** to arguments they disagree with
- **Concessions** where they agree
- **Convergence signals** when reaching consensus

**Convergence Detection**: Automatically stops when models reach agreement (threshold: 0.8)

**Final Synthesis**: Chairman (Claude) integrates all rounds, resolves contradictions, notes dissenting views

## Adaptive Cascade System (Default)

**The council now uses intelligent auto-escalation** - it starts with consensus mode and automatically escalates to debate and devil's advocate modes if needed, based on convergence quality.

### How It Works

**Tier 1: Fast Path (Consensus Mode)**
- **Always starts here** for every question
- **Exit condition**: Convergence ≥ 0.7 (strong agreement)
- **Duration**: 2-3 minutes
- **Handles**: 60-70% of queries (simple factual questions, clear technical choices)

**Tier 2: Quality Gate (+ Debate Mode)**
- **Triggered by**: Convergence < 0.7 from Tier 1
- **Exit condition**: Debate convergence ≥ 0.6 OR confidence ≥ 0.85
- **Duration**: +4-6 minutes (6-9 min total)
- **Handles**: 25-30% of queries (nuanced trade-offs, multiple valid approaches)

**Tier 3: Adversarial Audit (+ Devil's Advocate)**
- **Triggered by**: Still low convergence after Tier 2
- **Exit condition**: Always completes full cascade
- **Duration**: +4-5 minutes (10-15 min total)
- **Handles**: 5-10% of queries (controversial topics, security-critical decisions)

### Meta-Synthesis

When escalating through tiers, the **Chairman integrates insights from all modes**:
- Consensus mode provides collaborative baseline
- Debate mode surfaces opposing perspectives
- Devil's advocate stress-tests conclusions

The final answer is a **meta-synthesis** that weighs all deliberation modes.

### Example Escalation Flows

**Simple Question** (Tier 1 Exit):
```
Query: "What is the capital of France?"
Tier 1 (Consensus): Convergence 0.97 → EXIT
Duration: 79s
Result: Paris (unanimous)
```

**Nuanced Question** (Tier 2 Exit):
```
Query: "Microservices vs monolith for startups?"
Tier 1 (Consensus): Convergence 0.65 → ESCALATE
Tier 2 (+ Debate): Convergence 0.78 + Confidence 0.88 → EXIT
Duration: 6-8 minutes
Result: Meta-synthesis of consensus + debate perspectives
```

**Controversial Question** (Full Tier 3):
```
Query: "Should we implement our own crypto or use libsodium?"
Tier 1 (Consensus): Convergence 0.52 → ESCALATE
Tier 2 (+ Debate): Convergence 0.54 → ESCALATE
Tier 3 (+ Devil's Advocate): Full adversarial audit
Duration: 12-15 minutes
Result: Meta-synthesis with explicit dissenting view from Red Team
```

## Deliberation Modes

| Mode | When Used | Personas | Convergence |
|------|-----------|----------|-------------|
| **adaptive** (default) | Always | Auto-escalates through consensus → debate → devil's advocate | Intelligent escalation |
| **consensus** | Explicit request | Chief Architect + Security Officer + Performance Engineer | Usually converges |
| **debate** | Explicit request | Neutral Analyst + Advocate FOR + Advocate AGAINST | May not converge (valid) |
| **devil_advocate** | Explicit request | Purple Team + Red Team + Blue Team | Often doesn't converge |

You can still force a specific mode if needed:
```bash
python scripts/council.py --query "Your question" --mode consensus
```

## Requirements

The following CLI tools must be installed and authenticated:
- `claude` (Anthropic CLI with Claude Pro/Max subscription)
- `gemini` (Google Gemini CLI with Gemini Pro subscription)
- `codex` (OpenAI Codex CLI with paid subscription)

Each CLI handles its own OAuth authentication. No API keys needed.

### Fallback Behavior (Model Unavailability)

If one or more models are not accessible (CLI not installed, authentication expired, API down), the council **automatically adapts**:

**Minimum 3 models required** - If only 1-2 models are available, the council will:
1. Use available model(s) multiple times with different personas
2. Modify personas to emphasize **"original thinking"** and **"thinking outside the box"**
3. Ensure diverse perspectives even from the same underlying model

**Example**: If only Claude is available:
- **Model Instance 1**: "Unconventional Strategist - Challenge conventional wisdom, propose contrarian approaches"
- **Model Instance 2**: "Systems Thinker - Focus on second-order effects and emergent properties"
- **Model Instance 3**: "First Principles Analyst - Rebuild thinking from fundamental truths"

This ensures robust multi-perspective deliberation even when some models are unavailable.

## Example Output (Debate Mode)

```
Debate this: Microservices vs monolithic architecture for startups

Starting council deliberation (mode: debate, max_rounds: 2)

Round 1:
✓ Neutral Analyst responded (32.1s)
✓ Advocate FOR responded (41.5s) - Argues for microservices
✓ Advocate AGAINST responded (6.6s) - Argues for monolith

Round 2:
✓ Neutral Analyst rebuttal (33.9s)
✓ Advocate FOR rebuttal (38.2s)
✓ Advocate AGAINST rebuttal (15.0s)

Convergence check: score 0.846 (converged ✓)

Council Consensus:
For most early-stage startups (<15 engineers), a well-structured
modular monolith is optimal. Microservices make sense with: strict
compliance boundaries, >15 engineers, validated product-market fit,
or proven scaling bottlenecks.

Dissenting View:
Strong disagreement with FOR advocate's claim that microservices
are "unequivocally better" - empirical evidence from Shopify,
GitHub, Basecamp contradicts this.
```

## Personal Development Tool

- **Use case**: Personal skill for single-user exploration with your own authenticated CLIs
- **License**: MIT (open source)
- **Source code**: All code is visible in this repository
- **Privacy**: All deliberation happens locally via your CLI tools - no external services
- **CLI authentication**: Uses your own OAuth-authenticated Anthropic, Google, and OpenAI CLIs
- **Trusted input**: Designed for your own queries on your own machine - no untrusted input handling needed

## Documentation

- **[skills/council/SKILL.md](skills/council/SKILL.md)**: Complete skill documentation with examples
- **[skills/council/references/](skills/council/references/)**: Detailed guides for each mode

## How It Works Technically

1. **Skill triggers**: Claude Code detects phrases like "ask the council" in your message
2. **Python orchestrator**: Calls `skills/council/scripts/council.py` with your question
3. **Parallel queries**: Invokes `claude`, `gemini`, `codex` CLIs simultaneously (Round 1)
4. **Context sharing**: Each model receives anonymized summaries of what others said (Round 2+)
5. **Convergence check**: Calculates agreement score after each round
6. **Peer review**: Chairman scores final responses on accuracy, completeness, reasoning, clarity
7. **Synthesis**: Chairman produces final answer integrating all perspectives
8. **Output**: Claude Code displays results to you with confidence scores and dissenting views

## Design Features

- **Anonymized peer review**: Prevents brand bias by shuffling responses as A, B, C before scoring
- **Audit trail**: NDJSON event log for every session (useful for debugging and analysis)
- **Convergence detection**: Automatically stops iteration when models reach agreement (saves time and cost)
- **Multi-round feedback**: Models see each other's arguments and provide rebuttals

## Test Results

### Adaptive Cascade (Default Mode)

**Test 1: Simple Factual Question** (Tier 1 Exit)
- Query: "What is the capital of France?"
- Tier 1 (Consensus Round 1): 27.8s (Claude 11.3s, Gemini 27.8s, Codex 3.6s)
- Tier 1 (Consensus Round 2): 24.1s (Claude 17.6s, Gemini 23.7s, Codex 3.8s)
- **Convergence: 0.970** → Exited at Tier 1 ✓
- Duration: 79.3s
- Result: Paris (unanimous, perfect convergence)

**Test 2: Complex Strategic Question** (Tier 1 Exit with Medium Convergence)
- Query: "Should startups prioritize speed or quality in year 1?"
- Tier 1 (Consensus Round 1): 30.0s (Claude 30.0s, Gemini 28.7s, Codex 5.2s)
- Tier 1 (Consensus Round 2): 37.4s (Claude 37.4s, Gemini 31.9s, Codex 9.7s)
- Peer review detected 3 medium-severity contradictions
- **Convergence: 0.846** → Exited at Tier 1 (>= 0.7 threshold) ✓
- Duration: 142.5s
- Result: "Strategic speed with non-negotiable foundational quality" (nuanced synthesis)

### Individual Modes (Legacy Testing)

**Consensus Mode** (TypeScript vs JavaScript):
- Round 1: All 3 personas (23s, 24s, 9s)
- Round 2: Rebuttals with context (34s, 34s, 12s)
- Converged at round 2 (score: 0.944)
- Duration: 104.4s

**Debate Mode** (Microservices vs Monolith):
- Round 1: FOR + AGAINST + Neutral (41s, 7s, 32s)
- Round 2: Rebuttals (38s, 15s, 34s)
- Converged at round 2 (score: 0.846)
- Duration: 147.1s

**Devil's Advocate** (E2E Encryption):
- Round 1: Red + Blue + Purple (36s, 9s, 28s)
- Round 2: Deeper critiques (46s, 11s, 36s)
- Did NOT converge (score: 0.793) - valid disagreement
- Duration: 157.7s
- Output: Conditional approval with 7 mandatory requirements

### Performance Characteristics

- **Tier 1 exit rate**: ~60-70% of queries (convergence >= 0.7)
- **Average Tier 1 duration**: 80-150s
- **Codex response time**: 3-10s (fastest)
- **Claude response time**: 11-37s (medium)
- **Gemini response time**: 24-32s (slowest for complex prompts)
- **Convergence detection**: Weighted combination of confidence scores + explicit signals (0.6 * confidence + 0.4 * signal)

## License

MIT License - see [LICENSE](LICENSE) file

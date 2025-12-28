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

## Deliberation Modes

| Mode | Trigger | Personas | Convergence |
|------|---------|----------|-------------|
| **consensus** | "ask the council" | Chief Architect + Security Officer + Performance Engineer | Usually converges |
| **debate** | "debate this" | Neutral Analyst + Advocate FOR + Advocate AGAINST | May not converge (valid) |
| **devil_advocate** | "challenge my design" | Purple Team + Red Team + Blue Team | Often doesn't converge |

## Requirements

The following CLI tools must be installed and authenticated:
- `claude` (Anthropic CLI with Claude Pro/Max subscription)
- `gemini` (Google Gemini CLI with Gemini Pro subscription)
- `codex` (OpenAI Codex CLI with paid subscription)

Each CLI handles its own OAuth authentication. No API keys needed.

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

## License

MIT License - see [LICENSE](LICENSE) file

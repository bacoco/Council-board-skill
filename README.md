# Council

**Stop trusting a single AI. Get answers that survive expert debate.**

Council orchestrates Claude, Gemini, and Codex to debate, challenge, and synthesize answers together. Three perspectives. One synthesis.

## See It In Action

```
Council Deliberation: Microservices vs Monolith for Startups

Participants: Chief Architect, Security Officer, Performance Engineer
Rounds: 2 (converged)
Confidence: 0.89

Consensus:
For early-stage startups (<15 engineers), start with a modular monolith.
Extract services only when you have evidence of scaling bottlenecks.

Dissenting View:
Performance Engineer notes that certain compliance requirements
(HIPAA, PCI) may justify early service boundaries.
```
*~2 minutes for complex questions*

## Quick Start

```bash
# Add the marketplace (one time)
claude plugin marketplace add bacoco/Council-board-skill

# Install the plugin
claude plugin install council@council-board
```

Then just ask:
```
"Ask the council: Should we use microservices?"
```

<details>
<summary>Prerequisites</summary>

Requires these CLIs installed and authenticated:
- `claude` (Anthropic CLI with Claude Pro/Max)
- `gemini` (Google Gemini CLI)
- `codex` (OpenAI Codex CLI)

</details>

## Example Phrases

| What you say | What happens |
|--------------|--------------|
| "Ask the council: [question]" | 3 models collaborate to answer |
| "Debate this: [topic]" | Models argue FOR and AGAINST |
| "Challenge my design: [proposal]" | Red Team attacks, Blue Team defends |
| "Peer review this code" | Security, architecture, performance review |

## Why This Beats Single-Model Answers

**Pressure-tested answers** — Models don't just answer once. They see each other's arguments and provide rebuttals. Like a real debate.

**Structured disagreement** — When models disagree, it automatically escalates to adversarial review. Disagreement is surfaced, not hidden.

**Knows when to stop** — Stops early when models agree. Continues when they don't. You see the confidence score.

## No API Keys Needed

Uses your existing `claude`, `gemini`, and `codex` CLI subscriptions. No separate API costs — just your regular CLI usage.

<details>
<summary>The 3 Personas</summary>

| Model | Role | Focus |
|-------|------|-------|
| Claude | Chief Architect | Strategy, trade-offs, maintainability |
| Gemini | Security Officer | Vulnerabilities, compliance, risk |
| Codex | Performance Engineer | Speed, algorithms, scalability |

In debate mode: Advocate FOR, Advocate AGAINST, Neutral Analyst.

</details>

<details>
<summary>Resilience</summary>

Council keeps working even when models fail:
- **2 models available**: Continues with 10% confidence penalty
- **Model keeps failing**: Automatically excluded (circuit breaker)
- **Slow model**: Timeout adapts based on history

</details>

<details>
<summary>Documentation</summary>

- [skills/council/SKILL.md](skills/council/SKILL.md) - Full technical docs
- [skills/council/references/](skills/council/references/) - Detailed guides

</details>

## License

MIT - [LICENSE](LICENSE)

# Council - Multi-Model AI Deliberation

**Get 3 AI perspectives on any question.** Council orchestrates Claude, Gemini, and Codex to debate, challenge, and synthesize answers together.

## How It Works

```
You: "Ask the council: Should we use microservices or a monolith?"
```

That's it. Council automatically:
1. Asks 3 different AI models your question
2. Each model sees what the others said
3. They debate and refine their positions
4. A final synthesis emerges with confidence score

**No configuration needed.** Just ask naturally.

## Example Phrases

| What you say | What happens |
|--------------|--------------|
| "Ask the council: [question]" | 3 models collaborate to answer |
| "Debate this: [topic]" | Models argue FOR and AGAINST |
| "Challenge my design: [proposal]" | Red Team attacks, Blue Team defends |
| "What do other AIs think about [topic]?" | Multi-perspective analysis |
| "Peer review this code" | Security, architecture, performance review |

## What Makes It Different

**Multi-round deliberation**: Models don't just answer once. They see each other's arguments (anonymized) and provide rebuttals. Like a real debate.

**Automatic escalation**: Starts with collaborative consensus. If models disagree, automatically escalates to structured debate, then adversarial review.

**Convergence detection**: Stops early when models agree. Continues if they don't. Disagreement is a valid output.

## Installation

```bash
/plugin install github:bacoco/Council-board-skill
```

Requires: `claude`, `gemini`, `codex` CLIs installed and authenticated.

## Sample Output

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

## The 3 Personas

Each model plays a specialized role:

| Model | Role | Focus |
|-------|------|-------|
| Claude | Chief Architect | Strategy, trade-offs, maintainability |
| Gemini | Security Officer | Vulnerabilities, compliance, risk |
| Codex | Performance Engineer | Speed, algorithms, scalability |

In debate mode, roles change to: Advocate FOR, Advocate AGAINST, Neutral Analyst.

## Resilience

Council keeps working even when models fail:
- **2 models available**: Continues with 10% confidence penalty
- **Model keeps failing**: Automatically excluded (circuit breaker)
- **Slow model**: Timeout adapts based on history

## Privacy

Everything runs locally via your authenticated CLIs. No data sent to external services beyond the model APIs you already use.

## Documentation

- [skills/council/SKILL.md](skills/council/SKILL.md) - Full technical documentation
- [skills/council/references/](skills/council/references/) - Detailed guides

## License

MIT - [LICENSE](LICENSE)

# Council

**Stop trusting a single AI. Get answers that survive expert debate.**

Council orchestrates Claude, Gemini, and Codex to debate, challenge, and synthesize answers together. Three perspectives. One synthesis.

## Inspired By

> *"LLM Council works together to answer your hardest questions"*
> — [Andrej Karpathy](https://github.com/karpathy/llm-council)

This project extends Karpathy's [LLM Council](https://github.com/karpathy/llm-council) concept with:
- **Dynamic persona generation** — AI-generated expert roles tailored to each question
- **Multi-round deliberation** — Models see each other's arguments and provide rebuttals
- **Automatic escalation** — Consensus → Debate → Devil's Advocate based on convergence
- **Claude Code integration** — Works as a native skill, not a separate web app

## See It In Action

```
Council Deliberation: Microservices vs Monolith for Startups

Personas: The Velocity Optimizer, The Complexity Cartographer, The Scale Prophet
Rounds: 2 (converged)
Confidence: 0.89

Consensus:
For early-stage startups (<15 engineers), start with a modular monolith.
Extract services only when you have evidence of scaling bottlenecks.

Dissenting View:
The Scale Prophet notes that certain compliance requirements
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
<summary>Prerequisites - CLI Installation</summary>

**Claude CLI** (requires Claude Pro/Max subscription):
```bash
npm install -g @anthropic-ai/claude-code
claude auth login
```

**Gemini CLI**:
```bash
npm install -g @anthropic-ai/gemini-cli
gemini auth login
```

**Codex CLI**:
```bash
npm install -g @openai/codex
codex auth
```

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
<summary>Dynamic Personas — Council Self-Analysis</summary>

Council **automatically generates personas tailored to your question**.

Example personas generated for "How to secure a payment API?":
- *The Cryptography Sentinel* — encryption and key management
- *The Compliance Navigator* — PCI-DSS and regulatory requirements
- *The Attack Surface Cartographer* — threat modeling and vulnerabilities

### We Asked the Council About Itself

**Question**: *"Are creative persona names like 'The Cryptography Sentinel' or 'The Latency Hunter' relevant for generating out-of-the-box thinking? Or is it just superficial marketing?"*

**Council Verdict** (Convergence: 0.834):

> Creative persona names are **neither superficial marketing nor cognitive magic — they're an effective prompt configuration tool**.

**Three levels of real impact:**

1. **Cognitive framing (real but moderate)** — "Hunter" invokes tracking, prey, patience — metaphors that color reasoning differently than "Engineer". The effect operates on style and focus, not fundamental logical structure.

2. **Multi-agent differentiation (crucial)** — Generic names ("Expert 1", "Analyst 2") create gravitational pull toward convergent outputs. Distinctive names maintain separation of reasoning threads and **expand the perceived search space**.

3. **Human reception (measurable)** — Creative labels increase perceived confidence, engagement, and willingness to consider divergent views — even when underlying content is equivalent.

**Dissent (Codex)**: *Without rigorous comparative data, attributing effects to creative names remains post-hoc rationalization. The "cognitive priming" analogy is potentially anthropomorphic.*

**Nuanced conclusion**: Marginal utility of creative names decreases sharply when behavioral specifications are already robust. Their power is maximal in under-specified contexts where the model must fill inferential gaps.

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

---

*Inspired by [Andrej Karpathy's LLM Council](https://github.com/karpathy/llm-council). Research support: [MIT "Debating LLMs" study (2024)](https://venturebeat.com/ai/a-weekend-vibe-code-hack-by-andrej-karpathy-quietly-sketches-the-missing) found that models produce more accurate results when prompted to critique each other's outputs.*

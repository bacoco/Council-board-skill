# Council

**Stop trusting a single AI. Get answers that survive expert debate.**

Council orchestrates Claude, Gemini, and Codex to debate, challenge, and synthesize answers together. Three perspectives. One synthesis.

---

## Quick Start

```bash
# Install the plugin
claude plugin marketplace add bacoco/Council-board-skill
claude plugin install council@council-board

# Verify setup
python3 skills/council/scripts/council.py --check
```

Then just ask:
```
"Ask the council: Should we use microservices?"
```

## Example Phrases

| What you say | What happens |
|--------------|--------------|
| "Ask the council: [question]" | 3 models collaborate to answer |
| "Debate this: [topic]" | Models argue FOR and AGAINST |
| "Challenge my design: [proposal]" | Red Team attacks, Blue Team defends |
| "Peer review this code" | Security, architecture, performance review |

## Modes

| Mode | Use For |
|------|---------|
| `consensus` | Technical questions, design decisions |
| `debate` | Controversial topics, binary choices |
| `devil_advocate` | Stress-testing, security reviews |
| `vote` | Multiple choice decisions |

## Why This Works

- **Multi-round deliberation** — Models see each other's arguments and provide rebuttals
- **Convergence detection** — Stops early when models agree, continues when they don't
- **Confidence scoring** — You see how confident the synthesis is

## No API Keys Needed

Uses your existing `claude`, `gemini`, and `codex` CLI subscriptions.

<details>
<summary>CLI Installation</summary>

```bash
# Claude CLI
npm install -g @anthropic-ai/claude-code && claude auth login

# Gemini CLI
npm install -g @anthropic-ai/gemini-cli && gemini auth login

# Codex CLI
npm install -g @openai/codex && codex auth
```
</details>

## Sample Output

```
Council Deliberation: Microservices vs Monolith for Startups

Rounds: 2 (converged)
Confidence: 0.89

Consensus:
For early-stage startups (<15 engineers), start with a modular monolith.
Extract services only when you have evidence of scaling bottlenecks.

Dissenting View:
Certain compliance requirements (HIPAA, PCI) may justify early service boundaries.
```

## Documentation

**[skills/council/SKILL.md](skills/council/SKILL.md)** — Full usage documentation

Reference guides in `skills/council/references/`:
- `modes.md` — Deliberation modes
- `security.md` — Input validation
- `resilience.md` — Graceful degradation

## Recent Improvements

- [x] Trail IO error handling for constrained filesystems
- [x] Thread-safe global state access
- [x] Subprocess cleanup helper
- [x] SOTA skill best practices (progressive disclosure)

## License

MIT - [LICENSE](LICENSE)

---

*Inspired by [Andrej Karpathy's LLM Council](https://github.com/karpathy/llm-council).*

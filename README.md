# Council

**Stop trusting a single AI. Get answers that survive expert debate.**

Council orchestrates Claude, Gemini, and Codex to debate, challenge, and synthesize answers together. Three perspectives. One synthesis.

---

### 🎯 Collective Intelligence, Amplified.

> You code alone. You decide alone. You doubt alone.
> **What if three AI minds deliberated before you commit?**

**Three models. Dynamic personas. One clear answer.**

**For Developers:**
- Pressure-test architecture, APIs, and stack decisions in minutes
- Devil's Advocate mode exposes blind spots before your PR does
- Anonymous peer review: candid critique, ego intact

**For Decision Makers:**
- Consensus detected = instant green light
- Full audit trail: every decision justified and shareable
- 2-hour debates become 10-minute decisions

*Debate mode stress-tests. Consensus mode synthesizes.*
*Personas adapt dynamically to your specific query—no scripting, no setup.*

**Doubt drags. Collective clarity ships.**

```bash
/council "Microservices or monolith?"  # → Clear verdict in ~2 min
```

---

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

# Verify setup
python3 skills/council/scripts/council.py --check
```

Then just ask:
```
"Ask the council: Should we use microservices?"
```

## Advanced Usage

### Deliberation Trail (`--trail`)

**Enabled by default.** Shows who said what at each round — full transparency into the deliberation process.

```bash
python3 skills/council/scripts/council.py --query "Your question" --trail
```

Output includes:
```json
{
  "deliberation_trail": [
    {
      "round": 1,
      "model": "claude",
      "persona": "The Type Guardian",
      "persona_role": "Evaluates type safety and maintainability",
      "answer": "TypeScript is recommended because...",
      "key_points": ["Type safety", "IDE support", "Refactoring"],
      "confidence": 0.92,
      "latency_ms": 23489
    }
  ],
  "trail_metadata": {
    "total_rounds": 2,
    "participants": 3,
    "total_contributions": 6,
    "consensus_reached": true
  }
}
```

Disable with `--no-trail` or set `enable_trail: false` in config.

### Performance Metrics (`--enable-perf-metrics`)

Disabled by default. Emits latency data per stage.

```bash
python3 skills/council/scripts/council.py --enable-perf-metrics --query "Your question"
```

### Configuration

All flags can be set persistently in [`skills/council/council.config.yaml`](skills/council/council.config.yaml):

```yaml
enable_trail: true          # Show deliberation trail (default: true)
enable_perf_metrics: false  # Show performance metrics (default: false)
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

## Full Transcript Example

Real deliberation on a coding question with actual timings:

<details>
<summary>Question: "Python async scraper hitting rate limits — backoff, semaphore, or queue?"</summary>

```
╔══════════════════════════════════════════════════════════════╗
║              COUNCIL SETUP VALIDATION                        ║
╠══════════════════════════════════════════════════════════════╣
║  ✓ claude   │ 2.0.76 (Claude Code)                          ║
║  ✓ gemini   │ 0.16.0                                        ║
║  ✓ codex    │ codex-cli 0.72.0                              ║
╠══════════════════════════════════════════════════════════════╣
║  STATUS: All CLIs ready ✓                                    ║
╚══════════════════════════════════════════════════════════════╝

Session: council-1767103092
Mode: consensus | Max rounds: 2

ROUND 1 - Initial Positions
├─ Personas: The Backpressure Alchemist, Guardian of the Semaphore Gates, The Queue Architect
├─ Claude (28.6s) ✓
├─ Gemini (38.1s) ✓
└─ Codex (10.0s) ✓

ROUND 2 - Rebuttals
├─ Personas: The Throttle Alchemist, Oracle of the Semaphore Gates, The Queue Architect
├─ Claude (31.8s) ✓
├─ Gemini (38.0s) ✓
└─ Codex (11.1s) ✓
└─ Convergence: 0.926 ✓

PEER REVIEW
├─ The Throttle Alchemist: 16/20
├─ Oracle of the Semaphore Gates: 18/20
└─ The Queue Architect: 20/20

SYNTHESIS (31.8s)
├─ Confidence: 0.91
└─ Contradictions resolved: 3

TOTAL: 168.6s (~2.8 min)

═══════════════════════════════════════════════════════════════

VERDICT: Use all three in a layered architecture.

Your question presents a false trichotomy — at 10,000 URLs/hour
(~2.8 req/sec), you need a hybrid approach:

1. Queue-Based Foundation (asyncio.Queue)
2. Per-Host Semaphores (NOT global!)
3. Token Bucket Rate Limiting (~2.5 req/sec)
4. Exponential Backoff with Jitter
5. Adaptive Tuning

Implementation Priority:
Queue → Per-host Semaphores → Token Bucket → Backoff → Adaptive

DISSENT: Participant A argued semaphores should be primary,
queue adds unnecessary complexity. Overruled: at 10k URLs/hour,
queues provide essential operational benefits.
```

</details>

## Documentation

**→ [skills/council/SKILL.md](skills/council/SKILL.md)** — Full technical documentation

**→ [skills/council/council.config.yaml](skills/council/council.config.yaml)** — Configuration template

Reference guides in `skills/council/references/`:
- `modes.md` — Deliberation modes (consensus, debate, devil_advocate)
- `failure-modes.md` — Error handling, timeouts, recovery
- `resilience.md` — Graceful degradation, circuit breaker
- `security.md` — Input validation, secret redaction
- `examples.md` — Usage examples

## Roadmap

Future improvements identified by Council self-evaluation:

- [ ] **Core Logic Tests** — Unit tests for deliberation engine, convergence algorithm, persona generation
- [ ] **Persistent State** — Save circuit breaker state, metrics, adaptive timeouts across sessions (JSON/SQLite)
- [ ] **Direct API Providers** — Bypass CLI fragility with native API implementations
- [ ] **CI/CD Pipeline** — GitHub Actions, automated testing, version compatibility matrix
- [ ] **Benchmarks** — Compare output quality/cost/latency vs single-model baselines
- [ ] **Health Checks** — Endpoints for container orchestration, graceful SIGTERM handling

## License

MIT - [LICENSE](LICENSE)

---

*Inspired by [Andrej Karpathy's LLM Council](https://github.com/karpathy/llm-council). Research support: [MIT "Debating LLMs" study (2024)](https://venturebeat.com/ai/a-weekend-vibe-code-hack-by-andrej-karpathy-quietly-sketches-the-missing) found that models produce more accurate results when prompted to critique each other's outputs.*

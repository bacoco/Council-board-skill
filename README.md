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
- Complex debates become structured, documented decisions

*Debate mode stress-tests. Consensus mode synthesizes.*
*Personas adapt dynamically to your specific query—no scripting, no setup.*

**Doubt drags. Collective clarity ships.**

```bash
/council "Should we raise funding or stay bootstrapped?"
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
Council Deliberation: Should we raise funding or stay bootstrapped?

Personas: The Growth Strategist, The Financial Realist, The Founder's Advocate
Rounds: 2 (converged)
Confidence: 0.87

Consensus:
Stay bootstrapped if you can reach profitability within 12 months.
Raise only if your market has clear winner-take-all dynamics
or requires heavy upfront investment (hardware, regulatory, etc.).

Dissenting View:
The Growth Strategist notes that even profitable bootstrapped
companies may want strategic investors for network effects and
credibility in enterprise sales.
```

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

## Features

### 🎯 Deliberation Modes — When to Use What

**Not sure which mode to pick?** Here's a simple guide:

---

#### 🤝 **Consensus** (Default)
> *"I need a solid answer that experts would agree on."*

**How it works:** All three AIs discuss, see each other's arguments, and work toward agreement. Like a team meeting where everyone shares their view, responds to others, and reaches a shared conclusion.

**Use it when:**
- You want the "right" answer, not just opinions
- Multiple perspectives should be synthesized into one recommendation
- You need confidence before acting

**Examples:**
```
"Should I hire a generalist or a specialist for my first employee?"
"What's the best pricing strategy for a new SaaS product?"
"How should a small team handle customer support at scale?"
```

**What you get:** A clear recommendation with reasoning, plus dissenting views if the AIs disagreed.

---

#### ⚔️ **Debate**
> *"I want to hear the strongest case for both sides."*

**How it works:** One AI argues FOR, another argues AGAINST, the third synthesizes. Like watching a structured debate where each side makes their best case.

**Use it when:**
- You're genuinely torn between two paths
- You want to understand trade-offs before committing
- There's no obvious "right answer"

**Examples:**
```
"I'm torn between raising funding and staying bootstrapped"
"Show me both sides: launch fast with bugs vs slow with polish"
"What are the pros and cons of mandatory video calls for remote teams?"
```

**What you get:** Balanced arguments for both sides, with a summary of which points were strongest.

---

#### 😈 **Devil's Advocate**
> *"Attack my idea. Find every flaw before someone else does."*

**How it works:** Red Team tries to break your idea, Blue Team defends it, Purple Team synthesizes improvements. Like stress-testing a plan before you commit.

**Use it when:**
- You're about to make a big bet
- You want to find blind spots in your thinking
- You need to anticipate objections

**Examples:**
```
"What could go wrong with our plan to launch in 3 markets at once?"
"Tear apart our strategy to undercut competitors on price"
"Find the blind spots in my plan to quit and freelance"
```

**What you get:** Weaknesses identified, counter-arguments surfaced, and concrete suggestions to strengthen your approach.

---

#### 🗳️ **Vote**
> *"I have options. Help me pick."*

**How it works:** Each AI votes for their preferred option and explains why. You see the tally and all reasoning.

**Use it when:**
- You have 3+ distinct choices
- You want a quick decision with justification
- The options are clear, you just need a push

**Examples:**
```
"Help me choose: B2B, B2C, or both?"
"Which is best for our rebrand — minimal, bold, or playful?"
"Help me pick: expand to Europe, Asia, or Latin America first?"
```

**What you get:** Vote tally, winner recommendation, and each AI's reasoning.

---

**⭐ Default:** If you just say *"Ask the council: [question]"*, it uses **Consensus** mode automatically.

### 🌩️ STORM Modes — Structured Deep Analysis

Regular modes give you discussion. **STORM modes** give you structured workflows with evidence tracking — like having a research team follow a methodology.

> Think of it like this: Regular modes are a conversation. STORM modes are a process.

---

#### 📊 **storm_decision** — Make a tough choice
> *"I need to pick between options, but I want a rigorous process."*

**What happens:**
1. Lists all your options clearly
2. Scores each option against criteria that matter
3. Red-teams the leading choice (finds what could go wrong)
4. Gives a final recommendation with warning signs to watch for

**When to use:** Big decisions where you need to justify your choice to others (investors, team, yourself).

**Example:**
```
"I need a rigorous comparison: build billing ourselves, use Stripe, or white-label?"
"Help me formally evaluate: partner with A, partner with B, or go alone?"
```

---

#### 🔬 **storm_research** — Understand something deeply
> *"I need to really understand this topic, not just get a quick answer."*

**What happens:**
1. Generates different perspectives on the topic
2. Creates questions each perspective would ask
3. Drafts an explanation
4. Critiques and improves the draft
5. Produces a final comprehensive report

**When to use:** Learning something new, preparing for a presentation, writing documentation.

**Example:**
```
"I need to deeply understand how recommendation algorithms work"
"Give me a comprehensive analysis of the carbon credit market"
```

---

#### 🔍 **storm_review** — Audit something thoroughly
> *"Review this and tell me what's wrong with it."*

**What happens:**
1. Scans for obvious issues
2. Models potential threats/risks
3. Analyzes quality from multiple angles
4. Suggests specific fixes

**When to use:** Before launching, before signing, before committing to something.

**Example:**
```
"Thoroughly review this contract before I sign" (with --context-file)
"Audit our onboarding flow and check for issues"
```

---

**Why STORM over regular modes?**
- Regular modes: Great for questions and decisions
- STORM modes: Better for complex analysis that needs structure and evidence tracking

### 📋 Full Reasoning Trail

Every deliberation is automatically saved so you can see exactly how the AIs reached their conclusion:

- What each AI expert said in each round
- How they responded to each other's arguments
- The final peer review scores
- Why they agreed or disagreed

**Find your trails in:** `council_trails/` folder — just click the file path shown after each session.

### 🔧 Direct Mode — Ask One Model Directly

Skip the full Council deliberation and ask a single model (or a few) directly.

| What you say | What happens |
|--------------|--------------|
| *"What does Claude think about this?"* | Claude responds directly |
| *"Just Gemini's opinion please"* | Gemini responds directly |
| *"Quick answer from Codex"* | Codex responds directly |
| *"Skip the council, ask Claude"* | Claude only, no deliberation |
| *"Run this by Gemini real quick"* | Fast Gemini response |
| *"Get Claude and Gemini's take"* | Both respond (no synthesis) |
| *"All models, no debate"* | All 3 respond sequentially |

### 📊 Check if Models are Working

| What you say | What happens |
|--------------|--------------|
| *"Check if all models are working"* | Verifies all CLIs installed |
| *"Ping all models"* | Quick test, each says OK |
| *"Ask each model: who made you?"* | Identity check |
| *"What tools does Codex have?"* | Lists Codex capabilities |

<details>
<summary>📈 Check Your Remaining Quota</summary>

Models don't show quota via command line. Check your usage online:

| Model | Where to Check |
|-------|----------------|
| Claude | [console.anthropic.com/settings/usage](https://console.anthropic.com/settings/usage) |
| Gemini | [aistudio.google.com](https://aistudio.google.com) |
| Codex | [platform.openai.com/usage](https://platform.openai.com/usage) |

</details>

<details>
<summary>🔧 CLI Reference (for developers)</summary>

```bash
# Direct mode - single model
python3 skills/council/scripts/council.py --direct --models claude --query "..." --human

# Direct mode - multiple models
python3 skills/council/scripts/council.py --direct --models claude,gemini,codex --query "..." --human

# Check CLI installation
python3 skills/council/scripts/council.py --check
```

</details>

### ⚙️ Customization

You can adjust Council's behavior in the settings file `council.config.yaml`:

| Setting | What It Does | Default |
|---------|--------------|---------|
| `max_rounds` | How many rounds of debate | 3 |
| `timeout` | How long to wait for each AI | 7 minutes |
| `enable_trail` | Save the full discussion | Yes |

<details>
<summary>🔧 Prerequisites - Installing the AI Tools</summary>

You need to install the command-line tools for each AI:

**Claude** (requires Claude Pro or Max subscription):
```
npm install -g @anthropic-ai/claude-code
claude auth login
```

**Gemini**:
```
npm install -g @anthropic-ai/gemini-cli
gemini auth login
```

**Codex**:
```
npm install -g @openai/codex
codex auth
```

</details>

## Example Phrases

Just talk naturally. Council detects your intent.

| What you say | Mode triggered |
|--------------|----------------|
| "Ask the council: should we expand to Europe?" | **Consensus** — collaborative answer |
| "I'm torn between hiring now or waiting" | **Debate** — FOR vs AGAINST |
| "What could go wrong with our launch plan?" | **Devil's advocate** — stress-test |
| "Help me choose between these 3 vendors" | **Vote** — tally with reasoning |
| "I need to deeply understand how pricing works" | **STORM research** — structured deep-dive |
| "Just ask Claude what it thinks" | **Direct** — single model, no debate |

## Why This Beats Single-Model Answers

**Pressure-tested answers** — Models don't just answer once. They see each other's arguments and provide rebuttals. Like a real debate.

**Structured disagreement** — When models disagree, it automatically escalates to adversarial review. Disagreement is surfaced, not hidden.

**Knows when to stop** — Stops early when models agree. Continues when they don't. You see the confidence score.

## No API Keys Needed

Uses your existing `claude`, `gemini`, and `codex` CLI subscriptions. No separate API costs — just your regular CLI usage.

<details>
<summary>Dynamic Personas — How They Work</summary>

Council **automatically generates expert personas tailored to your question**.

Example personas generated for "Should we expand internationally?":
- *The Market Scout* — market size, competition, timing
- *The Operations Realist* — logistics, hiring, legal complexity
- *The Financial Strategist* — costs, currency risk, break-even timeline

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

Real deliberation showing how the council works:

<details>
<summary>Question: "Should we open-source our core product?"</summary>

```
╔══════════════════════════════════════════════════════════════╗
║              COUNCIL SETUP VALIDATION                        ║
╠══════════════════════════════════════════════════════════════╣
║  ✓ claude   │ Claude Code                                   ║
║  ✓ gemini   │ Gemini CLI                                    ║
║  ✓ codex    │ Codex CLI                                     ║
╠══════════════════════════════════════════════════════════════╣
║  STATUS: All CLIs ready ✓                                    ║
╚══════════════════════════════════════════════════════════════╝

Session: council-1767103092
Mode: debate | Max rounds: 2

ROUND 1 - Initial Positions
├─ Personas: The Community Builder, The Revenue Guardian, The Market Analyst
├─ Claude ✓
├─ Gemini ✓
└─ Codex ✓

ROUND 2 - Rebuttals
├─ Claude ✓
├─ Gemini ✓
└─ Codex ✓
└─ Convergence: 0.82 ✓

PEER REVIEW
├─ The Community Builder: 17/20
├─ The Revenue Guardian: 18/20
└─ The Market Analyst: 16/20

SYNTHESIS
├─ Confidence: 0.85
└─ Contradictions resolved: 2

═══════════════════════════════════════════════════════════════

VERDICT: Open-source strategically, not completely.

The council recommends an "open core" model:

FOR open-sourcing (Community Builder):
- Builds trust and adoption faster than marketing
- Community contributions improve the product
- Reduces customer acquisition cost

AGAINST full open-source (Revenue Guardian):
- Pure open-source has no direct monetization
- Competitors can fork without contributing back
- Support burden shifts to you without revenue

RECOMMENDATION:
Open-source the core engine, keep enterprise features proprietary.
Examples: GitLab, Elastic, HashiCorp model.

DISSENT: The Community Builder argues full open-source
builds deeper trust. Counter: trust doesn't pay salaries.
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

## Recent Improvements

- [x] **Real Evidence Retrieval** — Researcher agent now performs actual repo/doc search with term extraction
- [x] **Source Reliability Scoring** — Multi-factor scoring (source type, authority indicators, snippet quality)
- [x] **Key Term Extraction** — Intelligent extraction of CamelCase, snake_case, acronyms from claims
- [x] **STORM Pipeline** — Evidence-grounded workflow graphs (decision, research, code-review)
- [x] **KnowledgeBase** — Shared artifact tracking claims, sources, and decisions
- [x] **Evidence-aware Convergence** — Confidence factors in evidence coverage
- [x] **Moderator Agent** — Workflow detection, shallow consensus detection
- [x] **Trail Files** — STORM trails with KB snapshots, node results, evidence reports
- [x] **Direct Mode** — Query models directly with `--direct` flag, no deliberation
- [x] **Natural Language Triggers** — "Ask Claude directly", "Just Gemini's opinion", etc.
- [x] **Peer Review Score Fix** — Auto-detect 0-5 vs 0-20 scale, display correct totals
- [x] **Skill Cleanup** — Tests and trails moved outside skill package (72KB → 35 files)
- [x] **Timeout Flag Fixed** — `--timeout` now properly respected (was hardcoded to 420s)
- [x] **Session State Reset** — All global state properly cleared between sessions

## Development

Run tests from project root:

```bash
python3 -m pytest tests/ -v
```

## Roadmap

Future improvements identified by Council self-evaluation:

- [x] **Core Logic Tests** — Unit tests for deliberation engine, convergence algorithm, persona generation
- [x] **STORM Workflows** — Decision, Research, Code Review graphs with evidence tracking
- [x] **KnowledgeBase** — Shared artifact for claims, sources, and decisions
- [x] **Source Reliability** — Multi-factor reliability scoring for retrieved evidence
- [x] **Real Evidence Retrieval** — Repo/doc search with key term extraction
- [ ] **Cross-Model Verification** — Query multiple models to independently verify claims
- [ ] **Web Retrieval** — Web search integration for external evidence
- [ ] **Persistent State** — Save circuit breaker state, metrics, adaptive timeouts across sessions (JSON/SQLite)
- [ ] **CI/CD Pipeline** — GitHub Actions, automated testing, version compatibility matrix
- [ ] **Benchmarks** — Compare output quality/cost/latency vs single-model baselines

## License

MIT - [LICENSE](LICENSE)

---

*Inspired by [Andrej Karpathy's LLM Council](https://github.com/karpathy/llm-council). Research support: [MIT "Debating LLMs" study (2024)](https://venturebeat.com/ai/a-weekend-vibe-code-hack-by-andrej-karpathy-quietly-sketches-the-missing) found that models produce more accurate results when prompted to critique each other's outputs.*

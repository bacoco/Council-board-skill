# Council - Multi-Model Deliberation Skill

> Claude Code skill for orchestrating collective AI intelligence through structured debate and synthesis

Enable Claude to coordinate opinions from multiple AI models (Claude variants, Gemini, GPT) via parallel analysis, anonymous peer review, and synthesis.

## Installation

### 1. Clone this skill repository

```bash
git clone https://github.com/bacoco/Council-board-skill.git
cd Council-board-skill
```

### 2. Skill is ready to use

Claude Code will automatically detect the skill from `.claude/skills/council/SKILL.md`.

### 3. Test the skill

In Claude Code, simply ask:

```
Ask the council: Should we use TypeScript or JavaScript for this project?
```

Claude will automatically activate the council skill and coordinate multi-model deliberation.

## Usage

### Trigger Phrases

The skill activates when you use these keywords:
- "Ask the council..."
- "Debate this..."
- "Vote on..."
- "Peer review..."
- "Get multiple opinions..."
- "What do other AIs think..."
- "Challenge my design..."
- "Multi-model validation..."

### Examples

**Consensus Mode (Default)**
```
Ask the council: What's the time complexity of quicksort?
```

**Debate Mode**
```
Debate this: Monolithic vs microservices architecture for a startup
```

**Specialist Mode**
```
Get specialist opinions: Best database for real-time chat (focus on latency and scalability)
```

**Devil's Advocate**
```
Challenge my design: [paste your architecture diagram]
```

**Peer Review**
```
Peer review this code: [paste code]
```

## How It Works

The council skill uses **three orchestration strategies**:

### Strategy 1: Parallel Claude Agents (Fastest)

- Spawns 3-5 Claude agents in parallel using Task tool
- Each agent analyzes from different perspective
- Synthesizes findings with contradictions highlighted
- **Best for**: Quick validation, internal debate

### Strategy 2: External Model APIs (Most Diverse)

- Queries Gemini and GPT via their APIs
- Combines with Claude's native analysis
- Anonymous peer review prevents bias
- **Best for**: Complex decisions needing diverse perspectives

### Strategy 3: Python Backend (CLI Models)

- Uses included Python orchestrator for CLI-based models
- Structured 3-stage pipeline: gather → review → synthesize
- **Best for**: Security-critical deliberations, audit trails

## Deliberation Modes

| Mode | Trigger | Use Case | Example |
|------|---------|----------|---------|
| **consensus** | "Ask the council" | Factual questions | "What is OAuth 2.0?" |
| **debate** | "Debate this" | Controversial topics | "React vs Vue?" |
| **vote** | "Vote on" | Binary choices | "Should we migrate to Rust?" |
| **specialist** | "Get specialist opinions" | Domain expertise | "Debug this GPU kernel" |
| **devil_advocate** | "Challenge my design" | Stress-test ideas | Architecture reviews |

## Configuration

The skill automatically adapts based on:
- Available API keys (Gemini, OpenAI)
- Claude Code model selection
- Task complexity (fast/balanced/thorough)

### Optional: External Model APIs

To enable Gemini and GPT participation:

```bash
# Add to your shell profile
export GEMINI_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"
```

Without these, the skill uses multiple Claude agents instead (still effective!).

## Security Features

- **Secret Redaction**: Auto-masks API keys, tokens, passwords
- **Injection Detection**: Blocks prompt injection attempts
- **Anonymous Review**: Responses shuffled to prevent brand bias
- **XML Sandwich Prompts**: Clear DATA/INSTRUCTION separation

## Budget Control

The skill automatically selects strategy based on question complexity:

| Budget | Agents/Models | Review | Typical Cost |
|--------|---------------|--------|--------------|
| **Fast** | 2 Claude agents | Skip if aligned | ~$0.01 |
| **Balanced** | 3 models (Claude + external) | Standard | ~$0.03 |
| **Thorough** | 5 models + full verification | Complete | ~$0.10 |

## Advanced Features

### Multi-Round Debate

```
Debate this over 3 rounds: "Is unit testing worth the time investment?"
```

Agents argue positions, rebut each other, then converge or clarify disagreements.

### Specialist Routing

```
Route to specialists: "Optimize this SQL query for a 100M row table"
```

Automatically assigns to models with strongest domain expertise.

### Parallel Analysis

```
Get parallel analyses: "Security review of this authentication flow"
```

Multiple angles examined simultaneously: XSS, CSRF, token handling, session management.

## Architecture

```
User Request
    ↓
Skill Triggers (keyword match)
    ↓
┌─────────────────────────────────────┐
│ Council Orchestration               │
│                                     │
│  Strategy Selection:                │
│  ├─ Parallel Claude agents          │
│  ├─ External API queries            │
│  └─ Python backend                  │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│ Stage 1: Opinion Gathering          │
│ (Parallel execution)                │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│ Stage 2: Peer Review                │
│ (Anonymized scoring)                │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│ Stage 3: Synthesis                  │
│ (Resolve contradictions)            │
└────────────┬────────────────────────┘
             │
         Final Answer
```

## Reference Documentation

- [`.claude/skills/council/SKILL.md`](.claude/skills/council/SKILL.md) - Main skill definition
- [`references/modes.md`](references/modes.md) - Detailed mode explanations
- [`references/prompts.md`](references/prompts.md) - XML prompt templates
- [`references/security.md`](references/security.md) - OWASP LLM mitigations
- [`references/schemas.md`](references/schemas.md) - Response formats

## Development

See [`CLAUDE.md`](CLAUDE.md) for architecture details and development guidelines.

## Contributing

Contributions welcome! Focus areas:
- Additional deliberation modes
- MCP server integration for external models
- Enhanced peer review algorithms
- Cost optimization strategies

## License

MIT

## Inspiration

- Andrej Karpathy's multi-stage deliberation approach
- OWASP LLM Top 10 security guidelines
- OpenCode's parallel agent orchestration

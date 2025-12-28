# Council Board - Multi-Model Deliberation Plugin

Multi-model deliberation orchestrator for Claude Code. Coordinates opinions from Claude Opus, Gemini Pro, and Codex through parallel analysis, anonymous peer review, and synthesis.

## Installation

```bash
/plugin install github:bacoco/Council-board-skill
```

Then restart Claude Code to load the plugin.

## Usage

Simply ask the council directly in your conversation:

```
Ask the council: Should we use TypeScript or JavaScript for this project?
```

```
Debate this: Monolith vs microservices architecture for a startup
```

```
Get multiple opinions: Best database for real-time chat app
```

Claude will automatically:
1. ✓ Query Gemini Pro
2. ✓ Query Codex
3. ✓ Provide Claude Opus analysis
4. ✓ Conduct anonymous peer review
5. ✓ Synthesize consensus with confidence score

## Deliberation Modes

| Mode | Trigger | Use Case |
|------|---------|----------|
| **consensus** | "ask the council" | Factual questions, technical validation |
| **debate** | "debate this" | Controversial topics, multi-round arguments |
| **vote** | "vote on" | Binary/multiple choice decisions |
| **specialist** | "get specialist opinions" | Domain expert routing |
| **devil_advocate** | "challenge my design" | Systematic critique |

## Requirements

The following CLI tools must be installed:
- `claude` (Anthropic CLI)
- `gemini` (Google Gemini CLI)
- `codex` (OpenAI Codex CLI)

Each CLI handles its own OAuth authentication with paid subscriptions.

## How It Works

### Stage 1: Opinion Gathering (Parallel)
- Queries all 3 models simultaneously
- Each provides analysis with key points, confidence, uncertainties

### Stage 2: Peer Review (Anonymized)
- Responses shuffled and labeled A, B, C
- Scored on accuracy, completeness, reasoning, clarity (1-5 scale)
- Prevents brand bias

### Stage 3: Synthesis
- Chairman (Claude) resolves contradictions
- Produces final answer with confidence score
- Notes dissenting views if significant

## Example Output

```markdown
## Council Deliberation: Best database for real-time chat?

**Participants**: Claude Opus, Gemini Pro, Codex

✓ Gemini Pro responded (12.3s)
✓ Codex responded (3.1s)
✓ Claude Opus analysis complete

Conducting anonymous peer review...
Peer review complete. Synthesizing consensus...

### Council Consensus

The council recommends Redis for pub/sub messaging combined with PostgreSQL for message persistence.

**Confidence**: 0.91 (strong agreement)

**Key Points:**
- Redis excels at real-time pub/sub with sub-millisecond latency
- PostgreSQL provides reliable message history and search
- All three models agreed on this hybrid approach
- No significant contradictions detected

**Dissenting View:** None
```

## Documentation

- [skills/council/SKILL.md](skills/council/SKILL.md) - Full skill documentation
- [skills/council/references/](skills/council/references/) - Detailed guides

## License

MIT

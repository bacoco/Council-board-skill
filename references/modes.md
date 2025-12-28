# Deliberation Modes

## CONSENSUS (Default)

**Usage**: Factual questions, problems with single correct answer
**Process**: 3-stage Karpathy (opinions → peer review → synthesis)
**Example**: "What is the time complexity of quicksort?"

Fast-path: Skip peer review if Stage 1 shows strong agreement.

## DEBATE

**Usage**: Open questions, controversial topics
**Process**: Multi-round argumentation until convergence or max_rounds
**Example**: "Monolith vs microservices for a startup?"

Parameters:
- `max_rounds`: 3 (default)
- Convergence detected when positions align or confidence drops

## VOTE

**Usage**: Binary or multiple choice decisions
**Process**: Weighted voting with justifications
**Example**: "React or Vue for this project?"

Output includes:
- Vote tally
- Per-model justification
- Majority recommendation

## SPECIALIST

**Usage**: Technical tasks requiring domain expertise
**Process**: Route to expert model, others validate
**Example**: "Debug this CUDA kernel" → prioritize Codex

Routing heuristics:
- CUDA/GPU → Codex
- System design → Claude
- Math proofs → Gemini

## DEVIL_ADVOCATE

**Usage**: Validate ideas, stress-test designs
**Process**: One model systematically challenges
**Example**: "Critique my business plan"

Designated challenger identifies:
- Unstated assumptions
- Failure modes
- Edge cases
- Counter-examples

## Budget Modes

| Budget | Council Size | Peer Review | Typical Cost |
|--------|--------------|-------------|--------------|
| `fast` | 2 | Skip if consensus | ~$0.01-0.02 |
| `balanced` | 3 | Standard | ~$0.03-0.05 |
| `thorough` | 5 | Full + verification | ~$0.08-0.15 |

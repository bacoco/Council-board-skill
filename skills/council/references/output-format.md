# Council Output Format

## Standard Response Template

```markdown
## Council Deliberation: [Question]

**Participants**: Chief Architect (Claude), Security Officer (Gemini), Performance Engineer (Codex)
**Rounds Completed**: 2 of 3 (converged at round 2)
**Convergence Score**: 0.944 (converged)
**Session Duration**: 104.4s

### Round 1: Initial Positions

**Chief Architect** (Confidence: 0.85)
- [key architectural points]

**Security Officer** (Confidence: 0.90)
- [key security points]

**Performance Engineer** (Confidence: 0.80)
- [key performance points]

### Round 2: Rebuttals

**Chief Architect** (Confidence: 0.90)
- Rebuttals: [counter-arguments]
- Concessions: [points of agreement]

**Security Officer** (Confidence: 0.95)
- Rebuttals: [counter-arguments]
- Concessions: [points of agreement]

**Performance Engineer** (Confidence: 0.92)
- Rebuttals: [counter-arguments]
- Concessions: [points of agreement]

**Convergence**: Achieved (score: 0.944)

### Peer Review Scores

| Persona | Accuracy | Completeness | Reasoning | Clarity | Total |
|---------|----------|--------------|-----------|---------|-------|
| Chief Architect      | 5 | 5 | 5 | 5 | 20/20 |
| Security Officer     | 4 | 4 | 4 | 4 | 16/20 |
| Performance Engineer | 4 | 4 | 5 | 5 | 18/20 |

### Key Contradictions

- **Chief Architect** emphasizes X while **Security Officer** prioritizes Y
  - **Resolution**: [synthesis]

### Council Consensus

[Synthesized answer incorporating all perspectives]

**Final Confidence**: 0.91
**Dissenting View**: [minority perspective if significant]
```

## Degraded Mode Template

When operating with fewer than 3 models:

```markdown
## Council Deliberation: [Question]

**Participants**: Chief Architect (Claude), ~~Security Officer (Gemini)~~, Performance Engineer (Codex)
**Status**: DEGRADED (1 model unavailable)
**Confidence**: 0.81 (adjusted from 0.90)

> **Note**: Security Officer (Gemini) unavailable due to timeout.
> Results based on 2 of 3 perspectives.
```

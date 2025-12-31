# Deliberation Modes

## Contents
- Consensus (default)
- Debate
- Devil's Advocate
- Vote
- Adaptive

## CONSENSUS (Default)

**Use when**: Factual questions, technical validation, design decisions
**Process**: Multi-round with convergence detection
**Output**: Synthesized answer with confidence score

```bash
python3 ${SKILL_ROOT}/scripts/council.py --query "What's the time complexity of quicksort?" --mode consensus
```

### Round Flow

**Round 1 (Parallel)**:
- All 3 models provide initial analysis with dynamically-assigned personas
- Personas are generated based on the question type (technical, ethical, financial, etc.)

**Round 2+ (Rebuttals)**:
- Each model receives anonymized summaries of OTHER models' arguments
- Provides rebuttals to disagreements, concessions to agreements
- Signals convergence if consensus reached

**Convergence Detection**:
- Threshold: 0.8 (60% confidence weight + 40% explicit signal weight)
- If converged: Stop early, proceed to synthesis
- If not: Continue to max_rounds (default: 3)

## DEBATE

**Use when**: Controversial topics, binary decisions, competing approaches
**Personas**: Neutral Analyst, Advocate FOR, Advocate AGAINST
**Output**: Balanced analysis of both sides

```bash
python3 ${SKILL_ROOT}/scripts/council.py --query "Microservices vs monolith for startups" --mode debate
```

Parameters:
- `max_rounds`: 3 (default)
- Convergence detected when positions align or confidence drops
- Forces balanced consideration of both sides

## DEVIL_ADVOCATE

**Use when**: Stress-testing proposals, security reviews, finding edge cases
**Personas**: Purple Team (Integrator), Red Team (Attacker), Blue Team (Defender)
**Output**: Critique with identified weaknesses and mitigations

```bash
python3 ${SKILL_ROOT}/scripts/council.py --query "Proposal: Implement E2EE using AES-256" --mode devil_advocate
```

Designated challenger identifies:
- Unstated assumptions
- Failure modes
- Edge cases
- Counter-examples
- Security vulnerabilities

## VOTE

**Use when**: Binary or multiple choice decisions
**Process**: Weighted voting with justifications
**Output**: Vote tally + majority recommendation

```bash
python3 ${SKILL_ROOT}/scripts/council.py --query "React or Vue for this project?" --mode vote
```

Output includes:
- Vote counts per option
- Weighted scores (weight × confidence)
- Per-model justification
- Tie-breaking cascade: weighted score → raw count → highest confidence → alphabetical

## ADAPTIVE

**Use when**: Uncertain about question complexity
**Process**: Auto-escalates through modes based on convergence
**Output**: Final synthesis after appropriate mode completes

```bash
python3 ${SKILL_ROOT}/scripts/council.py --query "Should we rewrite in Rust?" --mode adaptive
```

Escalation path:
1. **consensus** → Check convergence
2. If low convergence → **debate**
3. If still unresolved → **devil_advocate**
4. Final meta-synthesis combining all phases

## Quorum Requirements

All modes require minimum 2 valid responses per round. If quorum fails:
- Round 1 failure: Session aborts with error
- Later rounds: Use previous round data, warn user

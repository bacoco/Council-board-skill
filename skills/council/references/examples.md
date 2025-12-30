# Council Examples

## Example 1: Technical Question (Consensus)

```bash
python3 ${SKILL_ROOT}/scripts/council.py \
  --query "What's the best database for real-time chat?" \
  --mode consensus --max-rounds 3
```

**Progress**:
1. "Round 1 started (max 3 rounds)"
2. "Chief Architect responded (16.2s)" - Architecture trade-offs
3. "Security Officer responded (12.3s)" - Security implications
4. "Performance Engineer responded (3.1s)" - Performance characteristics
5. "Round 2 started"
6. "Convergence check: score 0.91 (converged)"
7. **Synthesis**: "Redis for pub/sub + PostgreSQL for persistence..."

## Example 2: Debate Mode

```bash
python3 ${SKILL_ROOT}/scripts/council.py \
  --query "Microservices vs monolithic architecture for startups" \
  --mode debate --max-rounds 2
```

**Progress**:
1. "Neutral Analyst responded" - Analyzes both sides
2. "Advocate FOR responded" - Case for microservices
3. "Advocate AGAINST responded" - Case for monolith
4. Round 2: Rebuttals
5. **Synthesis**: "Context-dependent. For <15 engineers, modular monolith. Extract services based on evidence."
6. **Dissent**: "Strong disagreement with 'microservices are unequivocally better'"

## Example 3: Code Review with Context File

```bash
# Step 1: Create manifest
cat > /tmp/council_manifest.md << 'EOF'
# Council Context

## Question
Review auth module for security issues

## Files to Analyze

### src/auth.py
- Main authentication logic

### src/config.py
- JWT configuration
EOF

# Step 2: Call council
python3 ${SKILL_ROOT}/scripts/council.py \
  --query "Review this authentication module" \
  --context-file /tmp/council_manifest.md \
  --mode consensus
```

## Example 4: Devil's Advocate (Security Proposal)

```bash
python3 ${SKILL_ROOT}/scripts/council.py \
  --query "Proposal: Implement E2EE using AES-256" \
  --mode devil_advocate --max-rounds 2
```

**Progress**:
1. Purple Team (Integrator) - Initial synthesis
2. Red Team (Attacker) - Key management gaps, endpoint security, metadata exposure
3. Blue Team (Defender) - Justifies AES-256 choice
4. Round 2: Deeper critiques and mitigations
5. **Synthesis**: "CONDITIONAL APPROVAL: Must specify protocol (Signal/MLS), hardware-backed keys, key recovery, metadata protection"
6. **Dissent (Red Team)**: "Complexity is itself a vulnerability. Simpler server-side encryption may be safer than poorly implemented E2EE"

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Council is a multi-model deliberation orchestrator that coordinates opinions from Claude, Gemini, and Codex through parallel analysis, anonymous peer review, and synthesis. It's packaged as a Claude Code skill for seamless integration into the Claude Code IDE.

## Architecture

### Core Components

**Main Orchestrator** (`skills/council/scripts/council.py` - 3732 lines)
- Entry point for all deliberations
- Handles CLI argument parsing (`--query`, `--mode`, `--models`, `--context-file`, etc.)
- Manages the 3-stage deliberation pipeline: Gather Opinions → Peer Review → Synthesis
- Implements convergence detection (blends confidence scores + explicit signals)
- Supports multiple modes: `adaptive`, `consensus`, `debate`, `vote`, `devil_advocate`
- JSON output with optional Markdown trail files for transparency

**Model Provider Abstraction** (`skills/council/providers.py`)
- `ModelProvider`: Abstract base class for all model sources
- `CLIProvider`: Implementation for CLI-based models (claude, gemini, codex)
- `ProviderResponse`: Standardized response wrapper with latency, success, error, metadata
- `CouncilConfig`: YAML-based configuration loader for persistent settings

**Persona Manager** (`skills/council/persona_manager.py`)
- `PersonaManager`: Intelligent persona assignment based on question type
- `Persona`: Data class for persona definition (title, role, prompt_prefix, specializations)
- `PERSONA_LIBRARY`: 20+ pre-built personas (Historian, Scientist, Ethicist, Economist, etc.)
- Maps question types (TECHNICAL, ETHICAL, FINANCIAL, SCIENTIFIC, etc.) to specialist sets
- Falls back to cached personas if LLM persona generation fails

**Input Security** (`skills/council/security/input_validator.py`)
- `InputValidator`: Defense-in-depth validation against shell injection, prompt injection, secrets leakage
- `ValidationResult`: Detailed violation reporting with sanitized output
- Pattern-based detection for CWE-78 (shell injection), OWASP LLM01 (prompt injection), OWASP LLM06 (secrets)
- Query limit: 50K chars; Context limit: 200K chars; DoS protection via input length validation

### Deliberation Flow

1. **Opinion Gathering** (parallel, all models)
   - Models receive anonymized persona instructions
   - Each provides initial analysis with confidence score and convergence signal
   - Run in parallel with adaptive timeout (increases if model is slow)

2. **Rebuttal/Rounds** (rounds 2+)
   - Each model receives anonymized summaries of OTHER models' arguments
   - Provides rebuttals to disagreements, concessions to agreements
   - Convergence detection: 60% average confidence + 40% explicit signals = 0.8 threshold

3. **Peer Review** (after convergence or max_rounds)
   - "Chairman" model (default: claude) scores responses 0-20 on accuracy, completeness, reasoning, clarity

4. **Synthesis** (final)
   - Chairman produces unified answer incorporating all rounds
   - Includes dissenting views if models disagreed
   - Output: `{"answer": "...", "confidence": 0.92, "trail_file": "..."}`

### Deliberation Modes

- **Consensus** (default): Multi-round with convergence detection for factual/technical questions
- **Debate**: FOR/AGAINST personas for binary decisions, balanced argument presentation
- **Devil's Advocate**: Red Team (attack), Blue Team (defend), Purple Team (integrate) for stress-testing
- **Vote**: Tally votes, recommend majority view
- **Adaptive**: Auto-escalate based on convergence (consensus → debate → devil_advocate)

### Configuration

**Primary Config** (`skills/council/council.config.yaml`)
- `providers`: List of models to coordinate (claude, gemini, codex)
- `chairman`: Synthesis model (usually claude)
- `timeout`: Per-model timeout in seconds (default: 60, set to 420 for Codex with tools)
- `max_rounds`: Max deliberation rounds before forcing synthesis (default: 3)
- `convergence_threshold`: 0.0-1.0, default 0.8 (stricter = fewer early exits)
- `min_quorum`: Minimum valid responses required (default: 2)
- `enable_trail`: Save full deliberation to Markdown (default: true)
- `enable_perf_metrics`: Show per-stage latency breakdown (default: false)

## Common Development Tasks

### Running the Orchestrator

```bash
# Basic query
python3 skills/council/scripts/council.py --query "Microservices vs monolith?"
# With code review context
python3 skills/council/scripts/council.py \
  --query "Review this auth module" \
  --context-file /tmp/manifest.md \
  --mode devil_advocate

# PRD: Council STORM/Co-STORM-Inspired Upgrade

## 1) Context & Goals
- **Today:** Council orchestrates Claude, Gemini, and Codex through multi-round deliberation with personas, convergence detection, escalation, and audit trails.【F:README.md†L85-L110】【F:skills/council/references/modes.md†L10-L35】
- **Gap:** Debate alone can be style vs. style. Missing: proactive moderation, default retrieval + evidence scoring, shared knowledge artifact, and workflow structures proven in STORM/Co-STORM.
- **Goal:** Ship a disciplined, evidence-grounded orchestrator inside the existing `council` skill—no new skill packaging. Outputs should be defensible decisions with traceable evidence and residual risk.

## 2) Success Metrics
- **Quality:** Final confidence explicitly references evidence coverage and unresolved objections in ≥80% of runs (sampled logs).
- **Grounding:** ≥70% of final claims have at least one cited evidence source when retrieval is available.
- **Efficiency:** Moderator detects shallow consensus and triggers retrieval/verification in <2s overhead per round (measured internally).
- **UX:** CLI flags and defaults remain compatible; trail files include KB snapshots.

## 3) Scope
- **In-scope:** New agents (Moderator, Researcher, Evidence Judge), shared KnowledgeBase, workflow graphs (decision, research, code-review), evidence-aware confidence, trail updates, CLI defaults that keep current UX.
- **Out-of-scope (phase 1):** GUI/mind-map rendering, multi-session persistence store, new model backends, quota tracking.

## 4) Users & Jobs
- **Developers/Architects:** Want decisions with tradeoffs and tripwires to revisit.
- **Security/Reviewers:** Want red-team depth tied to evidence.
- **PM/Leads:** Want explainable confidence and decision records.

## 5) Functional Requirements
1) **Moderator-led loop**
   - Chooses workflow graph per query (decision/research/code-review).
   - Maintains `open_questions` queue; detects shallow consensus (agreement without evidence) and routes to retrieval/verification.
   - Emits next-step directives (finalize vs. retrieve vs. structured debate).
2) **Grounded-by-default**
   - Researcher retrieves sources for disputed claims or open questions (web/docs/repo).
   - Evidence Judge produces Claim→Evidence→Confidence table, flags unsupported/contradicted claims.
   - Final answer must indicate which claims are unsupported or contradictory.
3) **Shared KnowledgeBase (KB)**
   - Contains `concepts`, `claims` (with provenance), `sources` (relevance, reliability), `open_questions`, `decisions` (rationale, tradeoffs, revisit triggers).
   - Every agent turn must add/update KB (new claim/evidence/question or clarification).
   - KB snapshot is stored in the trail file.
4) **Workflow graphs**
   - **Decision Graph:** generate options → rubric scoring → red-team risks → evidence check → recommendation + tripwires.
   - **Research Graph:** perspectives → question plan → retrieve per question → outline → draft → critique → final report.
   - **Code Review Graph:** static scan prompts → threat model → perf/maintainability → patch suggestions → final checklist.
5) **Evidence-aware convergence**
   - Confidence blends: agreement (votes), evidence coverage, unresolved objections, source diversity.
   - Output: “Confidence X because Y claims lack sources / Z objections remain” plus KB pointers.
6) **Compatibility**
   - CLI remains `python3 skills/council/scripts/council.py ...`; defaults unchanged.
   - Degraded mode (1–2 models) still works with confidence penalty; chairman failover unchanged.

## 6) Non-Functional Requirements
- **Latency:** Minimal overhead (<2s/round moderator decision, <5s retrieval trigger setup excluding external fetch).
- **Resilience:** Circuit breaker + retries unchanged; new roles respect existing timeout budget.
- **Security:** File retrieval scoped to manifests; web retrieval uses allowlisted methods (no arbitrary shell).
- **Observability:** Trails include KB snapshots and moderator routing decisions.

## 7) Architecture & Components
- **New modules (under `skills/council/`):**
  - `agents/moderator.py`: selects workflow, maintains open questions, detects shallow consensus.
  - `agents/researcher.py`: orchestrates retrieval tools; clusters evidence by claim.
  - `agents/evidence_judge.py`: scores support/contradiction; emits claim table.
  - `knowledge_base.py`: dataclass for KB sections; merge/update helpers; serialization for trails.
  - `workflows/decision_graph.py`, `workflows/research_graph.py`, `workflows/code_review_graph.py`: node definitions and execution helpers.
  - `convergence.py` (new or extended): evidence-aware confidence calculation.
- **Touched modules:**
  - `scripts/council.py`: expose new mode flag(s) if needed (but keep defaults), wire Moderator entry.
  - `modes/consensus.py` (and other modes): integrate Moderator routing and KB updates per turn.
  - `core/prompts.py`: add prompt templates for Moderator/Researcher/Evidence Judge and KB contracts.
  - `core/trail.py`: persist KB snapshots and routing events.

## 8) Flow (Decision Graph example)
1) Moderator inspects query + prior KB, picks Decision Graph.
2) Panelists (Claude/Gemini/Codex personas) generate options and initial claims; KB updated.
3) Moderator checks evidence coverage → triggers Researcher to fetch sources per disputed claim.
4) Evidence Judge scores claims; updates KB claim table with support/contradiction.
5) Panelists red-team top options; KB records risks and mitigations.
6) Moderator checks convergence; if shallow, another retrieval/verification loop; else finalize.
7) Chairman synthesizes decision record (rationale, tradeoffs, tripwires) with confidence rationale.

## 9) Prompts & Contracts (high level)
- **Moderator prompt:** “Given KB + round outputs, choose next step: finalize / retrieve / debate. Maintain open_questions. Flag shallow consensus.”
- **Researcher prompt:** “Retrieve sources for these claims/questions. Return citations with snippets, timestamps, reliability.”
- **Evidence Judge prompt:** “Produce Claim→Evidence→Confidence table; mark unsupported/contradicted; note source diversity.”
- **Panel prompt addition:** “You must add or update KB: new claim, evidence link, or question; tag speculative vs. supported.”

## 10) Data Model (KB)
- `Concept`: id, title, definition, related_concepts[]
- `Claim`: id, text, owner, status (proposed/supported/contradicted/speculative), support_evidence_ids[], contradict_evidence_ids[], confidence
- `Source`: id, uri/path, snippet, retrieved_at, reliability, relevance
- `OpenQuestion`: id, prompt, owner, status (open/resolved/deferred), linked_claim_ids[]
- `Decision`: id, summary, tradeoffs, tripwires (conditions to revisit), confidence, supporting_claim_ids[]

## 11) Telemetry & Trails
- Trail file gains: KB snapshot per turn, moderator routing decisions, evidence coverage stats, final Claim→Evidence table.
- Perf metrics: moderator latency, retrieval latency, evidence-judge latency, coverage ratios.

## 12) Risks & Mitigations
- **Latency creep:** Keep prompts tight; reuse existing timeouts; allow “quick mode” skip for retrieval when offline.
- **Over-assertion:** Evidence Judge enforces unsupported/contradicted tagging; Moderator blocks finalize on empty coverage.
- **Tool failures:** Reuse circuit breaker; degraded mode still allowed but confidence penalized.

## 13) Rollout Plan

### Phase 1 (MVP) — ✅ IMPLEMENTED
- ✅ **Pipeline Abstraction** — Classic vs STORM dual-mode system (`pipelines/base.py`, `classic.py`, `storm.py`)
- ✅ **KnowledgeBase** — Full implementation with Claims, Sources, Decisions, OpenQuestions (`knowledge_base.py`)
- ✅ **Moderator Agent** — Workflow detection, shallow consensus detection, routing (`agents/moderator.py`)
- ✅ **Researcher Agent** — Full implementation with repo/doc search (`agents/researcher.py`)
- ✅ **Evidence Judge** — Claim evaluation, confidence adjustment (`agents/evidence_judge.py`)
- ✅ **Convergence Detector** — Classic + evidence-aware modes (`convergence.py`)
- ✅ **Decision Graph** — 5 nodes with model queries: options → rubric → red-team → evidence → recommendation (`workflows/decision.py`)
- ✅ **Trail Files** — STORM-specific trail generation with KB snapshots (`core/storm_trail.py`)
- ✅ **Unit Tests** — Comprehensive test coverage (`tests/test_storm_components.py`)

### Phase 2 — ✅ IMPLEMENTED
- ✅ **Research Graph** — 7 nodes: perspectives → questions → retrieve → outline → draft → critique → report (`workflows/research.py`)
- ✅ **Code Review Graph** — 5 nodes: static scan → threat model → quality → patches → checklist (`workflows/code_review.py`)
- ✅ **Prompt Templates** — All workflow agent prompts (`prompts/storm_prompts.py`)
- ✅ **Model Query Helpers** — Parallel queries, response parsing (`workflows/model_query.py`)

### Phase 2.5 — ✅ IMPLEMENTED
- ✅ **Real Evidence Retrieval** — Researcher now performs actual retrieval:
  - `KeyTermExtractor` — Extracts technical terms (CamelCase, snake_case, acronyms), filters stop words
  - `RepoSearcher` — Async grep-based repository search with relevance scoring
  - `DocSearcher` — Pattern-based documentation search with context extraction
  - `SourceReliabilityScorer` — Multi-factor reliability scoring (source type, authority indicators)
- ✅ **Cross-Model Verification** — Basic verification based on evidence coverage (full model-based verification pending)

### Phase 3 — 🔲 PLANNED
- Optional persistence across sessions
- UI/mind-map visualization
- Full cross-model claim verification (query multiple models independently)
- Web retrieval integration

## 14) Implementation Notes

### New STORM Modes
```bash
# Decision workflow with options, rubric scoring, red-team
python3 skills/council/scripts/council.py --mode storm_decision --query "..."

# Research workflow with perspectives and critique
python3 skills/council/scripts/council.py --mode storm_research --query "..."

# Code review with threat modeling
python3 skills/council/scripts/council.py --mode storm_review --query "..." --context-file code.py
```

### New Files Created
```
skills/council/
├── pipelines/          # Pipeline abstraction
│   ├── __init__.py
│   ├── base.py
│   ├── classic.py
│   └── storm.py
├── core/
│   └── storm_trail.py  # STORM trail generation
├── agents/             # STORM agents
│   ├── moderator.py
│   ├── researcher.py
│   └── evidence_judge.py
├── workflows/          # Workflow graphs
│   ├── __init__.py
│   ├── base.py
│   ├── decision.py
│   ├── research.py
│   ├── code_review.py
│   └── model_query.py
├── prompts/            # Prompt templates
│   ├── __init__.py
│   └── storm_prompts.py
├── knowledge_base.py   # Shared KB
└── convergence.py      # Evidence-aware convergence
```

## 15) Open Questions
- Retrieval scope defaults (web vs. repo vs. docs) and opt-in flags.
- How to score source reliability (static heuristic vs. model-judged).
- Confidence formula weights (agreement vs. evidence vs. objections vs. diversity) — start with heuristic, tune from logs.

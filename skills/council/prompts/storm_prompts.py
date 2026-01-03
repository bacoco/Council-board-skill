"""
STORM Prompt Templates - Structured prompts for STORM workflow agents.

Each prompt is designed to elicit structured, actionable responses that
can be parsed and added to the KnowledgeBase.
"""

from typing import Any, Dict, List, Optional


def format_prompt(template: str, **kwargs) -> str:
    """Format a prompt template with provided values."""
    return template.format(**kwargs)


def format_kb_context(kb_dict: Dict[str, Any], max_claims: int = 10) -> str:
    """Format KnowledgeBase state as context for prompts."""
    parts = []

    # Claims
    claims = kb_dict.get('claims', [])[:max_claims]
    if claims:
        parts.append("Current Claims:")
        for c in claims:
            status = c.get('status', 'proposed')
            parts.append(f"- [{status}] {c.get('text', '')[:100]}")

    # Open Questions
    questions = [q for q in kb_dict.get('open_questions', []) if q.get('status') == 'open']
    if questions:
        parts.append("\nOpen Questions:")
        for q in questions[:5]:
            parts.append(f"- {q.get('prompt', '')}")

    # Decisions
    decisions = kb_dict.get('decisions', [])
    if decisions:
        parts.append("\nPrior Decisions:")
        for d in decisions[:3]:
            parts.append(f"- {d.get('summary', '')}")

    return '\n'.join(parts) if parts else "No prior context."


# =============================================================================
# Moderator Prompts
# =============================================================================

MODERATOR_ROUTING_PROMPT = """You are the Moderator for a multi-model deliberation system.

QUERY: {query}

CURRENT STATE:
{kb_context}

ROUND: {round_num} of {max_rounds}
AGREEMENT LEVEL: {agreement:.0%}
EVIDENCE COVERAGE: {coverage:.0%}

Analyze the current state and decide the next action. Choose ONE:
1. CONTINUE - More deliberation needed
2. RETRIEVE - Need evidence for unsupported claims
3. VERIFY - Need to resolve contradictions
4. DEBATE - Escalate to structured debate
5. FINALIZE - Ready for synthesis

Respond in this exact format:
ACTION: [your choice]
REASON: [one sentence explanation]
CLAIMS_NEEDING_EVIDENCE: [comma-separated claim IDs or "none"]
OPEN_QUESTIONS: [key questions to resolve or "none"]
"""

MODERATOR_SHALLOW_CONSENSUS_PROMPT = """Analyze this deliberation for shallow consensus.

Shallow consensus = panelists agree but without citing evidence or addressing objections.

ROUND OUTPUTS:
{round_outputs}

EVIDENCE COVERAGE: {coverage:.0%}
AGREEMENT LEVEL: {agreement:.0%}

Is this shallow consensus? Respond:
SHALLOW: [yes/no]
REASON: [explanation]
MISSING_EVIDENCE: [what evidence is needed]
"""


# =============================================================================
# Decision Workflow Prompts
# =============================================================================

DECISION_GENERATE_OPTIONS_PROMPT = """Generate solution options for this decision.

QUERY: {query}

CONTEXT:
{context}

Generate 2-4 distinct options. For each option provide:
1. A clear name
2. Brief description (1-2 sentences)
3. Key pros (2-3 points)
4. Key cons (2-3 points)

Format your response as:
## Option 1: [Name]
Description: [description]
Pros:
- [pro 1]
- [pro 2]
Cons:
- [con 1]
- [con 2]

## Option 2: [Name]
...
"""

DECISION_RUBRIC_SCORING_PROMPT = """Score each option against these criteria.

OPTIONS:
{options}

CRITERIA (weight in parentheses):
{criteria}

For each option, provide a score 0-100 for each criterion.

Format your response as:
## [Option Name]
- feasibility: [score] - [brief rationale]
- effectiveness: [score] - [brief rationale]
- maintainability: [score] - [brief rationale]
- risk: [score] - [brief rationale]
- cost: [score] - [brief rationale]
TOTAL: [weighted average]

## [Next Option]
...
"""

DECISION_RED_TEAM_PROMPT = """Red-team these top options. Find risks, failure modes, and edge cases.

OPTIONS TO ANALYZE:
{options}

For each option, identify:
1. Potential failure modes
2. Hidden risks or assumptions
3. Edge cases that could cause problems
4. What could go wrong in practice

Format as:
## [Option Name] - Risk Analysis
FAILURE MODES:
- [failure mode 1]: [impact] - [likelihood]
- [failure mode 2]: [impact] - [likelihood]

HIDDEN RISKS:
- [risk 1]
- [risk 2]

EDGE CASES:
- [edge case 1]

OVERALL RISK LEVEL: [low/medium/high/critical]
"""

DECISION_RECOMMENDATION_PROMPT = """Synthesize a final recommendation.

QUERY: {query}

OPTIONS ANALYZED:
{options_summary}

RISK ANALYSIS:
{risk_summary}

EVIDENCE COVERAGE: {coverage:.0%}

Provide:
1. Your recommended option and why
2. Key tradeoffs to acknowledge
3. Tripwires (conditions that should trigger revisiting this decision)
4. Confidence level (0-100%) with rationale

Format as:
RECOMMENDATION: [option name]
RATIONALE: [2-3 sentences]

TRADEOFFS:
- [tradeoff 1]
- [tradeoff 2]

TRIPWIRES (revisit if):
- [condition 1]
- [condition 2]

CONFIDENCE: [X]%
CONFIDENCE_RATIONALE: [why this confidence level]
"""


# =============================================================================
# Research Workflow Prompts
# =============================================================================

RESEARCH_PERSPECTIVES_PROMPT = """Identify relevant perspectives for researching this topic.

QUERY: {query}

Identify 3-5 distinct perspectives or angles from which to examine this topic.
Consider: technical, historical, practical, comparative, theoretical, etc.

For each perspective, provide:
1. Name of the perspective
2. What it focuses on
3. 2-3 key questions from this perspective

Format as:
## Perspective 1: [Name]
Focus: [what this perspective examines]
Key Questions:
- [question 1]
- [question 2]
- [question 3]

## Perspective 2: [Name]
...
"""

RESEARCH_QUESTIONS_PROMPT = """Generate sub-questions to explore this topic thoroughly.

QUERY: {query}

PERSPECTIVES IDENTIFIED:
{perspectives}

Generate 5-10 specific, answerable questions that would help build a comprehensive understanding.
Questions should:
- Be specific enough to research
- Cover different aspects of the topic
- Progress from foundational to advanced

Format as numbered list:
1. [question]
2. [question]
...
"""

RESEARCH_DRAFT_PROMPT = """Write a comprehensive draft based on the research findings.

QUERY: {query}

OUTLINE:
{outline}

EVIDENCE/FINDINGS:
{findings}

Write a clear, well-structured response that:
1. Addresses the main query directly
2. Covers each section in the outline
3. Cites evidence where available
4. Acknowledges gaps or uncertainties

Use markdown formatting with headers for each section.
"""

RESEARCH_CRITIQUE_PROMPT = """Critique this draft for gaps and weaknesses.

DRAFT:
{draft}

ORIGINAL QUERY: {query}

Identify:
1. Missing information or perspectives
2. Unsupported claims
3. Areas needing more depth
4. Potential biases or one-sided arguments
5. Factual accuracy concerns

Format as:
GAPS:
- [gap 1]
- [gap 2]

UNSUPPORTED CLAIMS:
- [claim 1] - [what evidence is needed]

AREAS NEEDING DEPTH:
- [area 1]

CONCERNS:
- [concern 1]

OVERALL ASSESSMENT: [brief summary]
"""


# =============================================================================
# Code Review Workflow Prompts
# =============================================================================

REVIEW_STATIC_SCAN_PROMPT = """Perform initial static analysis on this code.

CODE:
```
{code}
```

FOCUS AREAS: {focus}

Identify:
1. Obvious bugs or errors
2. Code style issues
3. Potential security concerns (flag for deeper analysis)
4. Performance red flags
5. Maintainability issues

For each issue, provide:
- Severity: critical/high/medium/low/info
- Category: security/performance/maintainability/correctness/style
- Location: line number or function name if possible
- Description: what the issue is
- Suggestion: how to fix it

Format as:
## Issue 1
Severity: [level]
Category: [category]
Location: [location]
Description: [description]
Suggestion: [suggestion]

## Issue 2
...
"""

REVIEW_THREAT_MODEL_PROMPT = """Perform security threat modeling on this code.

CODE:
```
{code}
```

CONTEXT: {context}

Consider:
1. Input validation vulnerabilities
2. Authentication/authorization issues
3. Data exposure risks
4. Injection vulnerabilities
5. Cryptographic weaknesses

For each threat, provide:
- Name: brief threat name
- Attack Vector: how an attacker would exploit this
- Impact: what damage could result
- Likelihood: low/medium/high
- Mitigation: how to address this

Format as:
## Threat 1: [Name]
Attack Vector: [description]
Impact: [description]
Likelihood: [low/medium/high]
Severity: [critical/high/medium/low]
Mitigation: [recommendation]

## Threat 2: [Name]
...
"""

REVIEW_QUALITY_PROMPT = """Analyze code quality: performance and maintainability.

CODE:
```
{code}
```

Evaluate:
1. PERFORMANCE
   - Algorithmic complexity
   - Resource usage
   - Potential bottlenecks

2. MAINTAINABILITY
   - Code organization
   - Naming clarity
   - Documentation
   - Test coverage indicators

3. BEST PRACTICES
   - Design patterns usage
   - Error handling
   - Code duplication

Format as:
## Performance
Score: [1-10]
Issues:
- [issue 1]
Recommendations:
- [recommendation 1]

## Maintainability
Score: [1-10]
Issues:
- [issue 1]
Recommendations:
- [recommendation 1]

## Best Practices
Score: [1-10]
Issues:
- [issue 1]
Recommendations:
- [recommendation 1]

OVERALL QUALITY: [1-10]
"""

REVIEW_PATCHES_PROMPT = """Generate concrete patch suggestions for the identified issues.

ISSUES:
{issues}

THREATS:
{threats}

CODE CONTEXT:
```
{code}
```

For each fixable issue, provide a specific patch or code change.
Prioritize by severity.

Format as:
## Patch 1: [Issue being fixed]
Priority: [critical/high/medium/low]
Change:
```
[code snippet showing the fix]
```
Explanation: [why this fixes the issue]

## Patch 2: [Issue being fixed]
...
"""


# =============================================================================
# Evidence Prompts
# =============================================================================

EVIDENCE_JUDGE_PROMPT = """Evaluate these claims against the available evidence.

CLAIMS:
{claims}

EVIDENCE SOURCES:
{sources}

For each claim, determine:
1. Is it supported by evidence?
2. Is it contradicted by evidence?
3. Is there insufficient evidence to judge?

Format as:
## Claim: [claim text]
Status: [supported/contradicted/insufficient/unsupported]
Evidence: [source IDs that relate to this claim]
Confidence: [0-100]%
Rationale: [brief explanation]

## Claim: [next claim]
...

SUMMARY:
- Claims supported: [count]
- Claims contradicted: [count]
- Claims unsupported: [count]
"""

CLAIM_EXTRACTION_PROMPT = """Extract distinct claims from this response.

RESPONSE:
{response}

AUTHOR: {author}

Extract each distinct factual or analytical claim made.
Exclude opinions, questions, and meta-commentary.

Format as:
1. [claim text] | Confidence: [low/medium/high] | Speculative: [yes/no]
2. [claim text] | Confidence: [low/medium/high] | Speculative: [yes/no]
...
"""

"""
STORM Trail Generation - Trail files for STORM workflow deliberations.

Extends the standard trail format with:
- KnowledgeBase snapshots (claims, sources, decisions)
- Workflow node execution details
- Moderator routing decisions
- Evidence coverage metrics
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def generate_storm_trail_markdown(
    session_id: str,
    query: str,
    mode: str,
    workflow_name: str,
    node_results: Dict[str, Dict[str, Any]],
    kb_snapshot: Dict[str, Any],
    moderator_history: List[Dict[str, Any]],
    convergence: Dict[str, Any],
    evidence_report: Dict[str, Any],
    final_answer: str,
    confidence: float,
    duration_ms: int,
    models: List[str],
    context_preview: str = ""
) -> str:
    """
    Generate a human-readable Markdown document from STORM workflow execution.

    Args:
        session_id: Session identifier
        query: Original question
        mode: STORM mode (storm_decision, storm_research, storm_review)
        workflow_name: Name of the workflow graph executed
        node_results: Results from each workflow node
        kb_snapshot: Final KnowledgeBase state
        moderator_history: Moderator routing decisions
        convergence: Convergence check results
        evidence_report: Evidence evaluation results
        final_answer: Final synthesized answer
        confidence: Final confidence score
        duration_ms: Total execution duration
        models: Models used in deliberation
        context_preview: Preview of context provided

    Returns:
        Formatted Markdown string
    """
    lines = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Header
    lines.append("# STORM Deliberation Trail")
    lines.append("")
    lines.append("## Session Metadata")
    lines.append("")
    lines.append(f"- **Session ID**: `{session_id}`")
    lines.append(f"- **Timestamp**: {timestamp}")
    lines.append(f"- **Duration**: {duration_ms / 1000:.1f}s")
    lines.append(f"- **Mode**: {mode}")
    lines.append(f"- **Workflow**: {workflow_name}")
    lines.append(f"- **Models**: {', '.join(models)}")
    lines.append(f"- **Final Confidence**: {confidence:.2f}")
    lines.append("")

    # Query
    lines.append("## Query")
    lines.append("")
    lines.append(f"> {query}")
    lines.append("")

    if context_preview:
        lines.append("### Context Preview")
        lines.append("")
        lines.append("```")
        lines.append(context_preview[:500] + ("..." if len(context_preview) > 500 else ""))
        lines.append("```")
        lines.append("")

    lines.append("---")
    lines.append("")

    # Workflow Execution
    lines.append("## Workflow Execution")
    lines.append("")

    for node_id, result in node_results.items():
        # Handle both dict and NodeResult objects
        if hasattr(result, 'to_dict'):
            result = result.to_dict()
        elif not isinstance(result, dict):
            result = {'status': 'unknown', 'output': {}}

        status = result.get('status', 'unknown')
        status_icon = "✓" if status == 'completed' else "✗" if status == 'failed' else "○"
        latency = result.get('latency_ms', 0)

        lines.append(f"### {status_icon} Node: {node_id}")
        lines.append("")
        lines.append(f"- **Status**: {status}")
        lines.append(f"- **Latency**: {latency}ms")

        # Node-specific output
        output = result.get('output', {})
        if output:
            lines.append("")
            lines.append("**Output Summary**:")
            lines.append("")

            # Format key outputs based on node type
            if node_id == 'generate_options':
                options = output.get('options', [])
                lines.append(f"Generated {len(options)} options:")
                for opt in options[:5]:  # Limit display
                    lines.append(f"- **{opt.get('name', 'Unknown')}**: {opt.get('description', '')[:100]}")

            elif node_id == 'rubric_scoring':
                scored = output.get('scored_options', [])
                if scored:
                    lines.append("| Option | Score |")
                    lines.append("|--------|-------|")
                    for opt in scored[:5]:
                        lines.append(f"| {opt.get('name', '?')} | {opt.get('total_score', 0):.0%} |")

            elif node_id == 'red_team':
                risks = output.get('risks_by_option', {})
                total = output.get('total_risks', 0)
                lines.append(f"Identified {total} risk(s) across options")

            elif node_id == 'recommendation':
                rec = output.get('recommended_option', 'N/A')
                conf = output.get('confidence', 0)
                lines.append(f"- **Recommendation**: {rec}")
                lines.append(f"- **Confidence**: {conf:.0%}")

            elif node_id == 'checklist':
                total = output.get('total_items', 0)
                critical = output.get('critical_items', 0)
                lines.append(f"- **Total Items**: {total}")
                lines.append(f"- **Critical Items**: {critical}")

            else:
                # Generic output display
                for key, value in list(output.items())[:5]:
                    if isinstance(value, (str, int, float)):
                        lines.append(f"- {key}: {value}")

        # Claims/sources added
        claims = result.get('claims_added', [])
        sources = result.get('sources_added', [])
        questions = result.get('questions_added', [])

        if claims or sources or questions:
            lines.append("")
            lines.append("**KB Contributions**:")
            if claims:
                lines.append(f"- Claims added: {len(claims)}")
            if sources:
                lines.append(f"- Sources added: {len(sources)}")
            if questions:
                lines.append(f"- Questions added: {len(questions)}")

        if result.get('error'):
            lines.append("")
            lines.append(f"**Error**: {result['error']}")

        lines.append("")
        lines.append("---")
        lines.append("")

    # Moderator Decisions
    if moderator_history:
        lines.append("## Moderator Decisions")
        lines.append("")
        lines.append("| Round | Action | Reason |")
        lines.append("|-------|--------|--------|")
        for event in moderator_history:
            decision = event.get('decision', {})
            round_num = event.get('round', '?')
            action = decision.get('action', 'unknown')
            reason = decision.get('reason', '')[:50]
            lines.append(f"| {round_num} | {action} | {reason} |")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Knowledge Base Snapshot
    lines.append("## Knowledge Base Snapshot")
    lines.append("")

    # Metrics
    metrics = kb_snapshot.get('metrics', {})
    lines.append("### Metrics")
    lines.append("")
    lines.append(f"- **Evidence Coverage**: {metrics.get('evidence_coverage', 0):.0%}")
    lines.append(f"- **Unresolved Objections**: {metrics.get('unresolved_objections', 0)}")
    lines.append(f"- **Source Diversity**: {metrics.get('source_diversity', 0):.0%}")
    lines.append("")

    # Claims
    claims = kb_snapshot.get('claims', [])
    if claims:
        lines.append("### Claims")
        lines.append("")
        lines.append("| Status | Owner | Claim | Confidence |")
        lines.append("|--------|-------|-------|------------|")
        for claim in claims[:10]:  # Limit display
            status = claim.get('status', 'proposed')
            owner = claim.get('owner', '?')[:15]
            text = claim.get('text', '')[:50] + ("..." if len(claim.get('text', '')) > 50 else "")
            conf = claim.get('confidence', 0)
            lines.append(f"| {status} | {owner} | {text} | {conf:.0%} |")
        if len(claims) > 10:
            lines.append(f"| ... | ... | *({len(claims) - 10} more claims)* | ... |")
        lines.append("")

    # Sources
    sources = kb_snapshot.get('sources', [])
    if sources:
        lines.append("### Sources")
        lines.append("")
        for src in sources[:5]:
            lines.append(f"- **{src.get('id', '?')}**: {src.get('uri', 'unknown')[:60]}")
            lines.append(f"  - Reliability: {src.get('reliability', 0):.0%}, Relevance: {src.get('relevance', 0):.0%}")
        lines.append("")

    # Decisions
    decisions = kb_snapshot.get('decisions', [])
    if decisions:
        lines.append("### Decisions")
        lines.append("")
        for dec in decisions:
            lines.append(f"**{dec.get('summary', 'Decision')}** (Confidence: {dec.get('confidence', 0):.0%})")
            lines.append("")
            if dec.get('tradeoffs'):
                lines.append("Tradeoffs:")
                for t in dec['tradeoffs'][:3]:
                    lines.append(f"- {t}")
            if dec.get('tripwires'):
                lines.append("")
                lines.append("Tripwires:")
                for t in dec['tripwires'][:3]:
                    lines.append(f"- {t}")
            lines.append("")

    lines.append("---")
    lines.append("")

    # Convergence Analysis
    lines.append("## Convergence Analysis")
    lines.append("")
    lines.append(f"- **Converged**: {'Yes' if convergence.get('converged') else 'No'}")
    lines.append(f"- **Score**: {convergence.get('score', 0):.3f} (threshold: {convergence.get('threshold', 0.8):.2f})")
    lines.append("")

    components = convergence.get('components', {})
    if components:
        lines.append("### Component Scores")
        lines.append("")
        lines.append("| Component | Score |")
        lines.append("|-----------|-------|")
        for comp, score in components.items():
            lines.append(f"| {comp} | {score:.0%} |")
        lines.append("")

    lines.append(f"**Rationale**: {convergence.get('confidence_rationale', 'N/A')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Evidence Report
    lines.append("## Evidence Report")
    lines.append("")
    lines.append(f"- **Overall Coverage**: {evidence_report.get('overall_coverage', 0):.0%}")
    lines.append(f"- **Unsupported Claims**: {len(evidence_report.get('unsupported_claims', []))}")
    lines.append(f"- **Contradicted Claims**: {len(evidence_report.get('contradicted_claims', []))}")
    lines.append("")

    notes = evidence_report.get('evaluation_notes', [])
    if notes:
        lines.append("### Notes")
        lines.append("")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    lines.append("---")
    lines.append("")

    # Final Answer
    lines.append("## Final Answer")
    lines.append("")
    lines.append(final_answer)
    lines.append("")

    return "\n".join(lines)


def save_storm_trail_to_file(
    markdown_content: str,
    session_id: str,
    query: str,
    mode: str = "storm_decision",
    output_dir: str = "."
) -> Path:
    """
    Save STORM trail Markdown to file and return the path.

    Args:
        markdown_content: Generated Markdown string
        session_id: Session ID
        query: Original query (for filename)
        mode: STORM mode
        output_dir: Directory to save file (defaults to current directory)

    Returns:
        Path to the saved file
    """
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filename
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%Hh%M")

    # Create slug from query
    words = re.sub(r'[^a-zA-Z0-9\s]', '', query).lower().split()[:5]
    slug = "-".join(words) if words else "query"
    slug = slug[:30]

    filename = f"storm_{date_str}_{time_str}_{mode}_{slug}.md"
    filepath = output_path / filename

    filepath.write_text(markdown_content, encoding='utf-8')
    return filepath

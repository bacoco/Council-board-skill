"""
Trail markdown generation for Council deliberation.

Generates human-readable Markdown documents from deliberation trails.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional


def generate_trail_markdown(
    session_id: str,
    query: str,
    mode: str,
    deliberation_trail: List[dict],
    synthesis: dict,
    review: dict,
    devils_advocate_summary: Optional[dict],
    duration_ms: int,
    converged: bool,
    convergence_score: float,
    confidence: float,
    excluded_models: List[dict] = None,
    config_models: List[str] = None
) -> str:
    """
    Generate a human-readable Markdown document from the deliberation trail.

    Args:
        session_id: Council session identifier
        query: Original question
        mode: Deliberation mode (consensus, debate, etc.)
        deliberation_trail: List of round entries
        synthesis: Final synthesis result
        review: Peer review scores
        devils_advocate_summary: Devil's advocate analysis if applicable
        duration_ms: Total deliberation duration
        converged: Whether consensus was reached
        convergence_score: Convergence score (0-1)
        confidence: Final confidence score
        excluded_models: Models that were skipped/failed
        config_models: All configured models

    Returns:
        Formatted Markdown string with full reasoning chain
    """
    lines = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Header
    lines.append("# Council Deliberation Trail")
    lines.append("")
    lines.append("## Session Metadata")
    lines.append("")
    lines.append(f"- **Session ID**: `{session_id}`")
    lines.append(f"- **Timestamp**: {timestamp}")
    lines.append(f"- **Duration**: {duration_ms / 1000:.1f}s")
    lines.append(f"- **Mode**: {mode}")
    lines.append(f"- **Converged**: {'Yes' if converged else 'No'} (score: {convergence_score:.3f})")
    lines.append(f"- **Final Confidence**: {confidence:.2f}")
    lines.append("")
    lines.append("## Query")
    lines.append("")
    lines.append(f"> {query}")
    lines.append("")

    # Participation Status section - show which models participated vs failed/skipped
    if config_models:
        lines.append("## Participation Status")
        lines.append("")
        lines.append("| Model | Status | Details |")
        lines.append("|-------|--------|---------|")

        # Build participation info from trail and excluded_models
        participated_models = set()
        for entry in deliberation_trail:
            participated_models.add(entry.get("model", ""))

        excluded_by_model = {}
        if excluded_models:
            for exc in excluded_models:
                model = exc.get("model", "")
                if model not in excluded_by_model:
                    excluded_by_model[model] = exc

        for model in config_models:
            base_model = model.split('_instance_')[0] if '_instance_' in model else model
            if model in participated_models or base_model in participated_models:
                lines.append(f"| {base_model} | ✓ Participated | - |")
            elif model in excluded_by_model:
                exc = excluded_by_model[model]
                status = exc.get("status", "FAILED")
                reason = exc.get("reason", "unknown")
                # Truncate long reasons
                if len(reason) > 50:
                    reason = reason[:47] + "..."
                lines.append(f"| {base_model} | ✗ {status} | {reason} |")
            elif base_model in excluded_by_model:
                exc = excluded_by_model[base_model]
                status = exc.get("status", "FAILED")
                reason = exc.get("reason", "unknown")
                if len(reason) > 50:
                    reason = reason[:47] + "..."
                lines.append(f"| {base_model} | ✗ {status} | {reason} |")
            else:
                lines.append(f"| {base_model} | ? Unknown | No response recorded |")

        lines.append("")

    # Group trail entries by round
    rounds_data = {}
    for entry in deliberation_trail:
        round_num = entry["round"]
        if round_num not in rounds_data:
            rounds_data[round_num] = []
        rounds_data[round_num].append(entry)

    # Deliberation Rounds
    lines.append("---")
    lines.append("")
    lines.append("## Deliberation Rounds")
    lines.append("")

    for round_num in sorted(rounds_data.keys()):
        lines.append(f"### Round {round_num}")
        lines.append("")

        for entry in rounds_data[round_num]:
            persona = entry.get("persona", entry.get("model", "Unknown"))
            role = entry.get("persona_role", "")
            conf = entry.get("confidence", 0.0)
            latency = entry.get("latency_ms", 0)
            answer = entry.get("answer", "")
            key_points = entry.get("key_points", [])
            model = entry.get("model", "")

            lines.append(f"#### {persona}")
            lines.append(f"*Model: {model}*")
            if role:
                lines.append(f"*{role}*")
            lines.append("")
            lines.append(f"**Confidence**: {conf:.2f} | **Latency**: {latency}ms")
            lines.append("")

            # Answer
            lines.append("**Response**:")
            lines.append("")
            lines.append(answer)
            lines.append("")

            # Key Points
            if key_points:
                lines.append("**Key Points**:")
                for point in key_points:
                    lines.append(f"- {point}")
                lines.append("")

            lines.append("---")
            lines.append("")

    # Devil's Advocate Summary (if present)
    if devils_advocate_summary:
        lines.append("## Devil's Advocate Analysis")
        lines.append("")

        attackers = devils_advocate_summary.get("attacker", [])
        defenders = devils_advocate_summary.get("defender", [])
        synthesizers = devils_advocate_summary.get("synthesizer", [])
        takeaways = devils_advocate_summary.get("headline_takeaways", [])

        if attackers:
            lines.append("### Red Team (Attacker)")
            for point in attackers:
                lines.append(f"- {point}")
            lines.append("")

        if defenders:
            lines.append("### Blue Team (Defender)")
            for point in defenders:
                lines.append(f"- {point}")
            lines.append("")

        if synthesizers:
            lines.append("### Purple Team (Synthesizer)")
            for point in synthesizers:
                lines.append(f"- {point}")
            lines.append("")

        if takeaways:
            lines.append("### Key Takeaways")
            for point in takeaways:
                lines.append(f"- {point}")
            lines.append("")

        lines.append("---")
        lines.append("")

    # Peer Review Scores
    if review and isinstance(review, dict):
        scores = review.get("scores", review)
        if scores:
            lines.append("## Peer Review Scores")
            lines.append("")
            lines.append("| Participant | Accuracy | Completeness | Reasoning | Clarity | Total |")
            lines.append("|-------------|----------|--------------|-----------|---------|-------|")
            for participant, score_data in scores.items():
                if isinstance(score_data, dict):
                    acc = score_data.get("accuracy", "-")
                    comp = score_data.get("completeness", "-")
                    reas = score_data.get("reasoning", "-")
                    clar = score_data.get("clarity", "-")
                    total = sum(v for v in [acc, comp, reas, clar] if isinstance(v, (int, float)))
                    lines.append(f"| {participant} | {acc} | {comp} | {reas} | {clar} | {total}/20 |")
            lines.append("")
            lines.append("---")
            lines.append("")

    # Final Synthesis
    lines.append("## Council Consensus")
    lines.append("")
    final_answer = synthesis.get("final_answer", "") if synthesis else ""
    lines.append(final_answer)
    lines.append("")

    dissent = synthesis.get("dissenting_view") if synthesis else None
    if dissent:
        lines.append("### Dissenting View")
        lines.append("")
        if isinstance(dissent, dict):
            lines.append(f"**{dissent.get('advocate', 'Unknown')}**: {dissent.get('position', '')}")
            if dissent.get('rationale'):
                lines.append(f"*Rationale*: {dissent.get('rationale')}")
        else:
            lines.append(str(dissent))
        lines.append("")

    return "\n".join(lines)


def save_trail_to_file(
    markdown_content: str,
    session_id: str,
    query: str,
    mode: str = "consensus",
    output_dir: str = "./council_trails"
) -> Path:
    """
    Save trail Markdown to file and return the path.

    Args:
        markdown_content: Generated Markdown string
        session_id: Council session ID
        query: Original query (for filename)
        mode: Deliberation mode (consensus, debate, vote, etc.)
        output_dir: Directory to save file

    Returns:
        Path to the saved file
    """
    # Resolve path
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filename with readable timestamp and mode
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%Hh%M")

    # Create slug from query (first 5 words, alphanumeric only)
    words = re.sub(r'[^a-zA-Z0-9\s]', '', query).lower().split()[:5]
    slug = "-".join(words) if words else "query"
    slug = slug[:30]  # Limit length

    filename = f"council_{date_str}_{time_str}_{mode}_{slug}.md"
    filepath = output_path / filename

    # Write file
    filepath.write_text(markdown_content, encoding='utf-8')

    return filepath

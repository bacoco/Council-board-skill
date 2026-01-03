"""
Research Workflow Graph - Structured flow for research/explanatory queries.

Flow (from PRD):
1. Perspectives: Identify relevant perspectives/angles on the topic
2. Question Plan: Generate sub-questions to explore
3. Retrieve: Gather evidence for each sub-question
4. Outline: Structure the findings into an outline
5. Draft: Generate comprehensive draft
6. Critique: Self-critique and identify gaps
7. Final Report: Synthesize final research report
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .base import WorkflowGraph, WorkflowNode, WorkflowState, NodeResult, NodeStatus
from .model_query import query_chairman, query_models_parallel
from knowledge_base import KnowledgeBase
from prompts.storm_prompts import (
    RESEARCH_PERSPECTIVES_PROMPT, RESEARCH_QUESTIONS_PROMPT,
    RESEARCH_DRAFT_PROMPT, RESEARCH_CRITIQUE_PROMPT, format_prompt
)


@dataclass
class Perspective:
    """A perspective or angle for research."""
    id: str
    name: str
    description: str
    key_questions: List[str]


@dataclass
class ResearchSection:
    """A section in the research outline."""
    id: str
    title: str
    key_points: List[str]
    sources: List[str]  # Source IDs from KB


class ResearchGraph(WorkflowGraph):
    """
    Workflow graph for research/explanatory queries.

    Best for questions like:
    - "What is X and how does it work?"
    - "Explain the history of..."
    - "Give me an overview of..."
    """

    name = "research"
    description = "Research workflow with perspectives, retrieval, and synthesis"

    def __init__(self, state: WorkflowState):
        self._perspectives: List[Perspective] = []
        self._questions: List[str] = []
        self._outline: List[ResearchSection] = []
        self._draft: str = ""
        self._final_report: str = ""
        super().__init__(state)

    def _build_nodes(self) -> None:
        """Build the research workflow nodes."""

        # Node 1: Identify Perspectives
        self.add_node(WorkflowNode(
            id="perspectives",
            name="Identify Perspectives",
            description="Identify relevant angles and perspectives on the topic",
            execute_fn=self._node_perspectives
        ))

        # Node 2: Question Plan
        self.add_node(WorkflowNode(
            id="question_plan",
            name="Question Plan",
            description="Generate sub-questions to explore",
            dependencies=["perspectives"],
            execute_fn=self._node_question_plan
        ))

        # Node 3: Retrieve Evidence
        self.add_node(WorkflowNode(
            id="retrieve",
            name="Retrieve Evidence",
            description="Gather evidence for each sub-question",
            dependencies=["question_plan"],
            optional=True,  # Can proceed with limited evidence
            execute_fn=self._node_retrieve
        ))

        # Node 4: Build Outline
        self.add_node(WorkflowNode(
            id="outline",
            name="Build Outline",
            description="Structure findings into an outline",
            dependencies=["question_plan"],
            execute_fn=self._node_outline
        ))

        # Node 5: Draft Report
        self.add_node(WorkflowNode(
            id="draft",
            name="Draft Report",
            description="Generate comprehensive draft",
            dependencies=["outline"],
            execute_fn=self._node_draft
        ))

        # Node 6: Critique
        self.add_node(WorkflowNode(
            id="critique",
            name="Self-Critique",
            description="Identify gaps and weaknesses",
            dependencies=["draft"],
            optional=True,
            execute_fn=self._node_critique
        ))

        # Node 7: Final Report
        self.add_node(WorkflowNode(
            id="final_report",
            name="Final Report",
            description="Synthesize final research report",
            dependencies=["draft"],
            execute_fn=self._node_final_report
        ))

    async def _node_perspectives(self, state: WorkflowState) -> NodeResult:
        """
        Node 1: Identify perspectives on the topic.

        Queries the chairman model for diverse perspectives.
        """
        import time
        start = time.time()

        # Build prompt
        prompt = format_prompt(
            RESEARCH_PERSPECTIVES_PROMPT,
            query=state.query
        )

        # Query chairman for perspectives
        result = await query_chairman(prompt, state.chairman, state.timeout)
        latency = int((time.time() - start) * 1000)

        if not result.success:
            # Fallback to default perspectives
            self._perspectives = [
                Perspective(
                    id="p_technical",
                    name="Technical",
                    description="Technical/implementation perspective",
                    key_questions=["How does it work technically?", "What are the components?"]
                ),
                Perspective(
                    id="p_practical",
                    name="Practical",
                    description="Practical usage perspective",
                    key_questions=["How is it used in practice?", "What are common patterns?"]
                )
            ]
        else:
            # Parse perspectives from response
            self._perspectives = self._parse_perspectives(result.content)

        # Add concepts to KB
        for p in self._perspectives:
            state.kb.add_concept(
                title=p.name,
                definition=p.description,
                owner="research_graph",
                round_num=1
            )

        return NodeResult(
            node_id="perspectives",
            status=NodeStatus.COMPLETED,
            output={
                'topic': state.query[:100],
                'perspectives_count': len(self._perspectives),
                'perspectives': [
                    {'id': p.id, 'name': p.name, 'questions': p.key_questions}
                    for p in self._perspectives
                ]
            },
            latency_ms=latency
        )

    def _parse_perspectives(self, response: str) -> List[Perspective]:
        """Parse perspectives from model response."""
        perspectives = []
        pattern = r'##\s*Perspective\s*\d+:\s*(.+?)(?=##\s*Perspective|\Z)'
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)

        for i, match in enumerate(matches):
            lines = match.strip().split('\n')
            name = lines[0].strip() if lines else f"Perspective {i+1}"

            # Extract focus
            focus_match = re.search(r'Focus:\s*(.+?)(?:\n|$)', match, re.IGNORECASE)
            description = focus_match.group(1).strip() if focus_match else ""

            # Extract questions
            questions = []
            questions_section = re.search(r'Key Questions:\s*(.+?)(?=##|\Z)', match, re.DOTALL | re.IGNORECASE)
            if questions_section:
                for line in questions_section.group(1).split('\n'):
                    if line.strip().startswith('-'):
                        questions.append(line.strip().lstrip('- '))

            perspectives.append(Perspective(
                id=f"p_{i+1}",
                name=name,
                description=description,
                key_questions=questions[:3] if questions else [f"What about {name}?"]
            ))

        return perspectives if perspectives else [
            Perspective("p_1", "General", "General analysis", ["What are the key aspects?"])
        ]

    async def _node_question_plan(self, state: WorkflowState) -> NodeResult:
        """
        Node 2: Generate sub-questions to explore.
        """
        # Collect questions from all perspectives
        self._questions = []
        questions_added = []

        for p in self._perspectives:
            for q_text in p.key_questions:
                self._questions.append(q_text)
                q = state.kb.add_question(
                    prompt=q_text,
                    owner=f"perspective_{p.id}",
                    round_num=1
                )
                questions_added.append(q.id)

        return NodeResult(
            node_id="question_plan",
            status=NodeStatus.COMPLETED,
            output={
                'questions_count': len(self._questions),
                'questions': self._questions
            },
            questions_added=questions_added
        )

    async def _node_retrieve(self, state: WorkflowState) -> NodeResult:
        """
        Node 3: Retrieve evidence for sub-questions.

        Stub implementation - actual retrieval not yet implemented.
        """
        # Note: In full implementation, would use Researcher agent
        return NodeResult(
            node_id="retrieve",
            status=NodeStatus.COMPLETED,
            output={
                'note': 'Evidence retrieval not yet implemented',
                'questions_to_research': len(self._questions),
                'sources_found': 0
            }
        )

    async def _node_outline(self, state: WorkflowState) -> NodeResult:
        """
        Node 4: Build research outline.
        """
        self._outline = []

        for p in self._perspectives:
            section = ResearchSection(
                id=f"section_{p.id}",
                title=p.name,
                key_points=[f"Analysis of {q}" for q in p.key_questions],
                sources=[]
            )
            self._outline.append(section)

        return NodeResult(
            node_id="outline",
            status=NodeStatus.COMPLETED,
            output={
                'sections_count': len(self._outline),
                'outline': [
                    {'id': s.id, 'title': s.title, 'points': len(s.key_points)}
                    for s in self._outline
                ]
            }
        )

    async def _node_draft(self, state: WorkflowState) -> NodeResult:
        """
        Node 5: Generate draft report.

        Queries the chairman model to synthesize a comprehensive draft.
        """
        import time
        start = time.time()

        # Format outline for prompt
        outline_text = "\n".join([
            f"## {s.title}\n" + "\n".join(f"- {p}" for p in s.key_points)
            for s in self._outline
        ])

        # Build prompt
        prompt = format_prompt(
            RESEARCH_DRAFT_PROMPT,
            query=state.query,
            outline=outline_text,
            findings="(Based on structured analysis of the topic)"
        )

        # Query chairman for draft
        result = await query_chairman(prompt, state.chairman, state.timeout)
        latency = int((time.time() - start) * 1000)

        if result.success:
            self._draft = result.content
        else:
            # Fallback: build simple draft from outline
            draft_parts = [f"# Research: {state.query}\n"]
            for section in self._outline:
                draft_parts.append(f"\n## {section.title}\n")
                for point in section.key_points:
                    draft_parts.append(f"- {point}\n")
            self._draft = ''.join(draft_parts)

        # Add draft as a claim
        claim = state.kb.add_claim(
            text=f"Draft report generated with {len(self._outline)} sections",
            owner="research_graph",
            confidence=0.6 if not result.success else 0.75,
            round_num=1
        )

        return NodeResult(
            node_id="draft",
            status=NodeStatus.COMPLETED,
            output={
                'draft_length': len(self._draft),
                'sections': len(self._outline),
                'preview': self._draft[:500] + '...' if len(self._draft) > 500 else self._draft
            },
            claims_added=[claim.id],
            latency_ms=latency
        )

    async def _node_critique(self, state: WorkflowState) -> NodeResult:
        """
        Node 6: Self-critique the draft.

        Queries models to identify gaps and weaknesses.
        """
        import time
        start = time.time()

        # Build prompt
        prompt = format_prompt(
            RESEARCH_CRITIQUE_PROMPT,
            draft=self._draft[:4000],  # Limit draft length
            query=state.query
        )

        # Query chairman for critique
        result = await query_chairman(prompt, state.chairman, state.timeout)
        latency = int((time.time() - start) * 1000)

        if result.success:
            critiques = self._parse_critique(result.content)
        else:
            # Fallback critiques
            critiques = [
                "Draft could benefit from additional examples",
                "Consider adding more specific details"
            ]

        # Add critiques as open questions
        questions_added = []
        for critique in critiques:
            q = state.kb.add_question(
                prompt=f"Address: {critique}",
                owner="critique",
                round_num=1
            )
            questions_added.append(q.id)

        return NodeResult(
            node_id="critique",
            status=NodeStatus.COMPLETED,
            output={
                'critiques': critiques,
                'severity': 'minor' if len(critiques) <= 2 else 'moderate',
                'actionable': True
            },
            questions_added=questions_added,
            latency_ms=latency
        )

    def _parse_critique(self, response: str) -> List[str]:
        """Parse critique points from model response."""
        critiques = []

        # Look for GAPS section
        gaps_match = re.search(r'GAPS:\s*(.+?)(?=UNSUPPORTED|AREAS|CONCERNS|\Z)', response, re.DOTALL | re.IGNORECASE)
        if gaps_match:
            for line in gaps_match.group(1).split('\n'):
                if line.strip().startswith('-'):
                    critiques.append(line.strip().lstrip('- '))

        # Look for UNSUPPORTED CLAIMS section
        unsupported_match = re.search(r'UNSUPPORTED CLAIMS:\s*(.+?)(?=AREAS|CONCERNS|\Z)', response, re.DOTALL | re.IGNORECASE)
        if unsupported_match:
            for line in unsupported_match.group(1).split('\n'):
                if line.strip().startswith('-'):
                    critiques.append(line.strip().lstrip('- '))

        # Look for AREAS NEEDING DEPTH
        areas_match = re.search(r'AREAS NEEDING DEPTH:\s*(.+?)(?=CONCERNS|\Z)', response, re.DOTALL | re.IGNORECASE)
        if areas_match:
            for line in areas_match.group(1).split('\n'):
                if line.strip().startswith('-'):
                    critiques.append(line.strip().lstrip('- '))

        return critiques[:5] if critiques else ["Review for completeness"]

    async def _node_final_report(self, state: WorkflowState) -> NodeResult:
        """
        Node 7: Generate final report.
        """
        # Get critique if available
        critique_output = state.get_node_output("critique") or {}
        critiques = critique_output.get('critiques', [])

        # Build final report
        report_parts = [self._draft]

        if critiques:
            report_parts.append("\n---\n**Note:** The following areas could benefit from additional research:\n")
            for c in critiques:
                report_parts.append(f"- {c}\n")

        # Add evidence coverage note
        coverage = state.kb.evidence_coverage()
        report_parts.append(f"\n---\n**Evidence Coverage:** {coverage:.0%}\n")

        self._final_report = ''.join(report_parts)

        # Add decision to KB
        state.kb.add_decision(
            summary=f"Research report on: {state.query[:50]}",
            tradeoffs=[f"Based on {len(self._perspectives)} perspectives"],
            tripwires=["Update when new information becomes available"],
            confidence=0.7,
            round_num=1
        )

        return NodeResult(
            node_id="final_report",
            status=NodeStatus.COMPLETED,
            output={
                'report_length': len(self._final_report),
                'perspectives_covered': len(self._perspectives),
                'questions_addressed': len(self._questions),
                'evidence_coverage': coverage
            }
        )

    def _build_final_output(self) -> Dict[str, Any]:
        """Build final research output."""
        return {
            'success': True,
            'workflow': self.name,
            'report': self._final_report,
            'perspectives': [p.name for p in self._perspectives],
            'questions_explored': self._questions,
            'outline': [
                {'title': s.title, 'points': s.key_points}
                for s in self._outline
            ],
            'node_results': {
                node_id: result.to_dict()
                for node_id, result in self.state.node_results.items()
            },
            'kb_snapshot': self.state.kb.to_dict()
        }

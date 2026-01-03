"""
Decision Workflow Graph - Structured flow for decision-making queries.

Flow (from PRD):
1. Generate Options: Panelists propose solution options
2. Rubric Scoring: Evaluate options against criteria
3. Red-Team Risks: Identify risks and failure modes for top options
4. Evidence Check: Verify claims with available evidence
5. Recommendation: Final recommendation with tradeoffs and tripwires
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import asyncio

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .base import WorkflowGraph, WorkflowNode, WorkflowState, NodeResult, NodeStatus
from knowledge_base import KnowledgeBase, ClaimStatus


@dataclass
class Option:
    """A solution option proposed during deliberation."""
    id: str
    name: str
    description: str
    proposer: str
    pros: List[str]
    cons: List[str]
    score: float = 0.0  # Set during rubric scoring


@dataclass
class RubricCriterion:
    """A criterion for evaluating options."""
    name: str
    weight: float  # 0.0-1.0
    description: str


class DecisionGraph(WorkflowGraph):
    """
    Workflow graph for decision-making queries.

    Best for questions like:
    - "Should we use X or Y?"
    - "What's the best approach for..."
    - "Compare options A, B, C"
    """

    name = "decision"
    description = "Decision-making workflow with options, scoring, and risk analysis"

    # Default rubric criteria
    DEFAULT_CRITERIA = [
        RubricCriterion("feasibility", 0.25, "How practical is this to implement?"),
        RubricCriterion("effectiveness", 0.30, "How well does it solve the problem?"),
        RubricCriterion("maintainability", 0.20, "How easy to maintain long-term?"),
        RubricCriterion("risk", 0.15, "What's the risk level? (inverse)"),
        RubricCriterion("cost", 0.10, "What's the resource cost? (inverse)")
    ]

    def __init__(self, state: WorkflowState, criteria: List[RubricCriterion] = None):
        self.criteria = criteria or self.DEFAULT_CRITERIA
        self._options: List[Option] = []
        self._recommendation: Optional[Dict[str, Any]] = None
        super().__init__(state)

    def _build_nodes(self) -> None:
        """Build the decision workflow nodes."""

        # Node 1: Generate Options
        self.add_node(WorkflowNode(
            id="generate_options",
            name="Generate Options",
            description="Panelists propose solution options with pros/cons",
            execute_fn=self._node_generate_options
        ))

        # Node 2: Rubric Scoring
        self.add_node(WorkflowNode(
            id="rubric_scoring",
            name="Rubric Scoring",
            description="Evaluate each option against criteria",
            dependencies=["generate_options"],
            execute_fn=self._node_rubric_scoring
        ))

        # Node 3: Red-Team Risks
        self.add_node(WorkflowNode(
            id="red_team",
            name="Red-Team Analysis",
            description="Identify risks and failure modes for top options",
            dependencies=["rubric_scoring"],
            execute_fn=self._node_red_team
        ))

        # Node 4: Evidence Check
        self.add_node(WorkflowNode(
            id="evidence_check",
            name="Evidence Check",
            description="Verify claims against available evidence",
            dependencies=["red_team"],
            optional=True,  # Can proceed without evidence
            execute_fn=self._node_evidence_check
        ))

        # Node 5: Final Recommendation
        self.add_node(WorkflowNode(
            id="recommendation",
            name="Recommendation",
            description="Synthesize final recommendation with tripwires",
            dependencies=["rubric_scoring", "red_team"],
            execute_fn=self._node_recommendation
        ))

    async def _node_generate_options(self, state: WorkflowState) -> NodeResult:
        """
        Node 1: Generate solution options.

        In full implementation, this would query each model to propose options.
        Currently uses placeholder logic.
        """
        # TODO: Query models for options
        # For now, create placeholder structure

        # Add a question to KB for tracking
        q = state.kb.add_question(
            prompt=f"What are the viable options for: {state.query}",
            owner="decision_graph",
            round_num=1
        )

        # Placeholder options (would come from model responses)
        self._options = [
            Option(
                id="opt_1",
                name="Option A",
                description="First approach identified from query analysis",
                proposer="system",
                pros=["To be determined by model analysis"],
                cons=["To be determined by model analysis"]
            ),
            Option(
                id="opt_2",
                name="Option B",
                description="Alternative approach for comparison",
                proposer="system",
                pros=["To be determined by model analysis"],
                cons=["To be determined by model analysis"]
            )
        ]

        # Add claims for each option
        claims_added = []
        for opt in self._options:
            claim = state.kb.add_claim(
                text=f"Option '{opt.name}': {opt.description}",
                owner=opt.proposer,
                confidence=0.5,
                round_num=1
            )
            claims_added.append(claim.id)

        return NodeResult(
            node_id="generate_options",
            status=NodeStatus.COMPLETED,
            output={
                'options_count': len(self._options),
                'options': [{'id': o.id, 'name': o.name, 'description': o.description}
                            for o in self._options]
            },
            claims_added=claims_added,
            questions_added=[q.id]
        )

    async def _node_rubric_scoring(self, state: WorkflowState) -> NodeResult:
        """
        Node 2: Score options against rubric criteria.

        In full implementation, would have models score each option.
        Currently uses placeholder scoring.
        """
        scored_options = []

        for opt in self._options:
            # Placeholder: assign neutral scores
            # In production, each criterion would be scored by models
            criterion_scores = {
                c.name: 0.5 for c in self.criteria
            }

            # Calculate weighted score
            total_score = sum(
                criterion_scores[c.name] * c.weight
                for c in self.criteria
            )
            opt.score = total_score

            scored_options.append({
                'id': opt.id,
                'name': opt.name,
                'scores': criterion_scores,
                'total_score': round(total_score, 3)
            })

        # Sort by score
        scored_options.sort(key=lambda x: x['total_score'], reverse=True)

        return NodeResult(
            node_id="rubric_scoring",
            status=NodeStatus.COMPLETED,
            output={
                'criteria': [{'name': c.name, 'weight': c.weight} for c in self.criteria],
                'scored_options': scored_options,
                'top_option': scored_options[0] if scored_options else None
            }
        )

    async def _node_red_team(self, state: WorkflowState) -> NodeResult:
        """
        Node 3: Red-team analysis of top options.

        Identifies risks, failure modes, and edge cases.
        """
        # Get top options from scoring
        scoring_output = state.get_node_output("rubric_scoring")
        if not scoring_output:
            return NodeResult(
                node_id="red_team",
                status=NodeStatus.FAILED,
                output={},
                error="No scoring results available"
            )

        top_options = scoring_output.get('scored_options', [])[:2]  # Top 2

        risks_by_option = {}
        claims_added = []

        for opt_data in top_options:
            opt_id = opt_data['id']

            # Placeholder risks (would come from red-team model query)
            risks = [
                {
                    'description': f"Potential risk for {opt_data['name']}",
                    'severity': 'medium',
                    'mitigation': 'To be determined'
                }
            ]

            risks_by_option[opt_id] = risks

            # Add risk as a claim
            claim = state.kb.add_claim(
                text=f"Risk identified for {opt_data['name']}: {risks[0]['description']}",
                owner="red_team",
                status=ClaimStatus.PROPOSED,
                confidence=0.6,
                round_num=1
            )
            claims_added.append(claim.id)

        return NodeResult(
            node_id="red_team",
            status=NodeStatus.COMPLETED,
            output={
                'options_analyzed': len(top_options),
                'risks_by_option': risks_by_option,
                'total_risks': sum(len(r) for r in risks_by_option.values())
            },
            claims_added=claims_added
        )

    async def _node_evidence_check(self, state: WorkflowState) -> NodeResult:
        """
        Node 4: Verify claims with evidence.

        Checks KB for unsupported claims and notes them.
        In full implementation, would trigger researcher.
        """
        unsupported = state.kb.get_unsupported_claims()
        coverage = state.kb.evidence_coverage()

        # Add open question if coverage is low
        questions_added = []
        if coverage < 0.5 and unsupported:
            q = state.kb.add_question(
                prompt=f"Evidence needed for {len(unsupported)} claims",
                owner="evidence_check",
                linked_claims=[c.id for c in unsupported[:5]],
                round_num=1
            )
            questions_added.append(q.id)

        return NodeResult(
            node_id="evidence_check",
            status=NodeStatus.COMPLETED,
            output={
                'evidence_coverage': round(coverage, 3),
                'unsupported_claims': len(unsupported),
                'note': "Evidence retrieval not yet implemented" if unsupported else "All claims supported"
            },
            questions_added=questions_added
        )

    async def _node_recommendation(self, state: WorkflowState) -> NodeResult:
        """
        Node 5: Generate final recommendation with tripwires.
        """
        # Gather inputs from prior nodes
        scoring = state.get_node_output("rubric_scoring") or {}
        red_team = state.get_node_output("red_team") or {}
        evidence = state.get_node_output("evidence_check") or {}

        top_option = scoring.get('top_option', {})
        risks = red_team.get('risks_by_option', {}).get(top_option.get('id'), [])

        # Build recommendation
        self._recommendation = {
            'recommended_option': top_option.get('name', 'Unable to determine'),
            'score': top_option.get('total_score', 0.0),
            'rationale': f"Based on rubric scoring across {len(self.criteria)} criteria",
            'tradeoffs': [
                f"Scored {top_option.get('total_score', 0):.0%} overall",
                f"{len(risks)} risk(s) identified"
            ],
            'tripwires': [
                "Revisit if implementation timeline exceeds estimates by 50%",
                "Revisit if key assumptions prove incorrect",
                "Revisit if new options become available"
            ],
            'confidence': min(0.9, top_option.get('total_score', 0.5) + 0.2),
            'evidence_coverage': evidence.get('evidence_coverage', 0.0)
        }

        # Add decision to KB
        decision = state.kb.add_decision(
            summary=f"Recommend: {self._recommendation['recommended_option']}",
            tradeoffs=self._recommendation['tradeoffs'],
            tripwires=self._recommendation['tripwires'],
            confidence=self._recommendation['confidence'],
            round_num=1
        )

        return NodeResult(
            node_id="recommendation",
            status=NodeStatus.COMPLETED,
            output=self._recommendation
        )

    def _build_final_output(self) -> Dict[str, Any]:
        """Build final decision output."""
        return {
            'success': True,
            'workflow': self.name,
            'recommendation': self._recommendation,
            'options': [
                {'id': o.id, 'name': o.name, 'score': o.score}
                for o in sorted(self._options, key=lambda x: x.score, reverse=True)
            ],
            'node_results': {
                node_id: result.to_dict()
                for node_id, result in self.state.node_results.items()
            },
            'kb_snapshot': self.state.kb.to_dict()
        }

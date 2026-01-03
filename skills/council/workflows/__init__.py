"""
Workflow Graphs - Structured deliberation flows for STORM pipeline.

Workflow graphs define node sequences for different query types:
- DecisionGraph: Options → Rubric → Red-team → Recommend
- ResearchGraph: Perspectives → Questions → Retrieve → Report
- CodeReviewGraph: Static scan → Threat model → Patches → Checklist
"""

from .base import WorkflowGraph, WorkflowNode, NodeResult, WorkflowState
from .decision import DecisionGraph
from .research import ResearchGraph
from .code_review import CodeReviewGraph

# Workflow registry
WORKFLOWS = {
    'decision': DecisionGraph,
    'research': ResearchGraph,
    'code_review': CodeReviewGraph,
}


def get_workflow(name: str) -> type:
    """Get workflow class by name."""
    if name not in WORKFLOWS:
        raise ValueError(f"Unknown workflow: {name}. Available: {list(WORKFLOWS.keys())}")
    return WORKFLOWS[name]


__all__ = [
    'WorkflowGraph', 'WorkflowNode', 'NodeResult', 'WorkflowState',
    'DecisionGraph', 'ResearchGraph', 'CodeReviewGraph',
    'WORKFLOWS', 'get_workflow'
]

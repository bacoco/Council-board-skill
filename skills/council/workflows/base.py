"""
Workflow Graph Base - Foundation for structured deliberation flows.

A workflow graph is a sequence of nodes, where each node:
1. Has a specific purpose (e.g., "generate options", "evaluate risks")
2. Receives the current state (query, KB, prior results)
3. Produces structured output that updates the state
4. May trigger model queries or other operations

The graph executor runs nodes in sequence, handling errors and
collecting results for the final synthesis.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_base import KnowledgeBase


class NodeStatus(Enum):
    """Status of a workflow node execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class NodeResult:
    """Result from executing a workflow node."""
    node_id: str
    status: NodeStatus
    output: Dict[str, Any]
    error: Optional[str] = None
    latency_ms: int = 0
    # Claims/sources/questions added during this node
    claims_added: List[str] = field(default_factory=list)
    sources_added: List[str] = field(default_factory=list)
    questions_added: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'status': self.status.value,
            'output': self.output,
            'error': self.error,
            'latency_ms': self.latency_ms,
            'claims_added': self.claims_added,
            'sources_added': self.sources_added,
            'questions_added': self.questions_added
        }


@dataclass
class WorkflowState:
    """
    Shared state passed through workflow nodes.

    Contains the query, context, KB, and accumulated results.
    """
    query: str
    context: str
    kb: KnowledgeBase
    models: List[str]
    chairman: str
    timeout: int
    # Accumulated outputs from each node
    node_results: Dict[str, NodeResult] = field(default_factory=dict)
    # Current phase/stage name
    current_phase: str = "init"
    # Any errors encountered
    errors: List[str] = field(default_factory=list)
    # Metadata for tracking
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_node_output(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get output from a previous node."""
        result = self.node_results.get(node_id)
        return result.output if result else None

    def add_error(self, error: str) -> None:
        """Record an error."""
        self.errors.append(error)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'query': self.query[:100],
            'current_phase': self.current_phase,
            'nodes_completed': list(self.node_results.keys()),
            'errors': self.errors,
            'kb_summary': {
                'claims': len(self.kb.claims),
                'sources': len(self.kb.sources),
                'questions': len(self.kb.open_questions)
            }
        }


@dataclass
class WorkflowNode:
    """
    A single node in a workflow graph.

    Nodes are the building blocks of workflows. Each node:
    - Has an ID and description
    - Defines what it needs from prior nodes (dependencies)
    - Executes some logic (model query, aggregation, etc.)
    - Updates the workflow state
    """
    id: str
    name: str
    description: str
    # IDs of nodes that must complete before this one
    dependencies: List[str] = field(default_factory=list)
    # Whether this node is optional (can be skipped on error)
    optional: bool = False
    # Custom execution function (set by subclasses)
    execute_fn: Optional[Callable] = None

    async def execute(self, state: WorkflowState) -> NodeResult:
        """
        Execute this node.

        Override in subclasses or provide execute_fn.
        """
        if self.execute_fn:
            return await self.execute_fn(state)

        # Default: no-op
        return NodeResult(
            node_id=self.id,
            status=NodeStatus.COMPLETED,
            output={'message': f'Node {self.id} executed (no-op)'}
        )


class WorkflowGraph(ABC):
    """
    Abstract base for workflow graphs.

    A workflow graph defines:
    - A sequence of nodes to execute
    - How to synthesize final output from node results
    - Error handling and recovery strategies
    """

    name: str = "base"
    description: str = "Base workflow graph"

    def __init__(self, state: WorkflowState):
        self.state = state
        self._nodes: List[WorkflowNode] = []
        self._build_nodes()

    @abstractmethod
    def _build_nodes(self) -> None:
        """Build the node sequence. Override in subclasses."""
        pass

    @property
    def nodes(self) -> List[WorkflowNode]:
        """Get the node sequence."""
        return self._nodes

    def add_node(self, node: WorkflowNode) -> None:
        """Add a node to the graph."""
        self._nodes.append(node)

    async def execute(self) -> Dict[str, Any]:
        """
        Execute the full workflow.

        Runs nodes in sequence, respecting dependencies.
        Returns final output dict.
        """
        for node in self._nodes:
            self.state.current_phase = node.name

            # Check dependencies
            for dep_id in node.dependencies:
                dep_result = self.state.node_results.get(dep_id)
                if not dep_result or dep_result.status != NodeStatus.COMPLETED:
                    if node.optional:
                        self.state.node_results[node.id] = NodeResult(
                            node_id=node.id,
                            status=NodeStatus.SKIPPED,
                            output={'reason': f'Dependency {dep_id} not satisfied'}
                        )
                        continue
                    else:
                        self.state.add_error(f"Node {node.id} failed: dependency {dep_id} not satisfied")
                        return self._build_error_output()

            # Execute node
            try:
                result = await node.execute(self.state)
                self.state.node_results[node.id] = result

                if result.status == NodeStatus.FAILED and not node.optional:
                    self.state.add_error(f"Node {node.id} failed: {result.error}")
                    return self._build_error_output()

            except Exception as e:
                if node.optional:
                    self.state.node_results[node.id] = NodeResult(
                        node_id=node.id,
                        status=NodeStatus.FAILED,
                        output={},
                        error=str(e)
                    )
                else:
                    self.state.add_error(f"Node {node.id} exception: {str(e)}")
                    return self._build_error_output()

        return self._build_final_output()

    @abstractmethod
    def _build_final_output(self) -> Dict[str, Any]:
        """Build final output from node results. Override in subclasses."""
        pass

    def _build_error_output(self) -> Dict[str, Any]:
        """Build error output when workflow fails."""
        return {
            'success': False,
            'errors': self.state.errors,
            'partial_results': {
                node_id: result.to_dict()
                for node_id, result in self.state.node_results.items()
            },
            'state': self.state.to_dict()
        }

    def get_progress(self) -> Dict[str, Any]:
        """Get current workflow progress."""
        completed = sum(
            1 for r in self.state.node_results.values()
            if r.status == NodeStatus.COMPLETED
        )
        return {
            'total_nodes': len(self._nodes),
            'completed_nodes': completed,
            'current_phase': self.state.current_phase,
            'progress_pct': (completed / len(self._nodes) * 100) if self._nodes else 0
        }

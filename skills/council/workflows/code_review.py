"""
Code Review Workflow Graph - Structured flow for code review queries.

Flow (from PRD):
1. Static Scan: Initial code analysis prompts
2. Threat Model: Security threat modeling
3. Performance/Maintainability: Quality analysis
4. Patch Suggestions: Concrete improvement suggestions
5. Final Checklist: Consolidated review checklist
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .base import WorkflowGraph, WorkflowNode, WorkflowState, NodeResult, NodeStatus
from knowledge_base import KnowledgeBase, ClaimStatus


class IssueSeverity(Enum):
    """Severity levels for code issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class IssueCategory(Enum):
    """Categories of code issues."""
    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    CORRECTNESS = "correctness"
    STYLE = "style"


@dataclass
class CodeIssue:
    """An issue found during code review."""
    id: str
    category: IssueCategory
    severity: IssueSeverity
    title: str
    description: str
    location: Optional[str] = None  # file:line if applicable
    suggestion: Optional[str] = None


@dataclass
class Threat:
    """A security threat identified during review."""
    id: str
    name: str
    description: str
    attack_vector: str
    impact: str
    mitigation: str
    severity: IssueSeverity = IssueSeverity.MEDIUM


class CodeReviewGraph(WorkflowGraph):
    """
    Workflow graph for code review queries.

    Best for questions like:
    - "Review this code for security issues"
    - "What's wrong with this implementation?"
    - "How can I improve this code?"
    """

    name = "code_review"
    description = "Code review workflow with security, performance, and quality analysis"

    def __init__(self, state: WorkflowState):
        self._issues: List[CodeIssue] = []
        self._threats: List[Threat] = []
        self._patches: List[Dict[str, str]] = []
        self._checklist: List[Dict[str, Any]] = []
        super().__init__(state)

    def _build_nodes(self) -> None:
        """Build the code review workflow nodes."""

        # Node 1: Static Scan
        self.add_node(WorkflowNode(
            id="static_scan",
            name="Static Analysis",
            description="Initial code analysis for obvious issues",
            execute_fn=self._node_static_scan
        ))

        # Node 2: Threat Model
        self.add_node(WorkflowNode(
            id="threat_model",
            name="Threat Modeling",
            description="Security threat analysis",
            dependencies=["static_scan"],
            execute_fn=self._node_threat_model
        ))

        # Node 3: Performance/Maintainability
        self.add_node(WorkflowNode(
            id="quality_analysis",
            name="Quality Analysis",
            description="Performance and maintainability review",
            dependencies=["static_scan"],
            execute_fn=self._node_quality_analysis
        ))

        # Node 4: Patch Suggestions
        self.add_node(WorkflowNode(
            id="patches",
            name="Patch Suggestions",
            description="Concrete code improvement suggestions",
            dependencies=["static_scan", "threat_model", "quality_analysis"],
            execute_fn=self._node_patches
        ))

        # Node 5: Final Checklist
        self.add_node(WorkflowNode(
            id="checklist",
            name="Review Checklist",
            description="Consolidated review checklist",
            dependencies=["patches"],
            execute_fn=self._node_checklist
        ))

    async def _node_static_scan(self, state: WorkflowState) -> NodeResult:
        """
        Node 1: Static code analysis.

        In full implementation, would analyze code context for issues.
        """
        # Check if we have code context
        has_code = bool(state.context and len(state.context) > 50)

        if not has_code:
            return NodeResult(
                node_id="static_scan",
                status=NodeStatus.COMPLETED,
                output={
                    'note': 'No code context provided. Analysis will be limited.',
                    'issues_found': 0,
                    'has_code': False
                }
            )

        # Placeholder issues (would come from actual code analysis)
        self._issues = [
            CodeIssue(
                id="issue_1",
                category=IssueCategory.SECURITY,
                severity=IssueSeverity.MEDIUM,
                title="Potential input validation issue",
                description="User input should be validated before use",
                suggestion="Add input validation/sanitization"
            ),
            CodeIssue(
                id="issue_2",
                category=IssueCategory.MAINTAINABILITY,
                severity=IssueSeverity.LOW,
                title="Complex function",
                description="Function could be split for better readability",
                suggestion="Consider extracting helper functions"
            )
        ]

        # Add issues as claims to KB
        claims_added = []
        for issue in self._issues:
            claim = state.kb.add_claim(
                text=f"[{issue.severity.value.upper()}] {issue.title}: {issue.description}",
                owner="static_scan",
                status=ClaimStatus.PROPOSED,
                confidence=0.7,
                round_num=1
            )
            claims_added.append(claim.id)

        return NodeResult(
            node_id="static_scan",
            status=NodeStatus.COMPLETED,
            output={
                'has_code': True,
                'issues_found': len(self._issues),
                'issues': [
                    {
                        'id': i.id,
                        'category': i.category.value,
                        'severity': i.severity.value,
                        'title': i.title
                    }
                    for i in self._issues
                ]
            },
            claims_added=claims_added
        )

    async def _node_threat_model(self, state: WorkflowState) -> NodeResult:
        """
        Node 2: Security threat modeling.

        Identifies potential security threats and attack vectors.
        """
        # Get static scan results
        scan_output = state.get_node_output("static_scan") or {}

        if not scan_output.get('has_code'):
            return NodeResult(
                node_id="threat_model",
                status=NodeStatus.COMPLETED,
                output={
                    'note': 'Limited threat modeling without code context',
                    'threats_found': 0
                }
            )

        # Placeholder threats (would come from security analysis)
        self._threats = [
            Threat(
                id="threat_1",
                name="Injection Attack",
                description="Potential for injection if input not sanitized",
                attack_vector="Malicious user input",
                impact="Data breach or system compromise",
                mitigation="Implement input validation and parameterized queries",
                severity=IssueSeverity.HIGH
            )
        ]

        # Add threats as claims
        claims_added = []
        for threat in self._threats:
            claim = state.kb.add_claim(
                text=f"THREAT: {threat.name} - {threat.description}. Mitigation: {threat.mitigation}",
                owner="threat_model",
                status=ClaimStatus.PROPOSED,
                confidence=0.6,
                round_num=1
            )
            claims_added.append(claim.id)

        return NodeResult(
            node_id="threat_model",
            status=NodeStatus.COMPLETED,
            output={
                'threats_found': len(self._threats),
                'threats': [
                    {
                        'id': t.id,
                        'name': t.name,
                        'severity': t.severity.value,
                        'mitigation': t.mitigation
                    }
                    for t in self._threats
                ],
                'high_severity_count': sum(1 for t in self._threats if t.severity in [IssueSeverity.CRITICAL, IssueSeverity.HIGH])
            },
            claims_added=claims_added
        )

    async def _node_quality_analysis(self, state: WorkflowState) -> NodeResult:
        """
        Node 3: Performance and maintainability analysis.
        """
        scan_output = state.get_node_output("static_scan") or {}

        # Filter for quality-related issues
        quality_issues = [
            i for i in self._issues
            if i.category in [IssueCategory.PERFORMANCE, IssueCategory.MAINTAINABILITY]
        ]

        # Placeholder quality metrics
        metrics = {
            'complexity': 'medium',  # Would be calculated from code
            'test_coverage': 'unknown',
            'documentation': 'sparse',
            'code_style': 'mostly_consistent'
        }

        return NodeResult(
            node_id="quality_analysis",
            status=NodeStatus.COMPLETED,
            output={
                'quality_issues': len(quality_issues),
                'metrics': metrics,
                'recommendations': [
                    "Consider adding unit tests",
                    "Add docstrings to public functions",
                    "Review function complexity"
                ]
            }
        )

    async def _node_patches(self, state: WorkflowState) -> NodeResult:
        """
        Node 4: Generate patch suggestions.

        Concrete code improvement suggestions.
        """
        # Collect all issues that have suggestions
        self._patches = []

        for issue in self._issues:
            if issue.suggestion:
                self._patches.append({
                    'issue_id': issue.id,
                    'title': issue.title,
                    'suggestion': issue.suggestion,
                    'priority': issue.severity.value
                })

        for threat in self._threats:
            self._patches.append({
                'issue_id': threat.id,
                'title': f"Security: {threat.name}",
                'suggestion': threat.mitigation,
                'priority': threat.severity.value
            })

        # Sort by priority
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3, 'info': 4}
        self._patches.sort(key=lambda x: priority_order.get(x['priority'], 5))

        return NodeResult(
            node_id="patches",
            status=NodeStatus.COMPLETED,
            output={
                'patches_count': len(self._patches),
                'patches': self._patches,
                'high_priority': sum(1 for p in self._patches if p['priority'] in ['critical', 'high'])
            }
        )

    async def _node_checklist(self, state: WorkflowState) -> NodeResult:
        """
        Node 5: Generate final review checklist.
        """
        # Build checklist from all findings
        self._checklist = []

        # Security items
        if self._threats:
            self._checklist.append({
                'category': 'Security',
                'items': [
                    {'text': f"Address: {t.name}", 'severity': t.severity.value, 'done': False}
                    for t in self._threats
                ]
            })

        # Quality items
        quality_output = state.get_node_output("quality_analysis") or {}
        recommendations = quality_output.get('recommendations', [])
        if recommendations:
            self._checklist.append({
                'category': 'Quality',
                'items': [
                    {'text': r, 'severity': 'medium', 'done': False}
                    for r in recommendations
                ]
            })

        # Code issues
        if self._issues:
            self._checklist.append({
                'category': 'Issues',
                'items': [
                    {'text': f"{i.title}: {i.suggestion or i.description}", 'severity': i.severity.value, 'done': False}
                    for i in self._issues
                ]
            })

        # Calculate summary stats
        total_items = sum(len(c['items']) for c in self._checklist)
        critical_items = sum(
            1 for c in self._checklist
            for item in c['items']
            if item['severity'] in ['critical', 'high']
        )

        # Add decision to KB
        state.kb.add_decision(
            summary=f"Code review completed: {total_items} items, {critical_items} high priority",
            tradeoffs=[f"Found {len(self._threats)} security threats", f"Found {len(self._issues)} code issues"],
            tripwires=["Re-review after implementing fixes", "Update when code changes significantly"],
            confidence=0.75,
            round_num=1
        )

        return NodeResult(
            node_id="checklist",
            status=NodeStatus.COMPLETED,
            output={
                'checklist': self._checklist,
                'total_items': total_items,
                'critical_items': critical_items,
                'categories': [c['category'] for c in self._checklist]
            }
        )

    def _build_final_output(self) -> Dict[str, Any]:
        """Build final code review output."""
        # Calculate overall assessment
        critical_count = sum(1 for t in self._threats if t.severity == IssueSeverity.CRITICAL)
        high_count = sum(1 for t in self._threats if t.severity == IssueSeverity.HIGH)
        high_count += sum(1 for i in self._issues if i.severity == IssueSeverity.HIGH)

        if critical_count > 0:
            overall = "CRITICAL - Immediate action required"
            confidence = 0.9
        elif high_count > 0:
            overall = "NEEDS WORK - High priority issues found"
            confidence = 0.8
        elif self._issues:
            overall = "ACCEPTABLE - Minor improvements suggested"
            confidence = 0.7
        else:
            overall = "GOOD - No significant issues found"
            confidence = 0.85

        return {
            'success': True,
            'workflow': self.name,
            'assessment': overall,
            'confidence': confidence,
            'summary': {
                'threats': len(self._threats),
                'issues': len(self._issues),
                'patches': len(self._patches)
            },
            'threats': [
                {
                    'name': t.name,
                    'severity': t.severity.value,
                    'mitigation': t.mitigation
                }
                for t in self._threats
            ],
            'issues': [
                {
                    'title': i.title,
                    'category': i.category.value,
                    'severity': i.severity.value,
                    'suggestion': i.suggestion
                }
                for i in self._issues
            ],
            'checklist': self._checklist,
            'node_results': {
                node_id: result.to_dict()
                for node_id, result in self.state.node_results.items()
            },
            'kb_snapshot': self.state.kb.to_dict()
        }

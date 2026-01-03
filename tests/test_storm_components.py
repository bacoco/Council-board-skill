"""
Unit tests for STORM pipeline components.

Tests cover:
- KnowledgeBase: Claims, sources, decisions, metrics
- ConvergenceDetector: Evidence-aware convergence
- EvidenceJudge: Claim evaluation
- Moderator: Workflow detection, shallow consensus
- WorkflowGraph: Base workflow execution
- Trail generation: STORM trail files
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

# Ensure council package is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "skills" / "council"))

from knowledge_base import KnowledgeBase, Claim, Source, ClaimStatus
from convergence import ConvergenceDetector, ConvergenceResult
from agents.evidence_judge import EvidenceJudge, EvidenceReport
from agents.moderator import Moderator, ModeratorAction, WorkflowType
from workflows.base import WorkflowGraph, WorkflowNode, NodeResult, WorkflowState
from core.storm_trail import generate_storm_trail_markdown, save_storm_trail_to_file


# =============================================================================
# KnowledgeBase Tests
# =============================================================================

class TestKnowledgeBase:
    """Tests for KnowledgeBase claim/source tracking."""

    def test_add_claim_generates_id(self):
        """Adding a claim should generate a unique ID."""
        kb = KnowledgeBase()
        claim = kb.add_claim(
            text="The sky is blue",
            owner="claude",
            confidence=0.9
        )
        assert claim.id.startswith("claim_")
        assert claim.text == "The sky is blue"
        assert claim.owner == "claude"
        assert claim.confidence == 0.9
        assert claim.status == ClaimStatus.PROPOSED

    def test_add_multiple_claims(self):
        """Multiple claims should have unique IDs."""
        kb = KnowledgeBase()
        claim1 = kb.add_claim("Claim 1", "claude", 0.8)
        claim2 = kb.add_claim("Claim 2", "gemini", 0.7)

        assert claim1.id != claim2.id
        assert len(kb.claims) == 2

    def test_add_source(self):
        """Adding a source should track URI and reliability."""
        kb = KnowledgeBase()
        source = kb.add_source(
            uri="https://example.com/docs",
            snippet="Relevant documentation excerpt",
            source_type="documentation",
            reliability=0.95,
            relevance=0.8
        )

        assert source.id.startswith("source_")
        assert source.uri == "https://example.com/docs"
        assert source.reliability == 0.95

    def test_link_evidence_to_claim(self):
        """Claims can be linked to supporting sources."""
        kb = KnowledgeBase()
        claim = kb.add_claim("Test claim", "claude", 0.8)
        source = kb.add_source("https://example.com", "Evidence snippet", "web", 0.9, 0.8)

        kb.link_evidence_to_claim(claim.id, source.id, supports=True)

        updated_claim = kb.get_claim(claim.id)
        assert source.id in updated_claim.support_evidence_ids

    def test_update_claim_status(self):
        """Claim status can be updated (proposed -> supported)."""
        kb = KnowledgeBase()
        claim = kb.add_claim("Test claim", "claude", 0.8)

        kb.update_claim_status(claim.id, ClaimStatus.SUPPORTED)

        updated = kb.get_claim(claim.id)
        assert updated.status == ClaimStatus.SUPPORTED

    def test_evidence_coverage_empty_kb(self):
        """Empty KB should have 100% coverage (nothing to cover)."""
        kb = KnowledgeBase()
        assert kb.evidence_coverage() == 1.0

    def test_evidence_coverage_with_sources(self):
        """Claims with sources should increase coverage."""
        kb = KnowledgeBase()
        claim1 = kb.add_claim("Claim 1", "claude", 0.8)
        claim2 = kb.add_claim("Claim 2", "gemini", 0.7)
        source = kb.add_source("https://example.com", "Evidence snippet", "web", 0.9, 0.8)

        kb.link_evidence_to_claim(claim1.id, source.id, supports=True)

        # 1 of 2 claims has evidence = 50%
        assert kb.evidence_coverage() == 0.5

    def test_source_diversity(self):
        """Source diversity should reflect unique source types."""
        kb = KnowledgeBase()
        kb.add_source("https://a.com", "Snippet 1", "web", 0.8, 0.8)
        kb.add_source("https://b.com", "Snippet 2", "docs", 0.9, 0.9)
        kb.add_source("code.py", "Snippet 3", "repo", 0.95, 0.7)

        # 3 unique types out of max 4 = 0.75
        diversity = kb.source_diversity()
        assert 0 < diversity <= 1.0

    def test_to_dict_serialization(self):
        """KB should serialize to dict for trail files."""
        kb = KnowledgeBase()
        kb.add_claim("Test claim", "claude", 0.8)
        kb.add_source("https://example.com", "Evidence snippet", "web", 0.9, 0.8)

        data = kb.to_dict()

        assert "claims" in data
        assert "sources" in data
        assert "metrics" in data
        assert len(data["claims"]) == 1
        assert len(data["sources"]) == 1


# =============================================================================
# ConvergenceDetector Tests
# =============================================================================

class TestConvergenceDetector:
    """Tests for classic and evidence-aware convergence."""

    def test_classic_convergence_high_confidence(self):
        """High confidence + convergence signals should trigger convergence."""
        detector = ConvergenceDetector(threshold=0.8, evidence_aware=False)

        round_outputs = [
            {"confidence": 0.9, "converged": True},
            {"confidence": 0.85, "converged": True}
        ]

        result = detector.check_classic(round_outputs)

        assert result.converged is True
        assert result.score > 0.8

    def test_classic_convergence_low_confidence(self):
        """Low confidence should prevent convergence."""
        detector = ConvergenceDetector(threshold=0.8, evidence_aware=False)

        round_outputs = [
            {"confidence": 0.4, "converged": False},
            {"confidence": 0.5, "converged": False}
        ]

        result = detector.check_classic(round_outputs)

        assert result.converged is False
        assert result.score < 0.8

    def test_evidence_aware_convergence(self):
        """Evidence-aware convergence should factor in KB metrics."""
        detector = ConvergenceDetector(threshold=0.8, evidence_aware=True)

        # Create KB with some evidence
        kb = KnowledgeBase()
        claim = kb.add_claim("Test claim", "claude", 0.85)
        source = kb.add_source("https://example.com", "Evidence snippet", "web", 0.9, 0.8)
        kb.link_evidence_to_claim(claim.id, source.id, supports=True)

        round_outputs = [
            {"confidence": 0.9, "converged": True}
        ]

        result = detector.check_evidence_aware(round_outputs, kb)

        assert result.converged is True
        assert "evidence" in result.components

    def test_evidence_aware_penalizes_low_coverage(self):
        """Low evidence coverage should reduce convergence score."""
        detector = ConvergenceDetector(threshold=0.8, evidence_aware=True)

        # KB with no evidence
        kb = KnowledgeBase()
        kb.add_claim("Unsupported claim", "claude", 0.9)

        round_outputs = [
            {"confidence": 0.9, "converged": True}
        ]

        result = detector.check_evidence_aware(round_outputs, kb)

        # Score should be lower due to 0% evidence coverage
        assert result.score < 0.9


# =============================================================================
# EvidenceJudge Tests
# =============================================================================

class TestEvidenceJudge:
    """Tests for EvidenceJudge claim evaluation."""

    def test_evaluate_empty_kb(self):
        """Evaluating empty KB should return empty report."""
        kb = KnowledgeBase()
        judge = EvidenceJudge(kb)

        async def _run():
            return await judge.evaluate_all_claims()

        report = asyncio.run(_run())

        assert report.overall_coverage == 0.0
        assert len(report.unsupported_claims) == 0

    def test_evaluate_supported_claim(self):
        """Claim with source should be marked as supported."""
        kb = KnowledgeBase()
        claim = kb.add_claim("Supported claim", "claude", 0.8)
        source = kb.add_source("https://example.com", "Evidence snippet", "web", 0.9, 0.9)
        kb.link_evidence_to_claim(claim.id, source.id, supports=True)

        judge = EvidenceJudge(kb)

        async def _run():
            return await judge.evaluate_all_claims()

        report = asyncio.run(_run())

        assert report.overall_coverage == 1.0
        assert claim.id not in report.unsupported_claims

    def test_evaluate_unsupported_claim(self):
        """Claim without source should be marked unsupported."""
        kb = KnowledgeBase()
        claim = kb.add_claim("Unsupported claim", "claude", 0.8)

        judge = EvidenceJudge(kb)

        async def _run():
            return await judge.evaluate_all_claims()

        report = asyncio.run(_run())

        assert report.overall_coverage == 0.0
        assert claim.id in report.unsupported_claims

    def test_confidence_adjustment(self):
        """Evidence judge should adjust confidence based on coverage."""
        kb = KnowledgeBase()
        claim = kb.add_claim("Test claim", "claude", 0.8)
        source = kb.add_source("https://example.com", "Evidence snippet", "web", 0.9, 0.9)
        kb.link_evidence_to_claim(claim.id, source.id, supports=True)

        judge = EvidenceJudge(kb)

        # With full coverage, confidence should increase
        adjusted = judge.calculate_evidence_adjusted_confidence(0.7)
        assert adjusted >= 0.7  # Should be same or higher


# =============================================================================
# Moderator Tests
# =============================================================================

class TestModerator:
    """Tests for Moderator workflow detection and routing."""

    def test_detect_decision_workflow(self):
        """Should detect decision workflow for 'should we' questions."""
        kb = KnowledgeBase()
        moderator = Moderator(kb)

        workflow = moderator.detect_workflow(
            query="Should we use microservices or a monolith?",
            context=""
        )

        assert workflow == WorkflowType.DECISION

    def test_detect_research_workflow(self):
        """Should detect research workflow for 'what/how/explain' questions."""
        kb = KnowledgeBase()
        moderator = Moderator(kb)

        workflow = moderator.detect_workflow(
            query="What are the best practices for API design?",
            context=""
        )

        assert workflow == WorkflowType.RESEARCH

    def test_detect_code_review_workflow(self):
        """Should detect code review workflow for code context."""
        kb = KnowledgeBase()
        moderator = Moderator(kb)

        workflow = moderator.detect_workflow(
            query="Review this code for security issues",
            context="def process_input(user_data): exec(user_data)"
        )

        assert workflow == WorkflowType.CODE_REVIEW

    def test_analyze_round_detects_shallow_consensus(self):
        """Should detect shallow consensus when agreement is high but coverage low."""
        kb = KnowledgeBase()
        # Add claim without evidence
        kb.add_claim("Agreed but unsupported", "claude", 0.9)

        moderator = Moderator(kb)

        round_outputs = [
            {"confidence": 0.9, "converged": True, "agrees_with_consensus": True},
            {"confidence": 0.88, "converged": True, "agrees_with_consensus": True}
        ]

        decision = moderator.analyze_round(
            round_num=1,
            round_outputs=round_outputs,
            max_rounds=3
        )

        assert decision.shallow_consensus_detected is True
        assert decision.action == ModeratorAction.RETRIEVE

    def test_analyze_round_finalizes_with_evidence(self):
        """Should finalize when evidence coverage is high."""
        kb = KnowledgeBase()
        claim = kb.add_claim("Supported claim", "claude", 0.9)
        source = kb.add_source("https://example.com", "Evidence snippet", "web", 0.95, 0.9)
        kb.link_evidence_to_claim(claim.id, source.id, supports=True)

        moderator = Moderator(kb)

        round_outputs = [
            {"confidence": 0.9, "converged": True}
        ]

        decision = moderator.analyze_round(
            round_num=3,
            round_outputs=round_outputs,
            max_rounds=3
        )

        assert decision.action == ModeratorAction.FINALIZE


# =============================================================================
# WorkflowGraph Tests
# =============================================================================

class TestWorkflowGraph:
    """Tests for workflow graph execution."""

    def test_workflow_state_initialization(self):
        """WorkflowState should initialize with query and KB."""
        kb = KnowledgeBase()
        state = WorkflowState(
            query="Test query",
            context="Some context",
            kb=kb,
            models=["claude", "gemini"],
            chairman="claude",
            timeout=60
        )

        assert state.query == "Test query"
        assert state.kb is kb
        assert len(state.node_results) == 0

    def test_node_result_creation(self):
        """NodeResult should capture execution details."""
        result = NodeResult(
            node_id="test_node",
            status="completed",
            output={"answer": "test"},
            latency_ms=100
        )

        assert result.node_id == "test_node"
        assert result.status == "completed"
        assert result.latency_ms == 100


# =============================================================================
# Trail Generation Tests
# =============================================================================

class TestStormTrail:
    """Tests for STORM trail file generation."""

    def test_generate_trail_markdown(self):
        """Trail generation should produce valid markdown."""
        markdown = generate_storm_trail_markdown(
            session_id="test_session_123",
            query="Should we use microservices?",
            mode="storm_decision",
            workflow_name="DecisionGraph",
            node_results={
                "generate_options": {
                    "status": "completed",
                    "latency_ms": 1500,
                    "output": {"options": [{"name": "Microservices"}]}
                }
            },
            kb_snapshot={
                "claims": [{"text": "Microservices scale better", "status": "proposed", "owner": "claude", "confidence": 0.8}],
                "sources": [],
                "metrics": {"evidence_coverage": 0.0, "unresolved_objections": 0, "source_diversity": 0.0}
            },
            moderator_history=[],
            convergence={"converged": True, "score": 0.85, "threshold": 0.8, "components": {}, "confidence_rationale": "High agreement"},
            evidence_report={"overall_coverage": 0.0, "unsupported_claims": [], "contradicted_claims": [], "evaluation_notes": []},
            final_answer="Recommend microservices for this use case.",
            confidence=0.85,
            duration_ms=5000,
            models=["claude", "gemini"],
            context_preview="Architecture decision context..."
        )

        assert "# STORM Deliberation Trail" in markdown
        assert "test_session_123" in markdown
        assert "Should we use microservices?" in markdown
        assert "DecisionGraph" in markdown
        assert "generate_options" in markdown
        assert "Recommend microservices" in markdown

    def test_trail_markdown_includes_kb_snapshot(self):
        """Trail should include Knowledge Base claims and metrics."""
        markdown = generate_storm_trail_markdown(
            session_id="test_123",
            query="Test query",
            mode="storm_decision",
            workflow_name="DecisionGraph",
            node_results={},
            kb_snapshot={
                "claims": [
                    {"text": "Claim 1", "status": "verified", "owner": "claude", "confidence": 0.9}
                ],
                "sources": [
                    {"id": "src_1", "uri": "https://example.com", "reliability": 0.95, "relevance": 0.8}
                ],
                "metrics": {"evidence_coverage": 0.5, "unresolved_objections": 1, "source_diversity": 0.33}
            },
            moderator_history=[],
            convergence={"converged": True, "score": 0.82, "threshold": 0.8, "components": {}, "confidence_rationale": "OK"},
            evidence_report={"overall_coverage": 0.5, "unsupported_claims": [], "contradicted_claims": [], "evaluation_notes": []},
            final_answer="Test answer",
            confidence=0.82,
            duration_ms=1000,
            models=["claude"],
            context_preview=""
        )

        assert "Knowledge Base Snapshot" in markdown
        assert "Evidence Coverage" in markdown
        assert "Claim 1" in markdown

    def test_save_trail_to_file(self, tmp_path):
        """Trail file should be saved with correct naming."""
        markdown = "# Test Trail\n\nContent here"

        filepath = save_storm_trail_to_file(
            markdown_content=markdown,
            session_id="test_123",
            query="Should we migrate to cloud?",
            mode="storm_decision",
            output_dir=str(tmp_path)
        )

        assert filepath.exists()
        assert filepath.suffix == ".md"
        assert "storm_" in filepath.name
        assert "storm_decision" in filepath.name

        content = filepath.read_text()
        assert content == markdown


# =============================================================================
# Integration Test: Pipeline Flow
# =============================================================================

class TestStormPipelineFlow:
    """Integration tests for STORM pipeline components working together."""

    def test_full_decision_flow(self):
        """Test KB -> Moderator -> EvidenceJudge -> Convergence flow."""
        # Initialize components
        kb = KnowledgeBase()
        moderator = Moderator(kb)
        judge = EvidenceJudge(kb)
        detector = ConvergenceDetector(threshold=0.8, evidence_aware=True)

        # Step 1: Detect workflow
        workflow = moderator.detect_workflow(
            "Should we use Redis or PostgreSQL for caching?",
            ""
        )
        assert workflow == WorkflowType.DECISION

        # Step 2: Add claims from "deliberation"
        claim1 = kb.add_claim(
            "Redis is faster for in-memory operations",
            "claude",
            0.9
        )
        claim2 = kb.add_claim(
            "PostgreSQL provides better durability",
            "gemini",
            0.85
        )

        # Step 3: Add evidence
        source = kb.add_source(
            "https://redis.io/docs/performance",
            "Redis performance benchmarks show sub-millisecond latency",
            "docs",
            0.95,
            0.9
        )
        kb.link_evidence_to_claim(claim1.id, source.id, supports=True)

        # Step 4: Evaluate evidence
        async def _evaluate():
            return await judge.evaluate_all_claims()

        report = asyncio.run(_evaluate())

        # One claim supported, one not
        assert report.overall_coverage == 0.5
        assert claim2.id in report.unsupported_claims

        # Step 5: Check convergence
        round_outputs = [
            {"confidence": 0.9, "converged": True},
            {"confidence": 0.85, "converged": True}
        ]

        result = detector.check_evidence_aware(round_outputs, kb)

        # Should converge but with note about evidence gaps
        assert "evidence" in result.components

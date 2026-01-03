"""
Evidence Judge Agent - Scores claims against evidence.

The Evidence Judge:
1. Evaluates each claim against available evidence sources
2. Produces Claim → Evidence → Confidence table
3. Flags unsupported and contradicted claims
4. Calculates evidence-based confidence adjustments

Implementation: Uses text-matching heuristics for claim-evidence relevance.
Future: Can be enhanced with LLM-based semantic matching.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_base import KnowledgeBase, Claim, Source, ClaimStatus


# Common stop words to filter from term extraction
_STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'this', 'that',
    'these', 'those', 'it', 'its', 'they', 'them', 'their', 'we', 'our', 'you',
    'your', 'he', 'she', 'him', 'her', 'his', 'who', 'which', 'what', 'when',
    'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
    'most', 'other', 'some', 'such', 'not', 'only', 'own', 'same', 'than',
    'too', 'very', 'just', 'also', 'now', 'here', 'there', 'then', 'once'
}


class EvidenceRelation(Enum):
    """How evidence relates to a claim."""
    SUPPORTS = "supports"           # Evidence confirms the claim
    CONTRADICTS = "contradicts"     # Evidence refutes the claim
    PARTIAL = "partial"             # Mixed or incomplete support
    IRRELEVANT = "irrelevant"       # Evidence doesn't apply
    INSUFFICIENT = "insufficient"   # Not enough to judge


@dataclass
class ClaimEvidenceLink:
    """Link between a claim and a piece of evidence."""
    claim_id: str
    source_id: str
    relation: EvidenceRelation
    confidence: float  # 0.0-1.0 how confident in this judgment
    rationale: str     # Brief explanation

    def to_dict(self) -> Dict[str, Any]:
        return {
            'claim_id': self.claim_id,
            'source_id': self.source_id,
            'relation': self.relation.value,
            'confidence': round(self.confidence, 3),
            'rationale': self.rationale
        }


@dataclass
class ClaimVerdict:
    """Verdict for a single claim after evidence evaluation."""
    claim_id: str
    claim_text: str
    original_confidence: float
    adjusted_confidence: float
    status: ClaimStatus
    evidence_links: List[ClaimEvidenceLink]
    summary: str  # Human-readable verdict

    def to_dict(self) -> Dict[str, Any]:
        return {
            'claim_id': self.claim_id,
            'claim_text': self.claim_text[:100] + '...' if len(self.claim_text) > 100 else self.claim_text,
            'original_confidence': round(self.original_confidence, 3),
            'adjusted_confidence': round(self.adjusted_confidence, 3),
            'status': self.status.value,
            'evidence_links': [l.to_dict() for l in self.evidence_links],
            'summary': self.summary
        }


@dataclass
class EvidenceReport:
    """Full evidence evaluation report."""
    verdicts: List[ClaimVerdict]
    overall_coverage: float
    unsupported_claims: List[str]
    contradicted_claims: List[str]
    source_diversity: float
    evaluation_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'verdicts': [v.to_dict() for v in self.verdicts],
            'overall_coverage': round(self.overall_coverage, 3),
            'unsupported_claims': self.unsupported_claims,
            'contradicted_claims': self.contradicted_claims,
            'source_diversity': round(self.source_diversity, 3),
            'evaluation_notes': self.evaluation_notes
        }

    def summary_table(self) -> str:
        """Generate markdown table of claim verdicts."""
        lines = ["| Claim | Status | Confidence | Evidence |",
                 "|-------|--------|------------|----------|"]
        for v in self.verdicts:
            claim_short = v.claim_text[:40] + '...' if len(v.claim_text) > 40 else v.claim_text
            evidence_count = len(v.evidence_links)
            lines.append(
                f"| {claim_short} | {v.status.value} | "
                f"{v.adjusted_confidence:.0%} | {evidence_count} sources |"
            )
        return '\n'.join(lines)


class EvidenceJudge:
    """
    Evaluates claims against evidence sources.

    Uses text-matching heuristics to determine if evidence supports claims:
    - Extracts key terms from claims
    - Checks for term presence in evidence snippets
    - Scores relevance based on overlap
    - Adjusts confidence based on evidence quality
    """

    # Confidence adjustment factors
    SUPPORT_BOOST = 0.15       # Boost for each supporting source
    CONTRADICTION_PENALTY = 0.25  # Penalty for contradicting evidence
    NO_EVIDENCE_PENALTY = 0.1  # Penalty for no evidence at all

    def __init__(self, kb: KnowledgeBase):
        """
        Initialize Evidence Judge.

        Args:
            kb: KnowledgeBase with claims and sources to evaluate
        """
        self.kb = kb

    async def evaluate_all_claims(self) -> EvidenceReport:
        """
        Evaluate all claims in the KB against available evidence.

        Returns:
            EvidenceReport with verdicts for all claims
        """
        verdicts = []
        unsupported = []
        contradicted = []

        for claim in self.kb.claims:
            verdict = await self._evaluate_claim(claim)
            verdicts.append(verdict)

            if verdict.status == ClaimStatus.PROPOSED and not claim.support_evidence_ids:
                unsupported.append(claim.id)
            elif verdict.status == ClaimStatus.CONTRADICTED:
                contradicted.append(claim.id)

        return EvidenceReport(
            verdicts=verdicts,
            overall_coverage=self.kb.evidence_coverage(),
            unsupported_claims=unsupported,
            contradicted_claims=contradicted,
            source_diversity=self.kb.source_diversity(),
            evaluation_notes=self._generate_notes(verdicts)
        )

    async def evaluate_claims(self, claim_ids: List[str]) -> EvidenceReport:
        """
        Evaluate specific claims.

        Args:
            claim_ids: List of claim IDs to evaluate

        Returns:
            EvidenceReport for the specified claims
        """
        verdicts = []
        unsupported = []
        contradicted = []

        for claim_id in claim_ids:
            claim = self.kb.get_claim(claim_id)
            if not claim:
                continue

            verdict = await self._evaluate_claim(claim)
            verdicts.append(verdict)

            if verdict.status == ClaimStatus.PROPOSED and not claim.support_evidence_ids:
                unsupported.append(claim.id)
            elif verdict.status == ClaimStatus.CONTRADICTED:
                contradicted.append(claim.id)

        return EvidenceReport(
            verdicts=verdicts,
            overall_coverage=self.kb.evidence_coverage(),
            unsupported_claims=unsupported,
            contradicted_claims=contradicted,
            source_diversity=self.kb.source_diversity()
        )

    async def _evaluate_claim(self, claim: Claim) -> ClaimVerdict:
        """
        Evaluate a single claim against evidence.

        Uses text-matching heuristics:
        1. Extract key terms from claim
        2. Check for term overlap with evidence snippets
        3. Score relevance based on matches
        4. Adjust confidence accordingly
        """
        evidence_links = []
        adjusted_confidence = claim.confidence

        # Extract key terms from claim for matching
        claim_terms = self._extract_terms(claim.text)

        # Process supporting evidence with relevance scoring
        for source_id in claim.support_evidence_ids:
            source = self.kb.get_source(source_id)
            if source:
                # Calculate relevance based on term overlap
                relevance = self._calculate_relevance(claim_terms, source.snippet)
                relation = EvidenceRelation.SUPPORTS if relevance > 0.3 else EvidenceRelation.PARTIAL

                evidence_links.append(ClaimEvidenceLink(
                    claim_id=claim.id,
                    source_id=source_id,
                    relation=relation,
                    confidence=source.reliability * relevance,
                    rationale=f"Term overlap: {relevance:.0%} with {source.source_type} source"
                ))

                # Boost confidence based on relevance and source reliability
                boost = self.SUPPORT_BOOST * relevance * source.reliability
                adjusted_confidence = min(1.0, adjusted_confidence + boost)

        # Process contradicting evidence
        for source_id in claim.contradict_evidence_ids:
            source = self.kb.get_source(source_id)
            if source:
                relevance = self._calculate_relevance(claim_terms, source.snippet)
                evidence_links.append(ClaimEvidenceLink(
                    claim_id=claim.id,
                    source_id=source_id,
                    relation=EvidenceRelation.CONTRADICTS,
                    confidence=source.reliability * relevance,
                    rationale=f"Contradicting evidence from {source.source_type}"
                ))
                # Penalty scales with relevance
                penalty = self.CONTRADICTION_PENALTY * relevance
                adjusted_confidence = max(0.0, adjusted_confidence - penalty)

        # Penalty for no evidence
        if not evidence_links:
            adjusted_confidence = max(0.0, adjusted_confidence - self.NO_EVIDENCE_PENALTY)

        # Determine status
        if claim.contradict_evidence_ids:
            status = ClaimStatus.CONTRADICTED
            summary = f"Claim contradicted by {len(claim.contradict_evidence_ids)} source(s)"
        elif claim.support_evidence_ids:
            avg_relevance = sum(l.confidence for l in evidence_links) / len(evidence_links)
            status = ClaimStatus.SUPPORTED
            summary = f"Claim supported by {len(claim.support_evidence_ids)} source(s), avg relevance {avg_relevance:.0%}"
        else:
            status = claim.status  # Keep original (likely PROPOSED)
            summary = "No evidence linked to this claim"

        return ClaimVerdict(
            claim_id=claim.id,
            claim_text=claim.text,
            original_confidence=claim.confidence,
            adjusted_confidence=adjusted_confidence,
            status=status,
            evidence_links=evidence_links,
            summary=summary
        )

    def _extract_terms(self, text: str) -> set:
        """Extract significant terms from text for matching."""
        # Extract words, filter stop words and short terms
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_-]*\b', text.lower())
        return {w for w in words if w not in _STOP_WORDS and len(w) > 2}

    def _calculate_relevance(self, claim_terms: set, snippet: str) -> float:
        """Calculate relevance score between claim terms and evidence snippet."""
        if not claim_terms or not snippet:
            return 0.0

        snippet_terms = self._extract_terms(snippet)
        if not snippet_terms:
            return 0.0

        # Calculate Jaccard-like overlap
        overlap = claim_terms & snippet_terms
        if not overlap:
            return 0.0

        # Score: overlap / claim terms (how much of claim is covered)
        coverage = len(overlap) / len(claim_terms)

        # Bonus for high overlap
        if coverage > 0.5:
            coverage = min(1.0, coverage * 1.2)

        return coverage

    def _generate_notes(self, verdicts: List[ClaimVerdict]) -> List[str]:
        """Generate evaluation notes based on verdicts."""
        notes = []

        unsupported_count = sum(1 for v in verdicts if not v.evidence_links)
        if unsupported_count > 0:
            notes.append(f"{unsupported_count} claim(s) have no linked evidence")

        contradicted_count = sum(1 for v in verdicts if v.status == ClaimStatus.CONTRADICTED)
        if contradicted_count > 0:
            notes.append(f"{contradicted_count} claim(s) are contradicted by evidence")

        low_confidence = [v for v in verdicts if v.adjusted_confidence < 0.3]
        if low_confidence:
            notes.append(f"{len(low_confidence)} claim(s) have low confidence (<30%)")

        return notes

    def calculate_evidence_adjusted_confidence(self, base_confidence: float) -> float:
        """
        Calculate overall confidence adjusted for evidence coverage.

        This blends:
        - Base confidence (from agreement)
        - Evidence coverage
        - Unresolved objections
        - Source diversity

        Returns:
            Adjusted confidence 0.0-1.0
        """
        coverage = self.kb.evidence_coverage()
        objections = self.kb.unresolved_objections_count()
        diversity = self.kb.source_diversity()

        # Weights from PRD suggestion (to be tuned)
        w_agreement = 0.4
        w_coverage = 0.3
        w_objections = 0.2
        w_diversity = 0.1

        # Objection penalty: each unresolved objection reduces score
        objection_penalty = min(objections * 0.1, 0.3)

        adjusted = (
            w_agreement * base_confidence +
            w_coverage * coverage +
            w_objections * (1.0 - objection_penalty) +
            w_diversity * diversity
        )

        return max(0.0, min(1.0, adjusted))

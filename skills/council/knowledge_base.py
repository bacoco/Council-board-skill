"""
KnowledgeBase - Shared knowledge artifact for STORM pipeline.

The KnowledgeBase tracks all claims, evidence, open questions, and decisions
across deliberation rounds. Every agent turn must add/update the KB.

Data Model (from PRD):
- Concept: Definitions and relationships
- Claim: Assertions with provenance and support status
- Source: Evidence with reliability and relevance scores
- OpenQuestion: Unresolved issues tracked through deliberation
- Decision: Final recommendations with tradeoffs and tripwires
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import json


class ClaimStatus(Enum):
    """Status of a claim based on evidence evaluation."""
    PROPOSED = "proposed"       # Initial assertion, not yet evaluated
    SUPPORTED = "supported"     # Has supporting evidence
    CONTRADICTED = "contradicted"  # Evidence contradicts the claim
    SPECULATIVE = "speculative"  # Explicitly marked as uncertain
    WITHDRAWN = "withdrawn"     # Owner retracted the claim


class QuestionStatus(Enum):
    """Status of an open question."""
    OPEN = "open"           # Needs investigation
    RESOLVED = "resolved"   # Answered satisfactorily
    DEFERRED = "deferred"   # Acknowledged but out of scope


@dataclass
class Concept:
    """A concept or term with definition and relationships."""
    id: str
    title: str
    definition: str
    related_concepts: List[str] = field(default_factory=list)
    added_by: Optional[str] = None  # Persona/model that introduced it
    round_added: int = 1


@dataclass
class Source:
    """An evidence source with metadata."""
    id: str
    uri: str  # URL, file path, or reference
    snippet: str  # Relevant excerpt
    retrieved_at: str  # ISO timestamp
    reliability: float = 0.5  # 0.0-1.0, default neutral
    relevance: float = 0.5   # 0.0-1.0, default neutral
    source_type: str = "unknown"  # 'web', 'repo', 'docs', 'citation'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'uri': self.uri,
            'snippet': self.snippet[:200] + '...' if len(self.snippet) > 200 else self.snippet,
            'retrieved_at': self.retrieved_at,
            'reliability': self.reliability,
            'relevance': self.relevance,
            'source_type': self.source_type
        }


@dataclass
class Claim:
    """An assertion made during deliberation."""
    id: str
    text: str
    owner: str  # Persona/model that made the claim
    status: ClaimStatus = ClaimStatus.PROPOSED
    confidence: float = 0.5  # Owner's confidence in the claim
    support_evidence_ids: List[str] = field(default_factory=list)
    contradict_evidence_ids: List[str] = field(default_factory=list)
    round_added: int = 1
    round_updated: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'text': self.text,
            'owner': self.owner,
            'status': self.status.value,
            'confidence': self.confidence,
            'support_evidence': self.support_evidence_ids,
            'contradict_evidence': self.contradict_evidence_ids,
            'round_added': self.round_added
        }


@dataclass
class OpenQuestion:
    """An unresolved question requiring investigation."""
    id: str
    prompt: str  # The question itself
    owner: str   # Who raised it
    status: QuestionStatus = QuestionStatus.OPEN
    linked_claim_ids: List[str] = field(default_factory=list)
    resolution: Optional[str] = None
    round_added: int = 1
    round_resolved: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'prompt': self.prompt,
            'owner': self.owner,
            'status': self.status.value,
            'linked_claims': self.linked_claim_ids,
            'resolution': self.resolution
        }


@dataclass
class Decision:
    """A recommendation with tradeoffs and conditions to revisit."""
    id: str
    summary: str
    tradeoffs: List[str]
    tripwires: List[str]  # Conditions that should trigger revisiting
    confidence: float
    supporting_claim_ids: List[str] = field(default_factory=list)
    dissenting_views: List[str] = field(default_factory=list)
    round_finalized: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'summary': self.summary,
            'tradeoffs': self.tradeoffs,
            'tripwires': self.tripwires,
            'confidence': self.confidence,
            'supporting_claims': self.supporting_claim_ids,
            'dissenting_views': self.dissenting_views
        }


@dataclass
class KnowledgeBase:
    """
    Shared knowledge artifact for STORM deliberation.

    Tracks all concepts, claims, sources, questions, and decisions.
    Provides methods for querying and updating the knowledge state.
    """
    concepts: List[Concept] = field(default_factory=list)
    claims: List[Claim] = field(default_factory=list)
    sources: List[Source] = field(default_factory=list)
    open_questions: List[OpenQuestion] = field(default_factory=list)
    decisions: List[Decision] = field(default_factory=list)

    # Counters for generating unique IDs
    _next_ids: Dict[str, int] = field(default_factory=lambda: {
        'concept': 1, 'claim': 1, 'source': 1, 'question': 1, 'decision': 1
    })

    def _gen_id(self, prefix: str) -> str:
        """Generate a unique ID for an entity."""
        id_num = self._next_ids.get(prefix, 1)
        self._next_ids[prefix] = id_num + 1
        return f"{prefix}_{id_num:03d}"

    # =========================================================================
    # Add methods
    # =========================================================================

    def add_concept(self, title: str, definition: str, owner: str = "unknown",
                    related: List[str] = None, round_num: int = 1) -> Concept:
        """Add a new concept to the KB."""
        concept = Concept(
            id=self._gen_id('concept'),
            title=title,
            definition=definition,
            related_concepts=related or [],
            added_by=owner,
            round_added=round_num
        )
        self.concepts.append(concept)
        return concept

    def add_claim(self, text: str, owner: str, confidence: float = 0.5,
                  status: ClaimStatus = ClaimStatus.PROPOSED,
                  round_num: int = 1) -> Claim:
        """Add a new claim to the KB."""
        claim = Claim(
            id=self._gen_id('claim'),
            text=text,
            owner=owner,
            status=status,
            confidence=confidence,
            round_added=round_num
        )
        self.claims.append(claim)
        return claim

    def add_source(self, uri: str, snippet: str, source_type: str = "unknown",
                   reliability: float = 0.5, relevance: float = 0.5) -> Source:
        """Add a new evidence source to the KB."""
        source = Source(
            id=self._gen_id('source'),
            uri=uri,
            snippet=snippet,
            retrieved_at=datetime.utcnow().isoformat(),
            reliability=reliability,
            relevance=relevance,
            source_type=source_type
        )
        self.sources.append(source)
        return source

    def add_question(self, prompt: str, owner: str,
                     linked_claims: List[str] = None,
                     round_num: int = 1) -> OpenQuestion:
        """Add an open question to the KB."""
        question = OpenQuestion(
            id=self._gen_id('question'),
            prompt=prompt,
            owner=owner,
            linked_claim_ids=linked_claims or [],
            round_added=round_num
        )
        self.open_questions.append(question)
        return question

    def add_decision(self, summary: str, tradeoffs: List[str],
                     tripwires: List[str], confidence: float,
                     supporting_claims: List[str] = None,
                     dissenting_views: List[str] = None,
                     round_num: int = 1) -> Decision:
        """Add a final decision to the KB."""
        decision = Decision(
            id=self._gen_id('decision'),
            summary=summary,
            tradeoffs=tradeoffs,
            tripwires=tripwires,
            confidence=confidence,
            supporting_claim_ids=supporting_claims or [],
            dissenting_views=dissenting_views or [],
            round_finalized=round_num
        )
        self.decisions.append(decision)
        return decision

    # =========================================================================
    # Query methods
    # =========================================================================

    def get_claim(self, claim_id: str) -> Optional[Claim]:
        """Get a claim by ID."""
        return next((c for c in self.claims if c.id == claim_id), None)

    def get_source(self, source_id: str) -> Optional[Source]:
        """Get a source by ID."""
        return next((s for s in self.sources if s.id == source_id), None)

    def get_unsupported_claims(self) -> List[Claim]:
        """Get claims that lack supporting evidence."""
        return [c for c in self.claims
                if c.status == ClaimStatus.PROPOSED and not c.support_evidence_ids]

    def get_contradicted_claims(self) -> List[Claim]:
        """Get claims that have contradicting evidence."""
        return [c for c in self.claims if c.status == ClaimStatus.CONTRADICTED]

    def get_open_questions(self) -> List[OpenQuestion]:
        """Get all unresolved questions."""
        return [q for q in self.open_questions if q.status == QuestionStatus.OPEN]

    def claims_by_owner(self, owner: str) -> List[Claim]:
        """Get all claims made by a specific owner."""
        return [c for c in self.claims if c.owner == owner]

    # =========================================================================
    # Update methods
    # =========================================================================

    def link_evidence_to_claim(self, claim_id: str, source_id: str,
                               supports: bool = True, round_num: int = 1) -> bool:
        """Link a source to a claim as supporting or contradicting evidence."""
        claim = self.get_claim(claim_id)
        if not claim:
            return False

        if supports:
            if source_id not in claim.support_evidence_ids:
                claim.support_evidence_ids.append(source_id)
                if claim.status == ClaimStatus.PROPOSED:
                    claim.status = ClaimStatus.SUPPORTED
        else:
            if source_id not in claim.contradict_evidence_ids:
                claim.contradict_evidence_ids.append(source_id)
                claim.status = ClaimStatus.CONTRADICTED

        claim.round_updated = round_num
        return True

    def resolve_question(self, question_id: str, resolution: str,
                         round_num: int = 1) -> bool:
        """Mark a question as resolved."""
        question = next((q for q in self.open_questions if q.id == question_id), None)
        if not question:
            return False

        question.status = QuestionStatus.RESOLVED
        question.resolution = resolution
        question.round_resolved = round_num
        return True

    def update_claim_status(self, claim_id: str, status: ClaimStatus,
                            round_num: int = 1) -> bool:
        """Update the status of a claim."""
        claim = self.get_claim(claim_id)
        if not claim:
            return False
        claim.status = status
        claim.round_updated = round_num
        return True

    # =========================================================================
    # Metrics
    # =========================================================================

    def evidence_coverage(self) -> float:
        """
        Calculate the fraction of claims with at least one evidence source.

        Returns:
            Float 0.0-1.0 representing evidence coverage
        """
        if not self.claims:
            return 1.0  # No claims = nothing to cover

        supported = sum(1 for c in self.claims if c.support_evidence_ids)
        return supported / len(self.claims)

    def unresolved_objections_count(self) -> int:
        """Count claims that are contradicted but not resolved."""
        return len([c for c in self.claims if c.status == ClaimStatus.CONTRADICTED])

    def source_diversity(self) -> float:
        """
        Calculate source type diversity (0.0-1.0).

        Higher = more diverse source types used.
        """
        if not self.sources:
            return 0.0

        source_types = set(s.source_type for s in self.sources)
        # Normalize: assume max 4 source types (web, repo, docs, citation)
        return min(len(source_types) / 4.0, 1.0)

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize KB to dictionary for trail output."""
        return {
            'concepts': [{'id': c.id, 'title': c.title, 'definition': c.definition}
                         for c in self.concepts],
            'claims': [c.to_dict() for c in self.claims],
            'sources': [s.to_dict() for s in self.sources],
            'open_questions': [q.to_dict() for q in self.open_questions],
            'decisions': [d.to_dict() for d in self.decisions],
            'metrics': {
                'evidence_coverage': round(self.evidence_coverage(), 3),
                'unresolved_objections': self.unresolved_objections_count(),
                'source_diversity': round(self.source_diversity(), 3)
            }
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize KB to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeBase':
        """Deserialize KB from dictionary."""
        kb = cls()

        for c in data.get('concepts', []):
            kb.concepts.append(Concept(
                id=c['id'],
                title=c['title'],
                definition=c['definition'],
                related_concepts=c.get('related_concepts', []),
                added_by=c.get('added_by'),
                round_added=c.get('round_added', 1)
            ))

        for c in data.get('claims', []):
            kb.claims.append(Claim(
                id=c['id'],
                text=c['text'],
                owner=c['owner'],
                status=ClaimStatus(c.get('status', 'proposed')),
                confidence=c.get('confidence', 0.5),
                support_evidence_ids=c.get('support_evidence', []),
                contradict_evidence_ids=c.get('contradict_evidence', []),
                round_added=c.get('round_added', 1)
            ))

        for s in data.get('sources', []):
            kb.sources.append(Source(
                id=s['id'],
                uri=s['uri'],
                snippet=s['snippet'],
                retrieved_at=s.get('retrieved_at', ''),
                reliability=s.get('reliability', 0.5),
                relevance=s.get('relevance', 0.5),
                source_type=s.get('source_type', 'unknown')
            ))

        for q in data.get('open_questions', []):
            kb.open_questions.append(OpenQuestion(
                id=q['id'],
                prompt=q['prompt'],
                owner=q['owner'],
                status=QuestionStatus(q.get('status', 'open')),
                linked_claim_ids=q.get('linked_claims', []),
                resolution=q.get('resolution')
            ))

        for d in data.get('decisions', []):
            kb.decisions.append(Decision(
                id=d['id'],
                summary=d['summary'],
                tradeoffs=d.get('tradeoffs', []),
                tripwires=d.get('tripwires', []),
                confidence=d.get('confidence', 0.5),
                supporting_claim_ids=d.get('supporting_claims', []),
                dissenting_views=d.get('dissenting_views', [])
            ))

        return kb

    def summary(self) -> str:
        """Generate a human-readable summary of KB state."""
        lines = [
            f"KnowledgeBase Summary:",
            f"  Concepts: {len(self.concepts)}",
            f"  Claims: {len(self.claims)} ({len(self.get_unsupported_claims())} unsupported)",
            f"  Sources: {len(self.sources)}",
            f"  Open Questions: {len(self.get_open_questions())}",
            f"  Decisions: {len(self.decisions)}",
            f"  Evidence Coverage: {self.evidence_coverage():.1%}",
        ]
        return '\n'.join(lines)

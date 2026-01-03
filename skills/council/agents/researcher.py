"""
Researcher Agent - Retrieves evidence for disputed claims.

The Researcher:
1. Takes claims/questions that need evidence
2. Orchestrates retrieval from available sources (web, repo, docs)
3. Clusters retrieved evidence by claim
4. Updates the KnowledgeBase with new sources

Status: STUB - Core interface defined, retrieval logic to be implemented.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_base import KnowledgeBase, Source


@dataclass
class RetrievalRequest:
    """Request to retrieve evidence for a claim or question."""
    target_id: str        # claim_id or question_id
    target_text: str      # The claim/question text to find evidence for
    source_types: List[str] = None  # ['web', 'repo', 'docs'] - None = all
    max_sources: int = 3  # Max sources to retrieve per target


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""
    target_id: str
    sources_found: List[Source]
    success: bool
    error: Optional[str] = None
    latency_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'target_id': self.target_id,
            'sources_found': [s.to_dict() for s in self.sources_found],
            'success': self.success,
            'error': self.error,
            'latency_ms': self.latency_ms
        }


class Researcher:
    """
    Retrieves evidence from various sources.

    NOTE: This is a stub implementation. The actual retrieval logic
    (web search, repo search, doc search) needs to be implemented.
    Currently returns empty results with a notice.
    """

    # Available source types
    SOURCE_TYPES = ['web', 'repo', 'docs', 'citation']

    def __init__(self, kb: KnowledgeBase, allowed_sources: List[str] = None):
        """
        Initialize Researcher.

        Args:
            kb: KnowledgeBase to update with found sources
            allowed_sources: Which source types to use (default: all)
        """
        self.kb = kb
        self.allowed_sources = allowed_sources or self.SOURCE_TYPES
        self._retrieval_count = 0

    async def retrieve_for_claims(self, claim_ids: List[str],
                                  max_sources_per_claim: int = 3) -> List[RetrievalResult]:
        """
        Retrieve evidence for multiple claims.

        Args:
            claim_ids: List of claim IDs to find evidence for
            max_sources_per_claim: Max sources per claim

        Returns:
            List of RetrievalResult for each claim
        """
        results = []

        for claim_id in claim_ids:
            claim = self.kb.get_claim(claim_id)
            if not claim:
                results.append(RetrievalResult(
                    target_id=claim_id,
                    sources_found=[],
                    success=False,
                    error=f"Claim {claim_id} not found in KB"
                ))
                continue

            result = await self._retrieve_for_text(
                target_id=claim_id,
                text=claim.text,
                max_sources=max_sources_per_claim
            )
            results.append(result)

        return results

    async def retrieve_for_questions(self, question_ids: List[str],
                                     max_sources_per_question: int = 3) -> List[RetrievalResult]:
        """
        Retrieve evidence for open questions.

        Args:
            question_ids: List of question IDs to research
            max_sources_per_question: Max sources per question

        Returns:
            List of RetrievalResult for each question
        """
        results = []

        for q_id in question_ids:
            question = next(
                (q for q in self.kb.open_questions if q.id == q_id),
                None
            )
            if not question:
                results.append(RetrievalResult(
                    target_id=q_id,
                    sources_found=[],
                    success=False,
                    error=f"Question {q_id} not found in KB"
                ))
                continue

            result = await self._retrieve_for_text(
                target_id=q_id,
                text=question.prompt,
                max_sources=max_sources_per_question
            )
            results.append(result)

        return results

    async def _retrieve_for_text(self, target_id: str, text: str,
                                 max_sources: int) -> RetrievalResult:
        """
        Core retrieval logic for a piece of text.

        TODO: Implement actual retrieval:
        - Web search via existing adapters
        - Repo search via grep/glob
        - Doc search via embeddings or keyword matching

        Currently returns stub result.
        """
        self._retrieval_count += 1

        # STUB: Return empty result with notice
        # In full implementation, this would:
        # 1. Extract key terms from text
        # 2. Query each allowed source type
        # 3. Rank and dedupe results
        # 4. Add to KB and link to target

        return RetrievalResult(
            target_id=target_id,
            sources_found=[],
            success=True,  # No error, just no implementation
            error=None,
            latency_ms=0
        )

    async def retrieve_batch(self, requests: List[RetrievalRequest]) -> List[RetrievalResult]:
        """
        Batch retrieval for multiple targets.

        Args:
            requests: List of RetrievalRequest objects

        Returns:
            List of RetrievalResult objects
        """
        results = []
        for req in requests:
            result = await self._retrieve_for_text(
                target_id=req.target_id,
                text=req.target_text,
                max_sources=req.max_sources
            )
            results.append(result)
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        return {
            'retrieval_count': self._retrieval_count,
            'allowed_sources': self.allowed_sources,
            'sources_in_kb': len(self.kb.sources)
        }

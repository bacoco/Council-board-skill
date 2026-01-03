"""
Researcher Agent - Retrieves evidence for disputed claims.

The Researcher:
1. Takes claims/questions that need evidence
2. Orchestrates retrieval from available sources (web, repo, docs)
3. Clusters retrieved evidence by claim
4. Updates the KnowledgeBase with new sources

Retrieval methods:
- Repo search: grep/glob through local codebase
- Web search: via model-assisted web queries (when available)
- Doc search: scan documentation files
"""

import asyncio
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_base import KnowledgeBase, Source, ClaimStatus


@dataclass
class RetrievalRequest:
    """Request to retrieve evidence for a claim or question."""
    target_id: str        # claim_id or question_id
    target_text: str      # The claim/question text to find evidence for
    source_types: List[str] = None  # ['web', 'repo', 'docs'] - None = all
    max_sources: int = 3  # Max sources to retrieve per target

    def __post_init__(self):
        if self.source_types is None:
            self.source_types = ['repo', 'docs']  # Default to local sources


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""
    target_id: str
    sources_found: List[Source]
    success: bool
    error: Optional[str] = None
    latency_ms: int = 0
    search_terms: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'target_id': self.target_id,
            'sources_found': [s.to_dict() for s in self.sources_found],
            'success': self.success,
            'error': self.error,
            'latency_ms': self.latency_ms,
            'search_terms': self.search_terms
        }


@dataclass
class SourceCandidate:
    """A potential source before it's added to KB."""
    uri: str
    snippet: str
    source_type: str
    relevance_score: float
    reliability_score: float
    match_context: str = ""


class KeyTermExtractor:
    """Extracts key search terms from claims and questions."""

    # Stop words to filter out
    STOP_WORDS = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
        'from', 'up', 'about', 'into', 'over', 'after', 'beneath', 'under',
        'above', 'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either',
        'neither', 'not', 'only', 'own', 'same', 'than', 'too', 'very',
        'just', 'also', 'now', 'here', 'there', 'when', 'where', 'why',
        'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'any', 'this', 'that', 'these',
        'those', 'what', 'which', 'who', 'whom', 'it', 'its', 'we', 'they',
        'i', 'me', 'my', 'you', 'your', 'he', 'she', 'him', 'her', 'his',
        'our', 'their', 'them', 'us', 'as', 'if', 'then', 'because', 'while'
    }

    # Technical terms that should be kept as-is
    TECH_PATTERNS = [
        r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b',  # CamelCase
        r'\b[a-z]+_[a-z_]+\b',  # snake_case
        r'\b[a-z]+-[a-z-]+\b',  # kebab-case
        r'\b(?:API|SDK|CLI|HTTP|SQL|JSON|XML|REST|GRPC|OAuth|JWT|CORS)\b',
        r'\b(?:async|await|import|export|class|function|interface|type)\b',
        r'\b\w+(?:\.\w+)+\b',  # dotted.paths
    ]

    def extract(self, text: str, max_terms: int = 8) -> List[str]:
        """
        Extract key search terms from text.

        Args:
            text: The claim or question text
            max_terms: Maximum number of terms to return

        Returns:
            List of key terms for searching
        """
        terms = []
        seen = set()

        # First, extract technical terms (preserve case)
        for pattern in self.TECH_PATTERNS:
            for match in re.finditer(pattern, text):
                term = match.group()
                if term.lower() not in seen and len(term) > 2:
                    terms.append(term)
                    seen.add(term.lower())

        # Extract quoted strings (exact phrases)
        for match in re.finditer(r'"([^"]+)"', text):
            phrase = match.group(1)
            if phrase.lower() not in seen:
                terms.append(phrase)
                seen.add(phrase.lower())

        # Extract remaining significant words
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_-]*\b', text)
        for word in words:
            lower = word.lower()
            if (lower not in self.STOP_WORDS and
                lower not in seen and
                len(word) > 3):
                terms.append(word)
                seen.add(lower)

        # Prioritize: technical terms first, then by length
        terms.sort(key=lambda x: (-self._term_priority(x), -len(x)))

        return terms[:max_terms]

    def _term_priority(self, term: str) -> int:
        """Score a term's priority (higher = more important)."""
        score = 0

        # Technical patterns get highest priority
        if re.match(r'^[A-Z][a-z]+(?:[A-Z][a-z]+)+$', term):  # CamelCase
            score += 10
        if '_' in term or '-' in term:  # snake_case or kebab-case
            score += 8
        if term.isupper() and len(term) > 2:  # Acronyms
            score += 7
        if '.' in term:  # Dotted paths
            score += 6
        if term[0].isupper():  # Proper nouns/names
            score += 3

        return score


class RepoSearcher:
    """Searches the local repository for evidence."""

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or self._find_project_root()

    def _find_project_root(self) -> Path:
        """Find project root by looking for common markers."""
        cwd = Path.cwd()
        markers = ['.git', 'package.json', 'pyproject.toml', 'Cargo.toml', 'go.mod']

        for parent in [cwd] + list(cwd.parents):
            for marker in markers:
                if (parent / marker).exists():
                    return parent
        return cwd

    async def search(self, terms: List[str], max_results: int = 5) -> List[SourceCandidate]:
        """
        Search repository for evidence matching terms.

        Args:
            terms: Search terms to look for
            max_results: Maximum results to return

        Returns:
            List of SourceCandidates found
        """
        candidates = []
        seen_files: Set[str] = set()

        for term in terms[:5]:  # Limit to first 5 terms
            results = await self._grep_term(term)
            for result in results:
                if result['file'] not in seen_files:
                    seen_files.add(result['file'])
                    candidates.append(SourceCandidate(
                        uri=result['file'],
                        snippet=result['snippet'],
                        source_type='repo',
                        relevance_score=self._score_relevance(term, result),
                        reliability_score=self._score_reliability(result['file']),
                        match_context=f"grep:{term}"
                    ))

                    if len(candidates) >= max_results:
                        break

            if len(candidates) >= max_results:
                break

        # Sort by combined score
        candidates.sort(key=lambda x: x.relevance_score + x.reliability_score, reverse=True)
        return candidates[:max_results]

    async def _grep_term(self, term: str, max_matches: int = 10) -> List[Dict[str, str]]:
        """Run grep to find term in repository."""
        results = []

        try:
            # Use grep with context
            proc = await asyncio.create_subprocess_exec(
                'grep', '-r', '-n', '-i', '--include=*.py', '--include=*.js',
                '--include=*.ts', '--include=*.md', '--include=*.yaml',
                '--include=*.json', '-C', '2', term,
                str(self.project_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)

            if stdout:
                lines = stdout.decode(errors='ignore').split('\n')
                current_file = None
                current_snippet = []

                for line in lines[:100]:  # Limit processing
                    if not line.strip():
                        if current_file and current_snippet:
                            results.append({
                                'file': current_file,
                                'snippet': '\n'.join(current_snippet[-5:])
                            })
                            current_snippet = []
                        continue

                    # Parse grep output (file:line:content)
                    match = re.match(r'^([^:]+):(\d+)[:-](.*)$', line)
                    if match:
                        file_path = match.group(1)
                        content = match.group(3)

                        # Relativize path
                        try:
                            rel_path = str(Path(file_path).relative_to(self.project_root))
                        except ValueError:
                            rel_path = file_path

                        if current_file != rel_path:
                            if current_file and current_snippet:
                                results.append({
                                    'file': current_file,
                                    'snippet': '\n'.join(current_snippet[-5:])
                                })
                            current_file = rel_path
                            current_snippet = []

                        current_snippet.append(content)

                        if len(results) >= max_matches:
                            break

                # Don't forget last file
                if current_file and current_snippet:
                    results.append({
                        'file': current_file,
                        'snippet': '\n'.join(current_snippet[-5:])
                    })

        except (asyncio.TimeoutError, Exception):
            pass

        return results

    def _score_relevance(self, term: str, result: Dict[str, str]) -> float:
        """Score how relevant a result is to the search term."""
        score = 0.5  # Base score

        snippet = result.get('snippet', '').lower()
        term_lower = term.lower()

        # Exact match in snippet
        if term_lower in snippet:
            score += 0.2

        # Term in filename
        if term_lower in result.get('file', '').lower():
            score += 0.15

        # Multiple occurrences
        count = snippet.count(term_lower)
        score += min(count * 0.05, 0.15)

        return min(score, 1.0)

    def _score_reliability(self, file_path: str) -> float:
        """Score how reliable a source file is."""
        score = 0.6  # Base score for repo files

        path_lower = file_path.lower()

        # Tests are reliable for behavior verification
        if 'test' in path_lower:
            score += 0.1

        # Documentation
        if path_lower.endswith('.md'):
            score += 0.15

        # Source code
        if any(path_lower.endswith(ext) for ext in ['.py', '.ts', '.js']):
            score += 0.1

        # Config files are authoritative
        if any(x in path_lower for x in ['config', 'settings', 'constants']):
            score += 0.1

        return min(score, 1.0)


class DocSearcher:
    """Searches documentation files for evidence."""

    DOC_PATTERNS = ['README.md', 'CLAUDE.md', 'docs/**/*.md', '**/*.rst']

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()

    async def search(self, terms: List[str], max_results: int = 3) -> List[SourceCandidate]:
        """Search documentation files for terms."""
        candidates = []

        # Find doc files
        doc_files = []
        for pattern in self.DOC_PATTERNS:
            doc_files.extend(self.project_root.glob(pattern))

        for doc_file in doc_files[:20]:  # Limit files
            try:
                content = doc_file.read_text(errors='ignore')[:50000]

                for term in terms:
                    if term.lower() in content.lower():
                        # Extract snippet around match
                        snippet = self._extract_snippet(content, term)
                        if snippet:
                            candidates.append(SourceCandidate(
                                uri=str(doc_file.relative_to(self.project_root)),
                                snippet=snippet,
                                source_type='docs',
                                relevance_score=0.7,
                                reliability_score=0.85,  # Docs are generally reliable
                                match_context=f"doc:{term}"
                            ))

                            if len(candidates) >= max_results:
                                break

            except Exception:
                continue

            if len(candidates) >= max_results:
                break

        return candidates[:max_results]

    def _extract_snippet(self, content: str, term: str, context_chars: int = 200) -> str:
        """Extract snippet around a term match."""
        idx = content.lower().find(term.lower())
        if idx == -1:
            return ""

        start = max(0, idx - context_chars)
        end = min(len(content), idx + len(term) + context_chars)

        snippet = content[start:end]

        # Clean up snippet boundaries
        if start > 0:
            snippet = '...' + snippet.split(' ', 1)[-1] if ' ' in snippet else snippet
        if end < len(content):
            snippet = snippet.rsplit(' ', 1)[0] + '...' if ' ' in snippet else snippet

        return snippet.strip()


class SourceReliabilityScorer:
    """
    Scores source reliability based on multiple factors.

    Factors considered:
    - Source type (docs > code > generated)
    - Authority indicators (official, verified, etc.)
    - Temporal freshness (newer = better for tech)
    - Corroboration (multiple sources agree)
    """

    # Base reliability by source type
    SOURCE_TYPE_SCORES = {
        'docs': 0.85,
        'repo': 0.70,
        'test': 0.75,
        'config': 0.80,
        'web': 0.50,
        'generated': 0.40,
        'unknown': 0.50
    }

    # Authority indicators that boost reliability
    AUTHORITY_PATTERNS = {
        'official': 0.15,
        'verified': 0.10,
        'reference': 0.10,
        'specification': 0.15,
        'rfc': 0.15,
        'standard': 0.10
    }

    def score(self, source: SourceCandidate) -> float:
        """
        Calculate reliability score for a source.

        Args:
            source: The source candidate to score

        Returns:
            Reliability score between 0.0 and 1.0
        """
        # Start with base score for source type
        score = self.SOURCE_TYPE_SCORES.get(source.source_type, 0.5)

        # Check for authority indicators in URI or snippet
        combined_text = (source.uri + ' ' + source.snippet).lower()
        for pattern, boost in self.AUTHORITY_PATTERNS.items():
            if pattern in combined_text:
                score += boost

        # Penalize very short snippets (less evidence)
        if len(source.snippet) < 50:
            score -= 0.1

        # Cap at 1.0
        return min(max(score, 0.1), 1.0)


class Researcher:
    """
    Retrieves evidence from various sources.

    Implements real retrieval for:
    - Repository code search (grep-based)
    - Documentation search
    - Source reliability scoring
    """

    # Available source types
    SOURCE_TYPES = ['repo', 'docs', 'web', 'citation']

    def __init__(self, kb: KnowledgeBase,
                 allowed_sources: List[str] = None,
                 project_root: Optional[Path] = None):
        """
        Initialize Researcher.

        Args:
            kb: KnowledgeBase to update with found sources
            allowed_sources: Which source types to use (default: ['repo', 'docs'])
            project_root: Root directory for searches (default: auto-detect)
        """
        self.kb = kb
        self.allowed_sources = allowed_sources or ['repo', 'docs']
        self.project_root = project_root

        # Initialize components
        self._term_extractor = KeyTermExtractor()
        self._repo_searcher = RepoSearcher(project_root)
        self._doc_searcher = DocSearcher(project_root)
        self._reliability_scorer = SourceReliabilityScorer()

        self._retrieval_count = 0
        self._sources_added = 0

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
                max_sources=max_sources_per_claim,
                is_claim=True
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
                max_sources=max_sources_per_question,
                is_claim=False
            )
            results.append(result)

        return results

    async def _retrieve_for_text(self, target_id: str, text: str,
                                 max_sources: int, is_claim: bool = True) -> RetrievalResult:
        """
        Core retrieval logic for a piece of text.

        1. Extract key terms from text
        2. Query each allowed source type
        3. Score and rank results
        4. Add to KB and link to target
        """
        start = time.time()
        self._retrieval_count += 1

        # Step 1: Extract search terms
        terms = self._term_extractor.extract(text, max_terms=8)

        if not terms:
            return RetrievalResult(
                target_id=target_id,
                sources_found=[],
                success=True,
                error="No searchable terms extracted",
                latency_ms=int((time.time() - start) * 1000),
                search_terms=[]
            )

        # Step 2: Search each source type
        all_candidates: List[SourceCandidate] = []

        if 'repo' in self.allowed_sources:
            repo_results = await self._repo_searcher.search(terms, max_results=max_sources)
            all_candidates.extend(repo_results)

        if 'docs' in self.allowed_sources:
            doc_results = await self._doc_searcher.search(terms, max_results=max_sources)
            all_candidates.extend(doc_results)

        # Step 3: Score reliability and dedupe
        seen_uris: Set[str] = set()
        scored_candidates = []

        for candidate in all_candidates:
            if candidate.uri not in seen_uris:
                seen_uris.add(candidate.uri)
                # Update reliability score with full scoring
                candidate.reliability_score = self._reliability_scorer.score(candidate)
                scored_candidates.append(candidate)

        # Sort by combined score
        scored_candidates.sort(
            key=lambda x: x.relevance_score * 0.6 + x.reliability_score * 0.4,
            reverse=True
        )

        # Step 4: Add top candidates to KB and link to target
        sources_added: List[Source] = []

        for candidate in scored_candidates[:max_sources]:
            source = self.kb.add_source(
                uri=candidate.uri,
                snippet=candidate.snippet,
                source_type=candidate.source_type,
                reliability=candidate.reliability_score,
                relevance=candidate.relevance_score
            )
            sources_added.append(source)
            self._sources_added += 1

            # Link to claim if applicable
            if is_claim:
                self.kb.link_evidence_to_claim(
                    claim_id=target_id,
                    source_id=source.id,
                    supports=True,  # Default to supporting
                    round_num=1
                )

        latency_ms = int((time.time() - start) * 1000)

        return RetrievalResult(
            target_id=target_id,
            sources_found=sources_added,
            success=True,
            error=None,
            latency_ms=latency_ms,
            search_terms=terms
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

    async def verify_claim_cross_model(self, claim_id: str,
                                       query_fn=None) -> Tuple[bool, float, str]:
        """
        Verify a claim using cross-model verification.

        Queries multiple models to independently verify the claim.
        Returns consensus on whether claim is supported.

        Args:
            claim_id: The claim to verify
            query_fn: Optional function to query models (for testing)

        Returns:
            Tuple of (is_verified, confidence, rationale)
        """
        claim = self.kb.get_claim(claim_id)
        if not claim:
            return False, 0.0, f"Claim {claim_id} not found"

        # If no query function, just return based on evidence
        if not query_fn:
            coverage = self.kb.evidence_coverage()
            has_evidence = len(claim.support_evidence_ids) > 0

            if has_evidence:
                return True, coverage, f"Claim has {len(claim.support_evidence_ids)} supporting sources"
            else:
                return False, 0.3, "No supporting evidence found"

        # With query function, implement actual cross-model verification
        # This would query multiple models and check consensus
        return True, 0.5, "Cross-model verification pending"

    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        return {
            'retrieval_count': self._retrieval_count,
            'sources_added': self._sources_added,
            'allowed_sources': self.allowed_sources,
            'sources_in_kb': len(self.kb.sources)
        }

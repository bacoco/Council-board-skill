"""
Semantic Embeddings and Similarity Module.

Provides embedding-based semantic similarity for evidence matching.
Uses a lightweight approach that works with CLI-based models.

Two modes:
1. LLM-judged similarity: Ask model to score semantic similarity (accurate but slow)
2. Local embeddings: Use sentence-transformers if available (fast but requires install)
"""

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from functools import lru_cache

# Try to import sentence-transformers for local embeddings
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None
    np = None


@dataclass
class SimilarityResult:
    """Result of semantic similarity comparison."""
    score: float  # 0.0-1.0
    method: str   # 'embedding', 'llm', 'term_overlap'
    cached: bool
    details: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'score': round(self.score, 4),
            'method': self.method,
            'cached': self.cached,
            'details': self.details
        }


class EmbeddingCache:
    """Thread-safe cache for embeddings and similarity scores."""

    def __init__(self, max_size: int = 1000):
        self._embeddings: Dict[str, List[float]] = {}
        self._similarities: Dict[str, float] = {}
        self._max_size = max_size

    def _text_hash(self, text: str) -> str:
        """Create hash key for text."""
        return hashlib.md5(text.encode()).hexdigest()[:16]

    def _pair_hash(self, text1: str, text2: str) -> str:
        """Create hash key for text pair (order-independent)."""
        h1, h2 = self._text_hash(text1), self._text_hash(text2)
        return f"{min(h1, h2)}_{max(h1, h2)}"

    def get_embedding(self, text: str) -> Optional[List[float]]:
        return self._embeddings.get(self._text_hash(text))

    def set_embedding(self, text: str, embedding: List[float]) -> None:
        if len(self._embeddings) >= self._max_size:
            # Simple eviction: remove oldest
            oldest_key = next(iter(self._embeddings))
            del self._embeddings[oldest_key]
        self._embeddings[self._text_hash(text)] = embedding

    def get_similarity(self, text1: str, text2: str) -> Optional[float]:
        return self._similarities.get(self._pair_hash(text1, text2))

    def set_similarity(self, text1: str, text2: str, score: float) -> None:
        if len(self._similarities) >= self._max_size:
            oldest_key = next(iter(self._similarities))
            del self._similarities[oldest_key]
        self._similarities[self._pair_hash(text1, text2)] = score

    def clear(self) -> None:
        self._embeddings.clear()
        self._similarities.clear()

    def stats(self) -> Dict[str, int]:
        return {
            'embeddings_cached': len(self._embeddings),
            'similarities_cached': len(self._similarities),
            'max_size': self._max_size
        }


# Global cache instance
_CACHE = EmbeddingCache()


class SemanticMatcher:
    """
    Semantic similarity matcher using embeddings or LLM-judged similarity.

    Provides accurate semantic matching for evidence-claim relevance,
    replacing simple term-overlap heuristics.
    """

    # Prompt for LLM-based similarity judgment
    SIMILARITY_PROMPT = """Rate the semantic similarity between these two texts on a scale of 0.0 to 1.0.

TEXT A:
{text_a}

TEXT B:
{text_b}

Consider:
- Do they discuss the same concepts/topics?
- Is one evidence for/against the other?
- Do they share key technical terms or ideas?

Respond with ONLY a JSON object: {{"score": 0.XX, "reason": "brief explanation"}}"""

    def __init__(self,
                 use_local_embeddings: bool = True,
                 model_name: str = 'all-MiniLM-L6-v2',
                 llm_adapter: Optional[callable] = None):
        """
        Initialize semantic matcher.

        Args:
            use_local_embeddings: If True and sentence-transformers available, use local embeddings
            model_name: Sentence transformer model name
            llm_adapter: Async function to query LLM for similarity (fallback)
        """
        self._model = None
        self._use_local = use_local_embeddings and SENTENCE_TRANSFORMERS_AVAILABLE
        self._model_name = model_name
        self._llm_adapter = llm_adapter
        self._initialized = False

    def _ensure_initialized(self) -> bool:
        """Lazy initialization of embedding model."""
        if self._initialized:
            return self._model is not None

        self._initialized = True

        if self._use_local and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self._model = SentenceTransformer(self._model_name)
                return True
            except Exception:
                self._model = None
                return False
        return False

    def _compute_embedding(self, text: str) -> Optional[List[float]]:
        """Compute embedding for text using local model."""
        if not self._ensure_initialized():
            return None

        # Check cache first
        cached = _CACHE.get_embedding(text)
        if cached is not None:
            return cached

        try:
            # Truncate long texts
            text_truncated = text[:512] if len(text) > 512 else text
            embedding = self._model.encode(text_truncated, convert_to_numpy=True)
            embedding_list = embedding.tolist()
            _CACHE.set_embedding(text, embedding_list)
            return embedding_list
        except Exception:
            return None

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if np is None:
            # Fallback without numpy
            dot = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot / (norm1 * norm2)

        v1, v2 = np.array(vec1), np.array(vec2)
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def compute_similarity_sync(self, text1: str, text2: str) -> SimilarityResult:
        """
        Compute semantic similarity synchronously using local embeddings.

        Falls back to term overlap if embeddings unavailable.
        """
        # Check cache
        cached_score = _CACHE.get_similarity(text1, text2)
        if cached_score is not None:
            return SimilarityResult(
                score=cached_score,
                method='embedding',
                cached=True
            )

        # Try local embeddings
        emb1 = self._compute_embedding(text1)
        emb2 = self._compute_embedding(text2)

        if emb1 is not None and emb2 is not None:
            score = self._cosine_similarity(emb1, emb2)
            # Normalize to 0-1 range (cosine can be negative)
            score = (score + 1) / 2
            _CACHE.set_similarity(text1, text2, score)
            return SimilarityResult(
                score=score,
                method='embedding',
                cached=False,
                details=f'model={self._model_name}'
            )

        # Fallback to term overlap
        return self._term_overlap_similarity(text1, text2)

    async def compute_similarity(self, text1: str, text2: str) -> SimilarityResult:
        """
        Compute semantic similarity asynchronously.

        Tries in order:
        1. Cache lookup
        2. Local embeddings
        3. LLM-judged similarity
        4. Term overlap fallback
        """
        # Check cache
        cached_score = _CACHE.get_similarity(text1, text2)
        if cached_score is not None:
            return SimilarityResult(
                score=cached_score,
                method='cached',
                cached=True
            )

        # Try local embeddings first (fast)
        if self._use_local:
            emb1 = self._compute_embedding(text1)
            emb2 = self._compute_embedding(text2)

            if emb1 is not None and emb2 is not None:
                score = self._cosine_similarity(emb1, emb2)
                score = (score + 1) / 2  # Normalize
                _CACHE.set_similarity(text1, text2, score)
                return SimilarityResult(
                    score=score,
                    method='embedding',
                    cached=False,
                    details=f'model={self._model_name}'
                )

        # Try LLM-judged similarity
        if self._llm_adapter is not None:
            try:
                result = await self._llm_similarity(text1, text2)
                if result is not None:
                    _CACHE.set_similarity(text1, text2, result.score)
                    return result
            except Exception:
                pass

        # Fallback to term overlap
        return self._term_overlap_similarity(text1, text2)

    async def _llm_similarity(self, text1: str, text2: str) -> Optional[SimilarityResult]:
        """Use LLM to judge semantic similarity."""
        if self._llm_adapter is None:
            return None

        prompt = self.SIMILARITY_PROMPT.format(
            text_a=text1[:500],  # Truncate for efficiency
            text_b=text2[:500]
        )

        try:
            response = await self._llm_adapter(prompt, timeout=30)
            if not response.success:
                return None

            # Parse JSON response
            content = response.content.strip()
            # Find JSON in response
            json_match = re.search(r'\{[^}]+\}', content)
            if json_match:
                data = json.loads(json_match.group())
                score = float(data.get('score', 0.5))
                reason = data.get('reason', '')
                return SimilarityResult(
                    score=max(0.0, min(1.0, score)),
                    method='llm',
                    cached=False,
                    details=reason[:100] if reason else None
                )
        except Exception:
            pass

        return None

    def _term_overlap_similarity(self, text1: str, text2: str) -> SimilarityResult:
        """Fallback term-overlap similarity (enhanced Jaccard)."""
        # Extract terms
        terms1 = self._extract_terms(text1)
        terms2 = self._extract_terms(text2)

        if not terms1 or not terms2:
            return SimilarityResult(score=0.0, method='term_overlap', cached=False)

        # Jaccard similarity
        intersection = terms1 & terms2
        union = terms1 | terms2

        if not union:
            return SimilarityResult(score=0.0, method='term_overlap', cached=False)

        score = len(intersection) / len(union)

        return SimilarityResult(
            score=score,
            method='term_overlap',
            cached=False,
            details=f'overlap={len(intersection)}/{len(union)}'
        )

    def _extract_terms(self, text: str) -> set:
        """Extract significant terms from text."""
        # Common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'this', 'that', 'these',
            'those', 'it', 'its', 'they', 'them', 'their', 'we', 'our', 'you',
            'your', 'he', 'she', 'him', 'her', 'his', 'who', 'which', 'what',
            'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both',
            'few', 'more', 'most', 'other', 'some', 'such', 'not', 'only',
            'own', 'same', 'than', 'too', 'very', 'just', 'also', 'now'
        }

        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_-]*\b', text.lower())
        return {w for w in words if w not in stop_words and len(w) > 2}

    async def batch_similarity(self,
                                pairs: List[Tuple[str, str]]) -> List[SimilarityResult]:
        """Compute similarity for multiple pairs efficiently."""
        tasks = [self.compute_similarity(t1, t2) for t1, t2 in pairs]
        return await asyncio.gather(*tasks)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            **_CACHE.stats(),
            'local_embeddings_available': SENTENCE_TRANSFORMERS_AVAILABLE,
            'model_loaded': self._model is not None
        }

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        _CACHE.clear()


# Default global matcher instance
_DEFAULT_MATCHER: Optional[SemanticMatcher] = None


def get_semantic_matcher(llm_adapter: Optional[callable] = None) -> SemanticMatcher:
    """Get or create the default semantic matcher."""
    global _DEFAULT_MATCHER
    if _DEFAULT_MATCHER is None:
        _DEFAULT_MATCHER = SemanticMatcher(llm_adapter=llm_adapter)
    return _DEFAULT_MATCHER


async def semantic_similarity(text1: str, text2: str,
                               llm_adapter: Optional[callable] = None) -> float:
    """
    Convenience function to compute semantic similarity.

    Args:
        text1: First text
        text2: Second text
        llm_adapter: Optional async LLM query function

    Returns:
        Similarity score 0.0-1.0
    """
    matcher = get_semantic_matcher(llm_adapter)
    result = await matcher.compute_similarity(text1, text2)
    return result.score

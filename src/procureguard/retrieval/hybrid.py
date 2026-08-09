"""Hybrid retriever using reciprocal rank fusion of BM25 and LSA."""

from __future__ import annotations

from collections import defaultdict

from procureguard.data.schema import Corpus
from procureguard.retrieval.base import Retriever


class HybridRrfRetriever(Retriever):
    """Reciprocal rank fusion of BM25 and LSA vector rankings."""

    name = "hybrid_rrf"

    def __init__(
        self,
        *,
        bm25: Retriever,
        lsa: Retriever,
        rrfK: int = 60,
        candidateCount: int = 200,
    ) -> None:
        self.bm25 = bm25
        self.lsa = lsa
        self.rrfK = rrfK
        self.candidateCount = candidateCount

    def fit(self, corpus: Corpus) -> None:
        self.bm25.fit(corpus)
        self.lsa.fit(corpus)

    def search(self, query: str, topK: int) -> list[str]:
        bm25Ranked = self.bm25.search(query, self.candidateCount)
        lsaRanked = self.lsa.search(query, self.candidateCount)
        scores: dict[str, float] = defaultdict(float)
        for rank, docId in enumerate(bm25Ranked, start=1):
            scores[docId] += 1.0 / (self.rrfK + rank)
        for rank, docId in enumerate(lsaRanked, start=1):
            scores[docId] += 1.0 / (self.rrfK + rank)
        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        return [docId for docId, _score in ranked[:topK]]

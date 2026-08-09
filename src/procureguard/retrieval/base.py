"""Retriever abstraction and shared tokenization."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Sequence

from procureguard.data.schema import Corpus, DocumentRecord

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric tokens."""
    return _TOKEN_PATTERN.findall(text.lower())


class Retriever(ABC):
    """Base class for deterministic retrievers."""

    name: str

    @abstractmethod
    def fit(self, corpus: Corpus) -> None:
        """Fit the retriever on the corpus (must be deterministic)."""

    @abstractmethod
    def search(self, query: str, topK: int) -> list[str]:
        """Return the top-K ranked document IDs for a query."""


class RetrieverRegistry:
    """Factory for building the four retrieval benchmark methods."""

    def __init__(
        self,
        *,
        lsaComponents: int = 100,
        rrfK: int = 60,
        candidateCount: int = 200,
        seed: int = 42,
    ) -> None:
        self.lsaComponents = lsaComponents
        self.rrfK = rrfK
        self.candidateCount = candidateCount
        self.seed = seed

    def build(self, name: str) -> Retriever:
        """Build a retriever by name."""
        from procureguard.retrieval.bm25 import BM25Retriever
        from procureguard.retrieval.hybrid import HybridRrfRetriever
        from procureguard.retrieval.lsa import LsaVectorRetriever
        from procureguard.retrieval.no_retrieval import NoRetrievalRetriever

        if name == "no_retrieval":
            return NoRetrievalRetriever()
        if name == "bm25":
            return BM25Retriever()
        if name == "lsa_vector":
            return LsaVectorRetriever(components=self.lsaComponents, seed=self.seed)
        if name == "hybrid_rrf":
            return HybridRrfRetriever(
                bm25=BM25Retriever(),
                lsa=LsaVectorRetriever(components=self.lsaComponents, seed=self.seed),
                rrfK=self.rrfK,
                candidateCount=self.candidateCount,
            )
        raise ValueError(f"Unknown retriever: {name}")


def corpusById(corpus: Corpus) -> dict[str, DocumentRecord]:
    """Return a mapping from document ID to document."""
    return {doc.docId: doc for doc in corpus.documents}


def orderedDocIds(corpus: Corpus) -> Sequence[str]:
    """Return corpus document IDs in deterministic order."""
    return [doc.docId for doc in corpus.documents]

"""BM25 retriever built on rank-bm25."""

from __future__ import annotations

from procureguard.data.schema import Corpus
from procureguard.retrieval.base import Retriever, tokenize


class BM25Retriever(Retriever):
    """Deterministic BM25Okapi retriever over the synthetic corpus."""

    name = "bm25"

    def __init__(self) -> None:
        self._docIds: list[str] = []
        self._bm25: object | None = None

    def fit(self, corpus: Corpus) -> None:
        from rank_bm25 import BM25Okapi

        self._docIds = [doc.docId for doc in corpus.documents]
        tokenizedDocs = [tokenize(doc.content) for doc in corpus.documents]
        self._bm25 = BM25Okapi(tokenizedDocs)

    def search(self, query: str, topK: int) -> list[str]:
        if self._bm25 is None:
            raise RuntimeError("BM25 retriever must be fit before searching.")
        scores = self._bm25.get_scores(tokenize(query))  # type: ignore[attr-defined]
        ranked = sorted(
            zip(self._docIds, scores, strict=True),
            key=lambda pair: (-pair[1], pair[0]),
        )
        return [docId for docId, _score in ranked[:topK]]

"""No-retrieval baseline retriever."""

from __future__ import annotations

from procureguard.data.schema import Corpus
from procureguard.retrieval.base import Retriever


class NoRetrievalRetriever(Retriever):
    """Baseline retriever that returns an empty ranking."""

    name = "no_retrieval"

    def fit(self, _corpus: Corpus) -> None:
        return None

    def search(self, query: str, topK: int) -> list[str]:
        del query, topK
        return []

"""LSA vector retriever using scikit-learn TF-IDF and TruncatedSVD.

This is a local, CPU-only, network-free embedding approach. No
sentence-transformers or model downloads are used.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from procureguard.data.schema import Corpus
from procureguard.retrieval.base import Retriever


class LsaVectorRetriever(Retriever):
    """Cosine vector search over deterministic LSA embeddings."""

    name = "lsa_vector"

    def __init__(self, *, components: int = 100, seed: int = 42) -> None:
        self.components = components
        self.seed = seed
        self._docIds: list[str] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._svd: TruncatedSVD | None = None
        self._docVectors: np.ndarray | None = None

    def fit(self, corpus: Corpus) -> None:
        contents = [doc.content for doc in corpus.documents]
        self._docIds = [doc.docId for doc in corpus.documents]
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"[a-z0-9]+",  # noqa: S106  # nosec B106
            sublinear_tf=True,
        )
        tfidf = self._vectorizer.fit_transform(contents)
        self._svd = TruncatedSVD(
            n_components=min(self.components, tfidf.shape[1]),
            random_state=self.seed,
        )
        self._docVectors = self._svd.fit_transform(tfidf)
        norms = np.linalg.norm(self._docVectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._docVectors = self._docVectors / norms

    def search(self, query: str, topK: int) -> list[str]:
        if self._vectorizer is None or self._svd is None or self._docVectors is None:
            raise RuntimeError("LSA retriever must be fit before searching.")
        queryTfidf = self._vectorizer.transform([query])
        if isinstance(queryTfidf, csr_matrix) and queryTfidf.nnz == 0:
            return []
        queryVector = self._svd.transform(queryTfidf)
        queryNorm = np.linalg.norm(queryVector)
        if queryNorm == 0:
            return []
        queryVector = queryVector / queryNorm
        scores = self._docVectors @ queryVector.reshape(-1, 1)
        ranked = sorted(
            zip(self._docIds, scores[:, 0].tolist(), strict=True),
            key=lambda pair: (-float(pair[1]), pair[0]),
        )
        return [docId for docId, _score in ranked[:topK]]

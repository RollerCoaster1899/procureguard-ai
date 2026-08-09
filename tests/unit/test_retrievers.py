"""Unit tests for the four retrieval methods."""

from __future__ import annotations

from procureguard.retrieval.base import RetrieverRegistry, corpusById


def _fitAll(corpus):
    registry = RetrieverRegistry(lsaComponents=50, rrfK=60, candidateCount=200, seed=42)
    retrievers = {
        method: registry.build(method)
        for method in ["no_retrieval", "bm25", "lsa_vector", "hybrid_rrf"]
    }
    for retriever in retrievers.values():
        retriever.fit(corpus)
    return retrievers


def testNoRetrievalReturnsEmpty(bundle):
    retrievers = _fitAll(bundle.corpus)
    assert retrievers["no_retrieval"].search("anything", 10) == []


def testBm25RetrievesSupplierDocs(bundle):
    retrievers = _fitAll(bundle.corpus)
    query = "recommend a supplier for electronics procurement for Summit Steel Works"
    ranked = retrievers["bm25"].search(query, 5)
    assert len(ranked) > 0
    assert all(docId in corpusById(bundle.corpus) for docId in ranked)


def testLsaDeterministic(bundle):
    retrievers = _fitAll(bundle.corpus)
    query = bundle.queries[0].query
    first = retrievers["lsa_vector"].search(query, 10)
    second = retrievers["lsa_vector"].search(query, 10)
    assert first == second
    assert len(first) == 10


def testHybridDeterministicAndFuses(bundle):
    retrievers = _fitAll(bundle.corpus)
    query = bundle.queries[0].query
    first = retrievers["hybrid_rrf"].search(query, 10)
    second = retrievers["hybrid_rrf"].search(query, 10)
    assert first == second
    assert len(first) <= 10


def testRetrieverSearchBeforeFitRaises(bundle):
    registry = RetrieverRegistry()
    retriever = registry.build("bm25")
    try:
        retriever.search("query", 5)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected RuntimeError when searching before fit")


def testUnknownRetrieverRaises():
    registry = RetrieverRegistry()
    try:
        registry.build("nope")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for unknown retriever")


def testCorpusByIdMapsAllDocs(bundle):
    mapping = corpusById(bundle.corpus)
    assert len(mapping) == len(bundle.corpus.documents)

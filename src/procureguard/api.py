"""FastAPI service for ProcureGuard AI.

Exposes ``/health`` and ``/v1/recommendations``. The recommendation endpoint
uses the guarded RAG + MCP decision path with dependency injection so tests can
swap in an in-process facts provider. The default provider is the deterministic
scripted provider; the live deepseek provider is only used when explicitly
configured with a key.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from procureguard import __version__
from procureguard.config import ExperimentConfig, loadExperimentConfig
from procureguard.data.io import loadBundle
from procureguard.data.schema import Corpus, DataBundle
from procureguard.logging_util import getLogger, setRunId
from procureguard.mcp.client import FactsProvider, InProcessFactsProvider
from procureguard.retrieval.base import Retriever, RetrieverRegistry, corpusById
from procureguard.workflow import engine
from procureguard.workflow.factory import buildDecisionProvider
from procureguard.workflow.provider import DecisionProvider

logger = getLogger("procureguard.api")


class RecommendationRequest(BaseModel):
    """Input contract for a procurement recommendation request."""

    query: str = Field(min_length=1, max_length=4000)
    supplierId: str = Field(min_length=1, max_length=64)
    category: str = Field(min_length=1, max_length=64)
    budgetAmount: float | None = Field(default=None, ge=0)


class RecommendationResponse(BaseModel):
    """Output contract for a procurement recommendation."""

    action: str
    rationale: str
    riskFlag: bool
    complianceFlag: bool
    budgetAdherence: bool
    requiresHumanApproval: bool
    proposedWrite: bool
    citations: list[str]
    quarantineHit: bool
    unsupportedFacts: list[str]
    provider: str
    requestId: str
    latencyMs: float
    toolCalls: int


class HealthResponse(BaseModel):
    """Health endpoint payload."""

    status: str
    version: str


class Container:
    """Application container with injectable dependencies."""

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        bundle: DataBundle,
        retriever: Retriever,
        factsProvider: FactsProvider,
        provider: DecisionProvider,
    ) -> None:
        self.config = config
        self.bundle = bundle
        self.retriever = retriever
        self.factsProvider = factsProvider
        self.provider = provider
        self.corpus: Corpus = bundle.corpus
        self.corpusById = corpusById(bundle.corpus)
        self.allDocIds = set(self.corpusById.keys())


def _defaultRootDir() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _loadBundleOrRaise(processedDir: Path) -> DataBundle:
    try:
        return loadBundle(processedDir)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Dataset not generated. Run `uv run python scripts/reproduce.py --smoke` "
            "or the full experiment before starting the API."
        ) from exc


def buildContainer(
    *,
    rootDir: Path | None = None,
    providerName: str | None = None,
    configPath: Path | None = None,
) -> Container:
    """Build the default application container from local artifacts."""
    rootDir = rootDir or _defaultRootDir()
    configPath = configPath or (rootDir / "configs" / "base.yaml")
    config = loadExperimentConfig(configPath, rootDir=rootDir, outputDir=rootDir / "reports")
    bundle = _loadBundleOrRaise(config.paths.processedDir)
    registry = RetrieverRegistry(
        lsaComponents=config.retrieval.lsaComponents,
        rrfK=config.retrieval.rrfK,
        candidateCount=config.retrieval.candidateCount,
        seed=config.data.seed,
    )
    retriever = registry.build("hybrid_rrf")
    retriever.fit(bundle.corpus)
    providerName = providerName or os.environ.get("PROCUREGUARD_PROVIDER", "scripted")
    provider = buildDecisionProvider(providerName, config)
    factsProvider = InProcessFactsProvider(bundle.fixtureFacts)
    return Container(
        config=config,
        bundle=bundle,
        retriever=retriever,
        factsProvider=factsProvider,
        provider=provider,
    )


def _requestId(request: Request) -> str:
    return request.headers.get("X-Request-ID") or uuid.uuid4().hex


def createApp(container: Container | None = None) -> FastAPI:
    """Create the FastAPI application with an injectable container."""
    container = container or buildContainer()
    app = FastAPI(title="ProcureGuard AI", version=__version__)
    app.state.container = container

    @app.middleware("http")
    async def correlationMiddleware(
        request: Request, callNext: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        requestId = _requestId(request)
        setRunId(requestId)
        response = await callNext(request)
        response.headers["X-Request-ID"] = requestId
        return response

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__)

    @app.post(
        "/v1/recommendations",
        response_model=RecommendationResponse,
    )
    async def recommend(
        payload: RecommendationRequest,
        request: Request,
    ) -> RecommendationResponse:
        requestId = _requestId(request)
        activeContainer = request.app.state.container
        ctx = engine.DecisionContext(
            query=payload.query,
            supplierId=payload.supplierId,
            category=payload.category,
            injectionAttempt=False,
            budgetAmount=payload.budgetAmount,
        )
        start = time.perf_counter()
        try:
            outcome = await engine.runGuardedRagMcp(
                ctx,
                activeContainer.provider,
                activeContainer.retriever,
                activeContainer.corpusById,
                activeContainer.factsProvider,
                activeContainer.allDocIds,
                topK=activeContainer.config.workflow.topK,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("recommendation failed requestId=%s", requestId)
            raise HTTPException(status_code=502, detail=f"Recommendation failed: {exc}") from exc
        latencyMs = (time.perf_counter() - start) * 1000.0
        decision = outcome.decision
        return RecommendationResponse(
            action=decision.action,
            rationale=decision.rationale,
            riskFlag=decision.riskFlag,
            complianceFlag=decision.complianceFlag,
            budgetAdherence=decision.budgetAdherence,
            requiresHumanApproval=decision.requiresHumanApproval,
            proposedWrite=decision.proposedWrite,
            citations=list(decision.citations),
            quarantineHit=decision.quarantineHit,
            unsupportedFacts=list(decision.unsupportedFacts),
            provider=activeContainer.provider.name,
            requestId=requestId,
            latencyMs=round(latencyMs, 4),
            toolCalls=outcome.toolCalls,
        )

    return app


app = createApp()

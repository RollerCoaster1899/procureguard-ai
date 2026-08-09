"""Structured logging helpers with a correlation run ID.

Every log record carries a run ID that is set for the duration of a reproduce
run or an API request. The run ID makes trace lines attributable to a single
benchmark run without leaking prompts or secrets.
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar

runIdVar: ContextVar[str] = ContextVar("procureguard_run_id", default="-")


def setRunId(runId: str) -> None:
    """Set the correlation run ID for the current context."""
    runIdVar.set(runId)


def getRunId() -> str:
    """Return the current correlation run ID."""
    return runIdVar.get()


class RunIdFilter(logging.Filter):
    """Attach the current run ID to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.runId = getRunId()
        return True


def configureLogging(level: int = logging.INFO) -> None:
    """Configure a single stdout handler with the run ID filter."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s][run=%(runId)s] %(message)s")
    )
    handler.addFilter(RunIdFilter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)


def getLogger(name: str) -> logging.Logger:
    """Return a logger with the run ID filter already configured."""
    logger = logging.getLogger(name)
    if not any(isinstance(handler, RunIdFilter) for handler in logger.handlers):
        logger.addFilter(RunIdFilter())
    return logger

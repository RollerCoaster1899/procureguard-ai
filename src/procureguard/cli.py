"""Command-line interface for ProcureGuard AI."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from procureguard import __version__

app = typer.Typer(
    name="procureguard",
    help="ProcureGuard AI: synthetic procurement decision copilot benchmark.",
)


@app.command("run")
def runCommand(
    smoke: Annotated[bool, typer.Option(help="Run a tiny deterministic smoke experiment.")] = False,
    output: Annotated[Path | None, typer.Option(help="Output directory for artifacts.")] = None,
    provider: Annotated[
        str,
        typer.Option(help="Decision provider: scripted (default) or deepseek (live)."),
    ] = "scripted",
    config: Annotated[Path | None, typer.Option(help="Path to base YAML config.")] = None,
    runId: Annotated[str | None, typer.Option(help="Correlation run ID.")] = None,
) -> None:
    """Run the full offline experiment (data, retrieval, workflow)."""
    from procureguard.pipeline import runExperiment

    combined = runExperiment(
        configPath=config,
        outputDir=output,
        smoke=smoke,
        providerName=provider,
        runId=runId,
    )
    retrieval = combined["retrieval"]
    workflow = combined["workflow"]
    typer.echo("Retrieval nDCG@10:")
    for method, entry in retrieval["results"].items():
        typer.echo(f"  {method}: {entry['metrics']['ndcg@10']['mean']:.4f}")
    typer.echo("Workflow recommendation accuracy:")
    for method, entry in workflow["results"].items():
        typer.echo(f"  {method}: {entry['metrics']['recommendationAccuracy']:.4f}")


@app.command("serve")
def serveCommand(
    host: Annotated[str, typer.Option(help="Bind host.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Bind port.")] = 8000,
) -> None:
    """Start the FastAPI service."""
    import uvicorn

    uvicorn.run("procureguard.api:app", host=host, port=port)


@app.command("mcp-server")
def mcpServerCommand() -> None:
    """Run the read-only MCP fixture server over stdio."""
    from procureguard.mcp import server

    server.main()


@app.command("version")
def versionCommand() -> None:
    """Print the package version."""
    typer.echo(__version__)


def main() -> None:
    app()


if __name__ == "__main__":
    main()

"""Command-line interface (``frp``). Subcommands are added milestone by milestone."""

from __future__ import annotations

import json

import typer

from frp import __version__
from frp.config import ConfigError, load_settings
from frp.envinfo import collect_env

app = typer.Typer(help="Face recognition pipeline.", no_args_is_help=True, add_completion=False)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def env() -> None:
    """Print hardware, library and git info as JSON (same data stored in result files)."""
    typer.echo(json.dumps(collect_env(), indent=2))


@app.command()
def check_config() -> None:
    """Validate configuration from the environment and exit non-zero if it is invalid."""
    try:
        load_settings()
    except ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo("configuration OK")


if __name__ == "__main__":
    app()

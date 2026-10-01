"""Command-line interface (``frp``). Subcommands are added milestone by milestone."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from PIL import Image

from frp import __version__
from frp.align import AlignedFace, align_face
from frp.config import ConfigError, load_settings
from frp.debug import alignment_overlay
from frp.detect import FacePolicy, select_faces
from frp.envinfo import collect_env
from frp.errors import FrpError, InvalidLandmarksError
from frp.factory import build_detector
from frp.imageio import load_image

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


@app.command("debug-align")
def debug_align(
    image: Annotated[Path, typer.Argument(help="Input image (JPEG, PNG, GIF, WebP, BMP, TIFF).")],
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Where to write the overlay PNG.")
    ] = None,
) -> None:
    """Detect the largest face, align it, and write a debug overlay picture.

    The overlay shows the detection (box, five landmarks) next to the aligned 112x112 crop with
    the canonical template points. The output contains the input's face: treat it as sensitive.
    """
    target = out or Path(f"{image.stem}_align.png")
    try:
        loaded = load_image(image.read_bytes())
        faces = build_detector().detect(loaded.array)
        chosen = select_faces(faces, FacePolicy.LARGEST)[0]
    except FrpError as exc:
        typer.echo(f"{exc.code}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except OSError as exc:
        typer.echo(f"Cannot read {image}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    aligned: AlignedFace | None = None
    message: str | None = None
    try:
        aligned = align_face(loaded.array, chosen.landmarks)
    except InvalidLandmarksError as exc:
        message = f"{exc.reason}: {exc}"
    overlay = alignment_overlay(loaded.array, [chosen], aligned, message)
    Image.fromarray(overlay).save(target)
    typer.echo(f"wrote {target}" + ("" if aligned else f" (alignment failed: {message})"))
    if aligned is None:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

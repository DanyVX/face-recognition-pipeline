"""Download and verify the pretrained models into ``MODEL_DIR`` (default ``models/``).

Usage: ``uv run python scripts/download_models.py [--list] [--force] [KEY ...]``

The InsightFace weights are licensed for non-commercial research use only (see MODEL_CARD.md).
"""

from __future__ import annotations

import argparse
import logging
import sys

from frp.config import load_settings
from frp.errors import FrpError
from frp.models import MODELS, ensure_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("keys", nargs="*", help="model keys to fetch (default: all)")
    parser.add_argument("--list", action="store_true", help="list known models and exit")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.list:
        for spec in MODELS.values():
            print(f"{spec.key:14} {spec.filename:18} {spec.size_bytes:>12,} B  {spec.license}")
        return 0

    unknown = [k for k in args.keys if k not in MODELS]
    if unknown:
        print(f"Unknown model key(s): {unknown}. Known: {sorted(MODELS)}", file=sys.stderr)
        return 2
    models_dir = load_settings().model_dir
    try:
        for key in args.keys or list(MODELS):
            path = ensure_model(MODELS[key], models_dir, force=args.force)
            print(f"OK  {key}: {path}")
    except FrpError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

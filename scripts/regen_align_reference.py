"""Regenerate tests/fixtures/align_reference.png from the synthetic scene.

Only run this when the alignment behavior is *intentionally* changed; the stored image is the
regression anchor that makes accidental changes visible. Review the diff of the PNG afterwards.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so that `tests.synthetic` is importable when run as a script

from tests.synthetic import synthetic_scene  # noqa: E402

from frp.align import align_face  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "align_reference.png"

if __name__ == "__main__":
    scene, landmarks = synthetic_scene()
    Image.fromarray(align_face(scene, landmarks).image).save(OUT)
    print(f"wrote {OUT}")

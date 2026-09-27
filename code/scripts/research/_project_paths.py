"""Portable paths shared by reviewer-package research scripts."""

import os
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = CODE_ROOT.parent

WORKSPACE_ROOT = PACKAGE_ROOT
SUPPLEMENT_ROOT = PACKAGE_ROOT
PAPER_ROOT = Path(os.environ.get("PAPER_ROOT", PACKAGE_ROOT / "build"))

PAPER_DATA_ROOT = SUPPLEMENT_ROOT / "paper_data"
SCRATCH_ROOT = Path(
    os.environ.get("SCRATCH_ROOT")
    or os.environ.get("OUTPUT_ROOT")
    or SUPPLEMENT_ROOT / "outputs"
)

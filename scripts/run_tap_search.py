#!/usr/bin/env python3
"""Memory-bounded, restartable paired detector-tap campaign for Ubuntu/WSL."""

import os
import sys
from pathlib import Path

# Set these before numpy/BLAS loads, also in every subprocess.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tdmr2d.tap_search import main


if __name__ == "__main__":
    raise SystemExit(main())

"""Filesystem helpers: run directories, CSV/JSON output, logging.

The canonical result schema (and the column order required by the spec) is
:data:`REQUIRED_COLUMNS`. CSVs always lead with those columns; any extra
diagnostic columns follow.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd

# Required summary/result columns, in the exact order mandated by the spec.
REQUIRED_COLUMNS: List[str] = [
    "family", "rate", "snr_db", "iti_coeff", "num_bits", "bit_errors", "BER",
    "block_errors", "block_error_rate", "seed", "runtime_sec",
]


def timestamp() -> str:
    """Sortable timestamp (microseconds keep parallel/quick runs unique)."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def ensure_output_tree(root: str | Path = "outputs") -> Path:
    """Create the standard ``outputs/{runs,summaries,figures,reports}`` tree."""
    root = Path(root)
    for sub in ("runs", "summaries", "figures", "reports"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def make_run_dir(base_dir: str | Path) -> Path:
    """Create and return a fresh timestamped run directory under ``base_dir``."""
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    run_dir = base / timestamp()
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with REQUIRED_COLUMNS first (when present), extras after."""
    present_required = [c for c in REQUIRED_COLUMNS if c in df.columns]
    extras = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    return df[present_required + extras]


def rows_to_frame(rows: Sequence[Dict]) -> pd.DataFrame:
    return order_columns(pd.DataFrame(list(rows)))


def save_csv(rows: Sequence[Dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_to_frame(rows).to_csv(path, index=False)
    return path


def save_json(obj, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
    return path


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)


def setup_logger(name: str, log_path: str | Path) -> logging.Logger:
    """Logger that writes both to ``log_path`` (run.log) and stdout."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def find_results_csvs(root: str | Path) -> List[Path]:
    """Locate every ``results.csv`` under ``root`` (recursively)."""
    root = Path(root)
    if root.is_file() and root.name.endswith(".csv"):
        return [root]
    return sorted(root.rglob("results.csv"))

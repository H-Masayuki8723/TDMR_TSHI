"""Sector-FSR aggregation and waterfall extrapolation.

The target region for storage studies can be around FSR=1e-8...1e-10, far below
what ordinary Monte Carlo can observe directly in small runs. This module
aggregates many sector-count chunks and fits a simple waterfall tail model:

    log10(FSR) = slope * SNR_dB + intercept

Only nonzero, non-saturated FSR observations are used for the fit by default.
Zero-error points are kept in the aggregate CSV as upper-bound evidence but are
not treated as exact FSR=0 observations in the log-linear fit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


def parse_float_list(spec: str | Sequence[float]) -> List[float]:
    if isinstance(spec, str):
        vals = []
        for part in spec.split(","):
            part = part.strip()
            if part:
                vals.append(float(part))
        return vals
    return [float(v) for v in spec]


def parse_column_list(spec: str | Sequence[str]) -> List[str]:
    if isinstance(spec, str):
        return [p.strip() for p in spec.split(",") if p.strip()]
    return [str(v) for v in spec]


def find_fsr_csvs(sources: Iterable[str | Path]) -> List[Path]:
    """Find result/aggregate CSVs under files or directories."""
    paths: List[Path] = []
    for src in sources:
        p = Path(src)
        if p.is_file():
            paths.append(p)
        elif p.is_dir():
            found = list(p.rglob("results.csv"))
            found.extend(p.rglob("*aggregate*.csv"))
            paths.extend(found)
    return sorted(dict.fromkeys(paths))


def load_fsr_rows(sources: Iterable[str | Path], *, name_filter: str | None = None) -> pd.DataFrame:
    """Load all CSV rows that carry sector-count based FSR columns."""
    frames: List[pd.DataFrame] = []
    for path in find_fsr_csvs(sources):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "sector_count" not in df.columns:
            continue
        if "sector_errors" not in df.columns and "FSR" not in df.columns:
            continue
        if name_filter and "name" in df.columns:
            df = df[df["name"].astype(str).str.contains(name_filter, na=False)]
        elif name_filter:
            continue
        if df.empty:
            continue
        df = df.copy()
        df["source"] = str(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    for col in ("snr_db", "iti_coeff", "sector_count", "sector_errors", "FSR",
                "bit_errors", "num_bits", "runtime_sec"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "sector_errors" not in out.columns:
        out["sector_errors"] = np.rint(out["FSR"] * out["sector_count"])
    if "FSR" not in out.columns:
        out["FSR"] = out["sector_errors"] / out["sector_count"]
    return out


def aggregate_fsr(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    """Aggregate chunked FSR rows by group columns and SNR."""
    if df.empty:
        return pd.DataFrame()
    group_cols = [c for c in group_cols if c in df.columns]
    keys = group_cols + ["snr_db"]
    required = {"snr_db", "sector_errors", "sector_count"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"FSR rows missing required column(s): {missing}")

    work = df.dropna(subset=["snr_db", "sector_count"]).copy()
    work = work[work["sector_count"] > 0]
    agg = {
        "sector_errors": ("sector_errors", "sum"),
        "sector_count": ("sector_count", "sum"),
    }
    if "bit_errors" in work.columns and "num_bits" in work.columns:
        agg["bit_errors"] = ("bit_errors", "sum")
        agg["num_bits"] = ("num_bits", "sum")
    if "runtime_sec" in work.columns:
        agg["runtime_sec"] = ("runtime_sec", "sum")
    out = work.groupby(keys, as_index=False, dropna=False).agg(**agg)
    out["FSR"] = out["sector_errors"] / out["sector_count"]
    if "bit_errors" in out.columns and "num_bits" in out.columns:
        out["BER"] = out["bit_errors"] / out["num_bits"]
    out["FSR_upper_95_zero_fail"] = np.where(
        out["sector_errors"] == 0,
        3.0 / out["sector_count"],
        np.nan,
    )
    return out.sort_values(keys).reset_index(drop=True)


def _fit_group(g: pd.DataFrame, *, max_fit_fsr: float, min_fit_fsr: float,
               min_points: int) -> Tuple[dict, pd.DataFrame]:
    usable = g[
        (g["FSR"] > float(min_fit_fsr))
        & (g["FSR"] < float(max_fit_fsr))
        & (g["sector_count"] > 0)
    ].sort_values("snr_db")
    base = {
        "fit_model": "log10_fsr_linear_vs_snr",
        "fit_points": int(len(usable)),
        "fit_min_snr_db": float(usable["snr_db"].min()) if len(usable) else np.nan,
        "fit_max_snr_db": float(usable["snr_db"].max()) if len(usable) else np.nan,
        "fit_min_fsr": float(usable["FSR"].min()) if len(usable) else np.nan,
        "fit_max_fsr": float(usable["FSR"].max()) if len(usable) else np.nan,
        "slope_log10_fsr_per_db": np.nan,
        "intercept_log10_fsr": np.nan,
        "r2_log10_fsr": np.nan,
        "fit_note": "",
    }
    if len(usable) < int(min_points):
        base["fit_note"] = f"insufficient nonzero/non-saturated FSR points (<{int(min_points)})"
        return base, usable

    x = usable["snr_db"].to_numpy(dtype=float)
    y = np.log10(usable["FSR"].to_numpy(dtype=float))
    w = np.sqrt(usable["sector_count"].to_numpy(dtype=float))
    slope, intercept = np.polyfit(x, y, deg=1, w=w)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    base.update({
        "slope_log10_fsr_per_db": float(slope),
        "intercept_log10_fsr": float(intercept),
        "r2_log10_fsr": float(r2),
    })
    if slope >= 0:
        base["fit_note"] = "non-negative slope; target SNR is not meaningful"
    return base, usable


def extrapolate_fsr_targets(agg: pd.DataFrame, group_cols: Sequence[str],
                            targets: Sequence[float], *,
                            max_fit_fsr: float = 0.8,
                            min_fit_fsr: float = 0.0,
                            min_points: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit each group and estimate SNR required for target FSR values."""
    if agg.empty:
        return pd.DataFrame(), pd.DataFrame()
    group_cols = [c for c in group_cols if c in agg.columns]
    groups = [((), agg)] if not group_cols else agg.groupby(group_cols, dropna=False)
    rows: List[dict] = []
    fit_points: List[pd.DataFrame] = []
    for key, g in groups:
        if group_cols:
            if not isinstance(key, tuple):
                key = (key,)
            group_meta = dict(zip(group_cols, key))
        else:
            group_meta = {}
        fit, usable = _fit_group(
            g, max_fit_fsr=max_fit_fsr, min_fit_fsr=min_fit_fsr, min_points=min_points,
        )
        if not usable.empty:
            tmp = usable.copy()
            for col, val in group_meta.items():
                tmp[col] = val
            fit_points.append(tmp)
        for target in targets:
            target = float(target)
            row = dict(group_meta)
            row.update(fit)
            row["target_fsr"] = target
            row["target_log10_fsr"] = float(np.log10(target))
            row["estimated_snr_db"] = np.nan
            row["extrapolation_beyond_fit_max_db"] = np.nan
            if np.isfinite(fit["slope_log10_fsr_per_db"]) and fit["slope_log10_fsr_per_db"] < 0:
                est = (np.log10(target) - fit["intercept_log10_fsr"]) / fit["slope_log10_fsr_per_db"]
                row["estimated_snr_db"] = float(est)
                row["extrapolation_beyond_fit_max_db"] = float(est - fit["fit_max_snr_db"])
            rows.append(row)
    return pd.DataFrame(rows), pd.concat(fit_points, ignore_index=True) if fit_points else pd.DataFrame()


def plot_fsr_extrapolation(agg: pd.DataFrame, targets_df: pd.DataFrame,
                           group_cols: Sequence[str], path: str | Path) -> Path:
    """Plot measured FSR points and fitted target crossings."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    group_cols = [c for c in group_cols if c in agg.columns]
    groups = [((), agg)] if not group_cols else agg.groupby(group_cols, dropna=False)
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ymin = 1.0
    for key, g in groups:
        if group_cols:
            if not isinstance(key, tuple):
                key = (key,)
            label = ", ".join(f"{c}={v:g}" if isinstance(v, (int, float)) else f"{c}={v}"
                              for c, v in zip(group_cols, key))
            filt = np.ones(len(targets_df), dtype=bool)
            for col, val in zip(group_cols, key):
                filt &= targets_df[col].astype(str).to_numpy() == str(val)
            tsub = targets_df[filt]
        else:
            label = "all"
            tsub = targets_df
        g = g.sort_values("snr_db")
        y = g["FSR"].to_numpy(dtype=float)
        yplot = np.maximum(y, np.where(g["sector_count"] > 0, 0.5 / g["sector_count"], 1e-12))
        ymin = min(ymin, float(np.min(yplot)))
        line, = ax.plot(g["snr_db"], yplot, marker="o", label=label)
        fit = tsub.dropna(subset=["estimated_snr_db"]).head(1)
        if not fit.empty:
            slope = float(fit.iloc[0]["slope_log10_fsr_per_db"])
            intercept = float(fit.iloc[0]["intercept_log10_fsr"])
            if np.isfinite(slope) and slope < 0:
                xmax = max(float(g["snr_db"].max()), float(tsub["estimated_snr_db"].max()))
                xs = np.linspace(float(g["snr_db"].min()), xmax, 100)
                ys = 10.0 ** (slope * xs + intercept)
                ax.plot(xs, ys, linestyle="--", color=line.get_color(), alpha=0.7)
    for target in sorted(targets_df["target_fsr"].dropna().unique(), reverse=True):
        ax.axhline(float(target), color="0.45", linestyle=":", linewidth=0.9)
        ax.text(0.995, float(target), f"{target:g}", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=8, color="0.35")
    ax.set_yscale("log")
    if ymin > 0:
        ax.set_ylim(bottom=min(ymin, float(targets_df["target_fsr"].min())) * 0.5, top=1.2)
    ax.set_xlabel("channel SNR (dB)")
    ax.set_ylabel("4KB sector FSR")
    ax.set_title("FSR waterfall extrapolation")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path

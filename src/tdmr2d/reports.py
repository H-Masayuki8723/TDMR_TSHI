"""Plotting and summary helpers.

Uses a headless ``Agg`` backend so figures render inside Docker without a
display. Zero-BER points are floored to half a bit error (``0.5 / num_bits``) so
they remain visible on a log axis; such points are drawn hollow and mean
"no errors observed (BER below the measurable floor)".
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tdmr2d-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .io import find_results_csvs, order_columns  # noqa: E402


def _floor(df: pd.DataFrame) -> float:
    nb = int(df["num_bits"].min()) if "num_bits" in df and len(df) else 1
    return 0.5 / max(nb, 1)


def _fmt_val(v) -> str:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return f"{v:g}"
    return str(v)


def _plot_grouped(df: pd.DataFrame, x: str, group: str, path: str | Path,
                  xlabel: str, title: str, target_ber: Optional[float],
                  label_prefix: Optional[str] = None, y_col: str = "BER",
                  ylabel: str = "pre-ECC BER") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if label_prefix is None:
        label_prefix = f"{group}="
    floor = _floor(df)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    sub = df[df[x].notna()].copy()
    for gval, g in sorted(sub.groupby(group), key=lambda kv: str(kv[0])):
        g = g.sort_values(x)
        y = g[y_col].to_numpy(dtype=float)
        yplot = [max(v, floor) for v in y]
        is_floor = [v <= 0 for v in y]
        line, = ax.plot(g[x].to_numpy(dtype=float), yplot, marker="o",
                        label=f"{label_prefix}{_fmt_val(gval)}")
        # Redraw floored (zero-error) points as hollow markers.
        for xi, yi, fl in zip(g[x].to_numpy(dtype=float), yplot, is_floor):
            if fl:
                ax.plot([xi], [yi], marker="o", markerfacecolor="white",
                        markeredgecolor=line.get_color(), linestyle="none")

    if target_ber:
        ax.axhline(target_ber, color="0.4", linestyle="--", linewidth=1.0,
                   label=f"target={target_ber:g}")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_ber_vs_snr(df: pd.DataFrame, path: str | Path,
                    target_ber: Optional[float] = None, title: str = "pre-ECC BER vs SNR") -> Path:
    return _plot_grouped(df, x="snr_db", group="iti_coeff", path=path,
                         xlabel="SNR (dB)", title=title, target_ber=target_ber)


def plot_ber_vs_iti(df: pd.DataFrame, path: str | Path,
                    target_ber: Optional[float] = None, title: str = "pre-ECC BER vs ITI") -> Path:
    return _plot_grouped(df, x="iti_coeff", group="snr_db", path=path,
                         xlabel="ITI cross-track coefficient", title=title, target_ber=target_ber)


def _slice(df: pd.DataFrame, col: str, val: float) -> pd.DataFrame:
    s = pd.to_numeric(df[col], errors="coerce")
    return df[(s - val).abs() < 1e-9]


def plot_compare_vs_snr(df: pd.DataFrame, path: str | Path, slice_iti: float,
                        target_ber: Optional[float] = None) -> Path:
    """Overlay BER vs SNR for each family at a fixed ITI value."""
    sub = _slice(df, "iti_coeff", slice_iti)
    return _plot_grouped(sub, x="snr_db", group="family", path=path,
                         xlabel="SNR (dB)",
                         title=f"pre-ECC BER vs SNR @ ITI={slice_iti:g} (by family)",
                         target_ber=target_ber, label_prefix="")


def plot_compare_vs_iti(df: pd.DataFrame, path: str | Path, slice_snr: float,
                        target_ber: Optional[float] = None) -> Path:
    """Overlay BER vs ITI for each family at a fixed SNR value."""
    sub = _slice(df, "snr_db", slice_snr)
    return _plot_grouped(sub, x="iti_coeff", group="family", path=path,
                         xlabel="ITI cross-track coefficient",
                         title=f"pre-ECC BER vs ITI @ SNR={slice_snr:g}dB (by family)",
                         target_ber=target_ber, label_prefix="")


def plot_ldpc_ber(df: pd.DataFrame, path: str | Path, target_ber: Optional[float] = None) -> Path:
    """Post-ECC BER (solid) vs pre-ECC BER (dashed) vs channel SNR, per ITI."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    floor = _floor(df)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for iti, g in sorted(df.groupby("iti_coeff"), key=lambda kv: kv[0]):
        g = g.sort_values("snr_db")
        x = g["snr_db"].to_numpy(dtype=float)
        post = [max(v, floor) for v in g["BER"].to_numpy(dtype=float)]
        line, = ax.plot(x, post, marker="o", label=f"post-ECC iti={iti:g}")
        if "pre_ecc_ber" in g:
            pre = [max(v, floor) for v in g["pre_ecc_ber"].to_numpy(dtype=float)]
            ax.plot(x, pre, marker="x", linestyle="--", color=line.get_color(),
                    alpha=0.6, label=f"pre-ECC iti={iti:g}")
    if target_ber:
        ax.axhline(target_ber, color="0.4", linestyle=":", linewidth=1.0, label=f"target={target_ber:g}")
    ax.set_yscale("log")
    ax.set_xlabel("channel SNR (dB)")
    ax.set_ylabel("BER")
    ax.set_title("LDPC: pre-ECC vs post-ECC BER")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_ldpc_fer(df: pd.DataFrame, path: str | Path) -> Path:
    """Frame (message) error rate vs channel SNR, per ITI."""
    return _plot_grouped(df, x="snr_db", group="iti_coeff", path=path,
                         xlabel="channel SNR (dB)", title="LDPC frame error rate (FER)",
                         target_ber=None, label_prefix="iti=",
                         y_col="block_error_rate", ylabel="FER")


def plot_sector_fsr(df: pd.DataFrame, path: str | Path) -> Path:
    """4KB-sector or configured-sector failure rate vs channel SNR, per ITI."""
    return _plot_grouped(df, x="snr_db", group="iti_coeff", path=path,
                         xlabel="channel SNR (dB)", title="Sector failure rate (FSR)",
                         target_ber=None, label_prefix="iti=",
                         y_col="FSR", ylabel="FSR")


def build_summary(csv_sources: Iterable[str | Path]) -> pd.DataFrame:
    """Concatenate result CSVs (or directories containing them) into one frame."""
    paths: List[Path] = []
    for src in csv_sources:
        paths.extend(find_results_csvs(src))
    if not paths:
        return pd.DataFrame()
    frames = []
    for p in paths:
        d = pd.read_csv(p)
        d["source"] = str(p)
        frames.append(d)
    return order_columns(pd.concat(frames, ignore_index=True))

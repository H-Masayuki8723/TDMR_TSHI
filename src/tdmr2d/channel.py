"""Simplified 2D TDMR readback channel.

Bit mapping
-----------
``0 -> -1``, ``1 -> +1`` (antipodal).

Channel model (``linear_2d_awgn``)
----------------------------------
For recorded symbols ``x[i, j]`` (track ``i``, down-track ``j``)::

    y[i, j] = c0          * x[i,   j]
            + c_down_prev  * x[i,   j-1]
            + c_down_next  * x[i,   j+1]
            + c_cross_up   * x[i-1, j]
            + c_cross_down * x[i+1, j]
            + noise

The cross-track taps model inter-track interference (ITI). Out-of-grid neighbours
are handled by the ``boundary`` policy (``zero`` padding by default; ``periodic``
and ``edge`` are also implemented for future use).

SNR / noise convention
----------------------
AWGN with ``noise ~ N(0, sigma^2)`` and

    sigma^2 = c0^2 * 10^(-SNR_dB / 10)

i.e. SNR is defined relative to the **main-tap symbol energy** ``c0^2`` (symbols
are +/-1). ``snr_db = None`` means noiseless (``sigma = 0``). This is a simple,
documented convention -- not a calibrated recording SNR (see README limitations).

This module is intentionally self-contained (numpy only) so it can later be
swapped for a numba/cupy implementation without touching the rest of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ChannelTaps:
    c0: float = 1.0
    c_down_prev: float = 0.15
    c_down_next: float = 0.15
    c_cross_up: float = 0.10
    c_cross_down: float = 0.10

    @classmethod
    def from_config(cls, ch) -> "ChannelTaps":
        return cls(
            c0=float(ch.c0),
            c_down_prev=float(ch.c_down_prev),
            c_down_next=float(ch.c_down_next),
            c_cross_up=float(ch.c_cross_up),
            c_cross_down=float(ch.c_cross_down),
        )


def map_bits_to_symbols(bits: np.ndarray) -> np.ndarray:
    """Map ``{0, 1}`` bits to ``{-1.0, +1.0}`` symbols."""
    return 2.0 * np.asarray(bits, dtype=np.float64) - 1.0


def snr_db_to_sigma(snr_db: Optional[float], c0: float) -> float:
    """Noise std-dev for the documented main-tap SNR convention."""
    if snr_db is None:
        return 0.0
    return float(c0) * 10.0 ** (-float(snr_db) / 20.0)


_PAD_MODE = {"zero": "constant", "periodic": "wrap", "edge": "edge"}


def _pad1(x: np.ndarray, boundary: str) -> np.ndarray:
    """Pad by one cell on every side according to the boundary policy."""
    try:
        mode = _PAD_MODE[boundary]
    except KeyError as exc:  # pragma: no cover - guarded by config validation
        raise ValueError(f"unknown boundary {boundary!r}; expected one of {sorted(_PAD_MODE)}") from exc
    if mode == "constant":
        return np.pad(x, 1, mode="constant", constant_values=0.0)
    return np.pad(x, 1, mode=mode)


def readback(symbols: np.ndarray, taps: ChannelTaps, snr_db: Optional[float],
             rng: Optional[np.random.Generator] = None, boundary: str = "zero") -> np.ndarray:
    """Pass a ``(num_tracks, bits_per_track)`` symbol grid through the 2D channel.

    Returns the real-valued readback ``y`` of the same shape. Noise (if any) is
    drawn from ``rng`` so a fixed seed reproduces it exactly.
    """
    x = np.asarray(symbols, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("symbols must be a 2D (tracks x down-track) array")

    p = _pad1(x, boundary)                 # shape (T+2, J+2)
    center = p[1:-1, 1:-1]
    down_prev = p[1:-1, 0:-2]              # x[i, j-1]
    down_next = p[1:-1, 2:]               # x[i, j+1]
    cross_up = p[0:-2, 1:-1]              # x[i-1, j]
    cross_down = p[2:, 1:-1]             # x[i+1, j]

    y = (taps.c0 * center
         + taps.c_down_prev * down_prev
         + taps.c_down_next * down_next
         + taps.c_cross_up * cross_up
         + taps.c_cross_down * cross_down)

    sigma = snr_db_to_sigma(snr_db, taps.c0)
    if sigma > 0.0:
        if rng is None:
            raise ValueError("rng is required when noise is enabled (snr_db is not None)")
        y = y + rng.normal(0.0, sigma, size=y.shape)
    return y

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
from typing import Dict, Optional

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


def _safe_ratio(num: float, den: float) -> Optional[float]:
    if den <= 0.0:
        return None
    return float(num / den)


def _safe_ratio_db(num: float, den: float) -> Optional[float]:
    if num <= 0.0 or den <= 0.0:
        return None
    return float(10.0 * np.log10(num / den))


def tap_energy_metrics(taps: ChannelTaps, sigma: Optional[float] = None) -> Dict:
    """Return nominal tap-energy ratios for documenting 1D/2D ITI conditions.

    These values depend only on channel coefficients, not on the transmitted
    constrained sequence. They are useful as a Level-1/2 reproducibility record:
    the same nominal ``iti_coeff`` can be compared through main/down/cross tap
    energy, even before a more physical reader-kernel calibration is available.
    """
    main = float(taps.c0 ** 2)
    down = float(taps.c_down_prev ** 2 + taps.c_down_next ** 2)
    cross = float(taps.c_cross_up ** 2 + taps.c_cross_down ** 2)
    interference = float(down + cross)
    noise_var = None if sigma is None else float(sigma ** 2)
    sinr_den = interference + (noise_var or 0.0)
    return {
        "tap_main_energy": main,
        "tap_downtrack_energy": down,
        "tap_crosstrack_energy": cross,
        "tap_interference_energy": interference,
        "tap_down_fraction_of_interference": _safe_ratio(down, interference),
        "tap_cross_fraction_of_interference": _safe_ratio(cross, interference),
        "tap_down_to_main": _safe_ratio(down, main),
        "tap_cross_to_main": _safe_ratio(cross, main),
        "tap_interference_to_main": _safe_ratio(interference, main),
        "tap_sir_db": _safe_ratio_db(main, interference),
        "tap_sinr_db": _safe_ratio_db(main, sinr_den),
        "awgn_sigma": None if sigma is None else float(sigma),
        "awgn_variance": noise_var,
    }


def _variance(count: int, total: float, total_sq: float) -> float:
    if count <= 0:
        return 0.0
    mean = total / count
    return float(max(0.0, total_sq / count - mean * mean))


def _rms(count: int, total_sq: float) -> float:
    if count <= 0:
        return 0.0
    return float(np.sqrt(max(0.0, total_sq / count)))


def _down_neighbors(center: np.ndarray, boundary: str) -> tuple[np.ndarray, np.ndarray]:
    if boundary == "periodic":
        return np.roll(center, 1, axis=1), np.roll(center, -1, axis=1)
    if boundary == "edge":
        prev = np.empty_like(center)
        nxt = np.empty_like(center)
        prev[:, 0] = center[:, 0]
        prev[:, 1:] = center[:, :-1]
        nxt[:, :-1] = center[:, 1:]
        nxt[:, -1] = center[:, -1]
        return prev, nxt
    prev = np.zeros_like(center)
    nxt = np.zeros_like(center)
    prev[:, 1:] = center[:, :-1]
    nxt[:, :-1] = center[:, 1:]
    return prev, nxt


def _cross_neighbors(x: np.ndarray, start: int, end: int, boundary: str) -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(start, end)
    if boundary == "periodic":
        return x[(rows - 1) % x.shape[0]], x[(rows + 1) % x.shape[0]]
    if boundary == "edge":
        return x[np.maximum(rows - 1, 0)], x[np.minimum(rows + 1, x.shape[0] - 1)]
    up = np.zeros((end - start, x.shape[1]), dtype=x.dtype)
    down = np.zeros_like(up)
    has_up = rows > 0
    has_down = rows < x.shape[0] - 1
    if np.any(has_up):
        up[has_up] = x[rows[has_up] - 1]
    if np.any(has_down):
        down[has_down] = x[rows[has_down] + 1]
    return up, down


def effective_interference_metrics(symbols: np.ndarray, taps: ChannelTaps,
                                   boundary: str = "zero",
                                   sigma: Optional[float] = None,
                                   chunk_tracks: int = 256) -> Dict:
    """Measure sequence-dependent down-track/cross-track interference.

    Unlike :func:`tap_energy_metrics`, this uses the actual transmitted symbol
    grid. It is chunked over tracks so diagnostic runs can sample large-ish
    grids without allocating several full-size interference arrays at once.
    """
    x = np.asarray(symbols, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("symbols must be a 2D (tracks x down-track) array")
    if boundary not in _PAD_MODE:
        raise ValueError(f"unknown boundary {boundary!r}; expected one of {sorted(_PAD_MODE)}")

    chunk = max(1, int(chunk_tracks))
    count = 0
    sums = {
        "main": 0.0,
        "down": 0.0,
        "cross": 0.0,
        "interference": 0.0,
        "noiseless": 0.0,
    }
    sums_sq = {k: 0.0 for k in sums}

    for start in range(0, x.shape[0], chunk):
        end = min(start + chunk, x.shape[0])
        center = x[start:end]
        down_prev, down_next = _down_neighbors(center, boundary)
        cross_up, cross_down = _cross_neighbors(x, start, end, boundary)

        main = taps.c0 * center
        down_i = taps.c_down_prev * down_prev + taps.c_down_next * down_next
        cross_i = taps.c_cross_up * cross_up + taps.c_cross_down * cross_down
        interference = down_i + cross_i
        noiseless = main + interference
        arrays = {
            "main": main,
            "down": down_i,
            "cross": cross_i,
            "interference": interference,
            "noiseless": noiseless,
        }
        n = int(center.size)
        count += n
        for key, arr in arrays.items():
            sums[key] += float(np.sum(arr))
            sums_sq[key] += float(np.sum(arr * arr))

    main_var = _variance(count, sums["main"], sums_sq["main"])
    down_var = _variance(count, sums["down"], sums_sq["down"])
    cross_var = _variance(count, sums["cross"], sums_sq["cross"])
    interference_var = _variance(count, sums["interference"], sums_sq["interference"])
    noiseless_var = _variance(count, sums["noiseless"], sums_sq["noiseless"])
    noise_var = None if sigma is None else float(sigma ** 2)
    sinr_den = interference_var + (noise_var or 0.0)

    return {
        "effective_sample_count": int(count),
        "effective_boundary": boundary,
        "effective_chunk_tracks": int(chunk),
        "effective_main_mean": float(sums["main"] / count) if count else 0.0,
        "effective_main_variance": main_var,
        "effective_main_rms": _rms(count, sums_sq["main"]),
        "effective_downtrack_interference_mean": float(sums["down"] / count) if count else 0.0,
        "effective_downtrack_interference_variance": down_var,
        "effective_downtrack_interference_rms": _rms(count, sums_sq["down"]),
        "effective_crosstrack_interference_mean": float(sums["cross"] / count) if count else 0.0,
        "effective_crosstrack_interference_variance": cross_var,
        "effective_crosstrack_interference_rms": _rms(count, sums_sq["cross"]),
        "effective_interference_mean": float(sums["interference"] / count) if count else 0.0,
        "effective_interference_variance": interference_var,
        "effective_interference_rms": _rms(count, sums_sq["interference"]),
        "effective_noiseless_readback_variance": noiseless_var,
        "effective_down_fraction_of_interference_variance": _safe_ratio(down_var, interference_var),
        "effective_cross_fraction_of_interference_variance": _safe_ratio(cross_var, interference_var),
        "effective_sir_db": _safe_ratio_db(main_var, interference_var),
        "effective_sinr_db": _safe_ratio_db(main_var, sinr_den),
        "effective_awgn_sigma": None if sigma is None else float(sigma),
        "effective_awgn_variance": noise_var,
    }


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

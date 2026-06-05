"""SISO channel detectors and equalizers.

The default ``soft_awgn`` detector treats every readback sample independently and
ignores ISI/ITI when forming LLRs. This module adds a heavier but more realistic
path for research sweeps:

* cancel cross-track interference using neighbouring soft-symbol estimates;
* run exact BCJR on each down-track 3-tap ISI channel
  ``y[j] = c_prev*x[j-1] + c0*x[j] + c_next*x[j+1] + n``.

The detector returns project-convention LLRs: positive means bit 1 / symbol +1.
It is exact for the down-track ISI model with ``boundary='zero'`` and approximate
for the 2D channel because ITI is cancelled from soft estimates rather than
jointly trellised over all tracks.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .channel import ChannelTaps


def _logsumexp(v: np.ndarray) -> float:
    if v.size == 0:
        return -np.inf
    m = float(np.max(v))
    if not np.isfinite(m):
        return m
    return float(m + np.log(np.exp(v - m).sum()))


def _prior_score(llr: float, symbol: float) -> float:
    if symbol == 0.0:
        return 0.0
    return 0.5 * float(llr) * float(symbol)


def bcjr_isi_track_llr(y: np.ndarray, taps: ChannelTaps, sigma: float,
                       apriori_llr: Optional[np.ndarray] = None,
                       llr_clip: Optional[float] = 30.0) -> np.ndarray:
    """BCJR LLRs for one zero-boundary down-track ISI sequence.

    State at trellis step ``j`` is ``(x[j-1], x[j])``. A transition to
    ``(x[j], x[j+1])`` emits ``y[j]``. The start/end boundary symbols are zero.
    """
    obs = np.asarray(y, dtype=np.float64)
    if obs.ndim != 1:
        raise ValueError("bcjr_isi_track_llr expects a 1D track")
    J = obs.shape[0]
    if J == 0:
        return np.zeros((0,), dtype=np.float64)
    if apriori_llr is None:
        apriori = np.zeros(J, dtype=np.float64)
    else:
        apriori = np.asarray(apriori_llr, dtype=np.float64)
        if apriori.shape != (J,):
            raise ValueError(f"apriori_llr shape {apriori.shape} != ({J},)")

    sigma_eff = max(float(sigma), 1.0e-6)
    inv_2var = 0.5 / (sigma_eff * sigma_eff)

    states = []
    for j in range(J + 1):
        if j == 0:
            states.append([(0.0, -1.0), (0.0, 1.0)])
        elif j == J:
            states.append([(-1.0, 0.0), (1.0, 0.0)])
        else:
            states.append([(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)])

    alpha = [np.full(len(s), -np.inf, dtype=np.float64) for s in states]
    beta = [np.full(len(s), -np.inf, dtype=np.float64) for s in states]
    alpha[0] = np.array([_prior_score(apriori[0], b) for _, b in states[0]], dtype=np.float64)
    beta[J].fill(0.0)

    trans = []
    for j in range(J):
        tj = []
        for si, (a, b) in enumerate(states[j]):
            for sk, (b2, c) in enumerate(states[j + 1]):
                if b != b2:
                    continue
                mu = taps.c_down_prev * a + taps.c0 * b + taps.c_down_next * c
                metric = -((obs[j] - mu) ** 2) * inv_2var
                if j + 1 < J:
                    metric += _prior_score(apriori[j + 1], c)
                tj.append((si, sk, float(metric)))
        trans.append(tj)

    for j in range(J):
        for _, sk, _ in trans[j]:
            vals = [alpha[j][si] + metric for si, sk2, metric in trans[j] if sk2 == sk]
            alpha[j + 1][sk] = _logsumexp(np.asarray(vals, dtype=np.float64))

    for j in range(J - 1, -1, -1):
        for si, _, _ in trans[j]:
            vals = [metric + beta[j + 1][sk] for si2, sk, metric in trans[j] if si2 == si]
            beta[j][si] = _logsumexp(np.asarray(vals, dtype=np.float64))

    out = np.empty(J, dtype=np.float64)
    for j in range(J):
        post = alpha[j] + beta[j]
        plus = [post[si] for si, (_, b) in enumerate(states[j]) if b > 0]
        minus = [post[si] for si, (_, b) in enumerate(states[j]) if b < 0]
        out[j] = _logsumexp(np.asarray(plus, dtype=np.float64)) - _logsumexp(np.asarray(minus, dtype=np.float64))
    if llr_clip is not None and llr_clip > 0:
        out = np.clip(out, -float(llr_clip), float(llr_clip))
    return out


def _initial_llr(y: np.ndarray, taps: ChannelTaps, sigma: float, clip: float) -> np.ndarray:
    if sigma <= 0.0:
        return np.where(y >= 0.0, clip, -clip).astype(np.float64)
    llr = 2.0 * taps.c0 * np.asarray(y, dtype=np.float64) / (sigma * sigma)
    return np.clip(llr, -clip, clip)


def _expected_from_llr(llr: np.ndarray) -> np.ndarray:
    return np.tanh(np.asarray(llr, dtype=np.float64) / 2.0)


def _cross_interference(expected: np.ndarray, taps: ChannelTaps) -> np.ndarray:
    up = np.zeros_like(expected, dtype=np.float64)
    down = np.zeros_like(expected, dtype=np.float64)
    up[1:] = expected[:-1]
    down[:-1] = expected[1:]
    return taps.c_cross_up * up + taps.c_cross_down * down


def bcjr_2d_equalized_llr(y: np.ndarray, taps: ChannelTaps, sigma: float,
                          boundary: str = "zero", iterations: int = 2,
                          llr_clip: float = 30.0) -> Tuple[np.ndarray, Dict]:
    """2D equalized channel-bit LLRs using soft ITI cancellation + BCJR ISI.

    ``iterations`` counts soft ITI-cancellation / BCJR passes. At least one pass
    is always run. Missing cross-track neighbours use the same zero boundary as
    the channel model.
    """
    if boundary != "zero":
        raise ValueError("bcjr_2d_equalized_llr currently requires boundary='zero'")
    obs = np.asarray(y, dtype=np.float64)
    if obs.ndim != 2:
        raise ValueError("bcjr_2d_equalized_llr expects a 2D grid")
    rounds = max(1, int(iterations))
    clip = float(llr_clip)

    llr = _initial_llr(obs, taps, sigma, clip)
    for _ in range(rounds):
        expected = _expected_from_llr(llr)
        z = obs - _cross_interference(expected, taps)
        next_llr = np.empty_like(llr)
        for t in range(obs.shape[0]):
            next_llr[t] = bcjr_isi_track_llr(z[t], taps, sigma=sigma, llr_clip=clip)
        llr = next_llr

    meta = {
        "channel_detector": "bcjr_2d_equalized",
        "equalizer_iterations": rounds,
        "channel_llr_clip": clip,
    }
    return llr, meta

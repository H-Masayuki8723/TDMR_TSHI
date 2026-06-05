"""1D and 2D modulation/recording constraints.

These predicates define which channel-bit patterns a constrained code is allowed
to emit. They are used by :mod:`tdmr2d.codebook` to enumerate valid codewords and
are independently unit-tested.

Two constraint flavours are provided:

* **1D down-track MTR** (``satisfies_mtr_1d``): limit the maximum run of
  consecutive *transitions* along a down-track sequence. A "transition" is an
  adjacent pair of differing bits, so the bit string ``0101`` has 3 transitions.
  Limiting the transition run suppresses the tightest down-track patterns.

* **2D-MTR on an 8x2 block** (``satisfies_mtr2d_block``): down-track MTR on each
  of the two tracks, *plus* a cross-track rule that forbids long 2x2
  checkerboard runs. The anti-diagonal checkerboard

      0 1            1 0
      1 0     and    0 1

  is the 2x2 pattern most vulnerable to inter-track interference (ITI) in TDMR;
  suppressing sustained checkerboard regions is the genuinely *2D* part of the
  constraint.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# bit helpers                                                                  #
# --------------------------------------------------------------------------- #
def bits_down(value: int, n: int) -> np.ndarray:
    """Integer -> length-``n`` bit vector in down-track order (bit ``d`` = bit ``d``)."""
    return np.array([(value >> d) & 1 for d in range(n)], dtype=np.uint8)


def block_of(value: int, cross: int, down: int) -> np.ndarray:
    """Integer -> ``(cross, down)`` block; cell ``(t, d)`` = bit ``t*down + d`` (track-major)."""
    flat = np.array([(value >> k) & 1 for k in range(cross * down)], dtype=np.uint8)
    return flat.reshape(cross, down)


# --------------------------------------------------------------------------- #
# 1D down-track MTR                                                            #
# --------------------------------------------------------------------------- #
def transitions(seq: np.ndarray) -> np.ndarray:
    """Boolean transition vector: ``True`` where adjacent bits differ."""
    seq = np.asarray(seq)
    if seq.shape[0] < 2:
        return np.zeros((0,), dtype=bool)
    return seq[1:] != seq[:-1]


def max_transition_run(seq: np.ndarray) -> int:
    """Longest run of consecutive transitions in ``seq``."""
    tr = transitions(seq).astype(np.int8)
    best = cur = 0
    for t in tr:
        cur = cur + 1 if t else 0
        if cur > best:
            best = cur
    return int(best)


def satisfies_mtr_1d(seq: np.ndarray, max_run: int) -> bool:
    """True iff the down-track transition run never exceeds ``max_run``."""
    return max_transition_run(seq) <= max_run


# --------------------------------------------------------------------------- #
# 2D-MTR on a (cross, down) block                                             #
# --------------------------------------------------------------------------- #
def _is_checker_2x2(block: np.ndarray, d: int) -> bool:
    """True iff the 2x2 window at down-track columns ``d, d+1`` is a checkerboard."""
    a = block[0, d]
    b = block[1, d]
    c = block[0, d + 1]
    e = block[1, d + 1]
    # All four orthogonally-adjacent pairs differ <=> anti-diagonal checkerboard.
    return a != b and a != c and e != b and e != c and a == e and b == c


def max_checker_run(block: np.ndarray) -> int:
    """Longest run of consecutive 2x2 checkerboard windows along down-track.

    A return value of ``r`` means a fully alternating ``2 x (r+1)`` region exists.
    Requires a block with exactly 2 tracks (cross == 2).
    """
    block = np.asarray(block)
    cross, down = block.shape
    if cross != 2 or down < 2:
        return 0
    best = cur = 0
    for d in range(down - 1):
        if _is_checker_2x2(block, d):
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)


def satisfies_mtr2d_block(block: np.ndarray, down_mtr: int, max_checker_run_allowed: int) -> bool:
    """2D-MTR validity for a ``(2, down)`` block.

    * each track satisfies down-track MTR ``<= down_mtr``;
    * no checkerboard run longer than ``max_checker_run_allowed`` (``0`` forbids
      any 2x2 checkerboard at all -> strict 2D constraint).
    """
    block = np.asarray(block)
    for t in range(block.shape[0]):
        if max_transition_run(block[t]) > down_mtr:
            return False
    if max_checker_run(block) > max_checker_run_allowed:
        return False
    return True

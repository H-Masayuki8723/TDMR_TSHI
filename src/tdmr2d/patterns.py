"""User/data bit-pattern generation.

All randomness flows through an explicit ``numpy.random.Generator`` so runs are
fully reproducible from a single seed.
"""

from __future__ import annotations

import numpy as np


def random_bit_grid(rng: np.random.Generator, num_tracks: int, bits_per_track: int) -> np.ndarray:
    """Return an i.i.d. uniform ``{0, 1}`` grid of shape ``(num_tracks, bits_per_track)``."""
    return rng.integers(0, 2, size=(num_tracks, bits_per_track), dtype=np.uint8)


def random_user_bits(rng: np.random.Generator, n: int) -> np.ndarray:
    """Return an i.i.d. uniform ``{0, 1}`` vector of length ``n``."""
    return rng.integers(0, 2, size=(n,), dtype=np.uint8)

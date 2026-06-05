"""Pre-ECC BER / block-error metrics.

BER is defined directly on channel bits (no error correction, no erasures):

    BER = bit_errors / num_bits

where ``bit_errors`` counts positions at which the hard-decided received bit
differs from the transmitted channel bit. A "block" is the ``cross x down`` tile
used for 2D coding (default 2x8 = 16 cells); a block is in error if *any* of its
bits is wrong.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def count_bit_errors(tx_bits: np.ndarray, rx_bits: np.ndarray) -> int:
    """Number of differing bit positions between transmitted and received grids."""
    tx = np.asarray(tx_bits)
    rx = np.asarray(rx_bits)
    if tx.shape != rx.shape:
        raise ValueError(f"shape mismatch: tx {tx.shape} vs rx {rx.shape}")
    return int(np.count_nonzero(tx != rx))


def ber(tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
    """Bit error rate = bit_errors / num_bits."""
    tx = np.asarray(tx_bits)
    n = tx.size
    if n == 0:
        return 0.0
    return count_bit_errors(tx, rx_bits) / n


def count_block_errors(tx_bits: np.ndarray, rx_bits: np.ndarray,
                       block_cross: int, block_down: int) -> Tuple[int, int]:
    """Return ``(block_errors, num_blocks)`` for ``block_cross x block_down`` tiles.

    The grid is ``(num_tracks, bits_per_track)``; blocks span ``block_cross``
    tracks and ``block_down`` down-track bits. The grid dimensions must tile the
    blocks exactly (enforced by config validation).
    """
    tx = np.asarray(tx_bits)
    rx = np.asarray(rx_bits)
    if tx.shape != rx.shape:
        raise ValueError(f"shape mismatch: tx {tx.shape} vs rx {rx.shape}")
    T, J = tx.shape
    if T % block_cross != 0 or J % block_down != 0:
        raise ValueError(
            f"grid {T}x{J} does not tile blocks {block_cross}x{block_down} exactly"
        )
    Tb, Jb = T // block_cross, J // block_down
    diff = (tx != rx).reshape(Tb, block_cross, Jb, block_down)
    block_err = diff.any(axis=(1, 3))
    return int(block_err.sum()), int(Tb * Jb)


def compute_metrics(tx_bits: np.ndarray, rx_bits: np.ndarray,
                    block_cross: int, block_down: int) -> Dict[str, float]:
    """Compute the full metric bundle for one run."""
    tx = np.asarray(tx_bits)
    num_bits = int(tx.size)
    bit_errors = count_bit_errors(tx, rx_bits)
    block_errors, num_blocks = count_block_errors(tx, rx_bits, block_cross, block_down)
    return {
        "num_bits": num_bits,
        "bit_errors": bit_errors,
        "BER": (bit_errors / num_bits) if num_bits else 0.0,
        "num_blocks": num_blocks,
        "block_errors": block_errors,
        "block_error_rate": (block_errors / num_blocks) if num_blocks else 0.0,
    }

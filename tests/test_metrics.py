"""Metrics tests: bit-error counting, BER, and block-error rate."""

import numpy as np

from tdmr2d.metrics import ber, compute_metrics, count_bit_errors, count_block_errors


def test_count_bit_errors():
    tx = np.zeros((4, 8), dtype=np.uint8)
    rx = tx.copy()
    rx[0, 0] = 1
    rx[3, 7] = 1
    assert count_bit_errors(tx, rx) == 2


def test_ber_value():
    tx = np.zeros((2, 8), dtype=np.uint8)  # 16 bits
    rx = tx.copy()
    rx[0, :4] = 1  # 4 bit errors
    assert ber(tx, rx) == 4 / 16


def test_block_error_rate_single_block():
    # One 2x8 block; a single flipped bit -> the whole block is in error.
    tx = np.zeros((2, 8), dtype=np.uint8)
    rx = tx.copy()
    rx[1, 5] = 1
    be, nb = count_block_errors(tx, rx, block_cross=2, block_down=8)
    assert (be, nb) == (1, 1)


def test_block_error_rate_multiple_blocks():
    # 4x16 grid -> 2x2 = 4 blocks of shape 2x8. Flip a bit in exactly two blocks.
    tx = np.zeros((4, 16), dtype=np.uint8)
    rx = tx.copy()
    rx[0, 0] = 1     # block (0,0)
    rx[3, 15] = 1    # block (1,1)
    be, nb = count_block_errors(tx, rx, block_cross=2, block_down=8)
    assert nb == 4
    assert be == 2


def test_compute_metrics_bundle():
    tx = np.zeros((2, 8), dtype=np.uint8)
    rx = tx.copy()
    rx[0, 0] = 1
    m = compute_metrics(tx, rx, block_cross=2, block_down=8)
    assert m["num_bits"] == 16
    assert m["bit_errors"] == 1
    assert m["BER"] == 1 / 16
    assert m["num_blocks"] == 1
    assert m["block_errors"] == 1
    assert m["block_error_rate"] == 1.0

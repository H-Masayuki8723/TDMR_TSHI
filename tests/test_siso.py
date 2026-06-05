"""BCJR / SISO channel detector tests."""

import numpy as np

from tdmr2d.channel import ChannelTaps, map_bits_to_symbols, readback
from tdmr2d.siso import bcjr_2d_equalized_llr, bcjr_isi_track_llr


def test_bcjr_isi_track_recovers_clean_isi_sequence():
    bits = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0], dtype=np.uint8)
    x = map_bits_to_symbols(bits)
    taps = ChannelTaps(c0=1.0, c_down_prev=0.25, c_down_next=0.15,
                       c_cross_up=0.0, c_cross_down=0.0)
    p = np.pad(x, 1, mode="constant", constant_values=0.0)
    y = taps.c_down_prev * p[:-2] + taps.c0 * p[1:-1] + taps.c_down_next * p[2:]

    llr = bcjr_isi_track_llr(y, taps, sigma=0.05, llr_clip=40.0)

    assert np.array_equal((llr >= 0.0).astype(np.uint8), bits)


def test_bcjr_2d_equalized_recovers_noiseless_grid():
    bits = np.array([
        [0, 1, 0, 1, 1, 0, 0, 1],
        [1, 1, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 1, 0, 1, 0, 1],
    ], dtype=np.uint8)
    taps = ChannelTaps(c0=1.0, c_down_prev=0.15, c_down_next=0.15,
                       c_cross_up=0.10, c_cross_down=0.10)
    y = readback(map_bits_to_symbols(bits), taps, snr_db=None, boundary="zero")

    llr, meta = bcjr_2d_equalized_llr(y, taps, sigma=0.05, iterations=2, llr_clip=40.0)

    assert meta["channel_detector"] == "bcjr_2d_equalized"
    assert np.array_equal((llr >= 0.0).astype(np.uint8), bits)


def test_bcjr_2d_equalized_requires_zero_boundary():
    raised = False
    try:
        bcjr_2d_equalized_llr(np.zeros((2, 4)), ChannelTaps(), sigma=1.0, boundary="periodic")
    except ValueError:
        raised = True
    assert raised

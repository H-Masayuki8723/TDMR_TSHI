"""Detector tests: sign->bit recovery and BER==0 on a clean channel."""

import numpy as np

from tdmr2d.channel import ChannelTaps, map_bits_to_symbols, readback
from tdmr2d.detector import HardThresholdDetector
from tdmr2d.metrics import ber


def test_threshold_sign_to_bit():
    rng = np.random.default_rng(3)
    bits = rng.integers(0, 2, size=(8, 50), dtype=np.uint8)
    symbols = map_bits_to_symbols(bits)  # 0->-1, 1->+1
    det = HardThresholdDetector(threshold=0.0)
    rx = det.detect(symbols)
    assert np.array_equal(rx, bits)


def test_pipeline_ber_zero_no_noise_no_interference():
    rng = np.random.default_rng(4)
    bits = rng.integers(0, 2, size=(16, 128), dtype=np.uint8)
    symbols = map_bits_to_symbols(bits)
    taps = ChannelTaps(c0=1.0, c_down_prev=0.0, c_down_next=0.0,
                       c_cross_up=0.0, c_cross_down=0.0)
    y = readback(symbols, taps, snr_db=None)
    rx = HardThresholdDetector(0.0).detect(y)
    assert ber(bits, rx) == 0.0


def test_pipeline_ber_zero_noiseless_with_interference():
    # |total interference| (0.5) < c0 (1.0) so the sign never flips -> BER 0.
    rng = np.random.default_rng(5)
    bits = rng.integers(0, 2, size=(16, 128), dtype=np.uint8)
    symbols = map_bits_to_symbols(bits)
    taps = ChannelTaps()  # default 0.15/0.15/0.10/0.10
    y = readback(symbols, taps, snr_db=None, boundary="zero")
    rx = HardThresholdDetector(0.0).detect(y)
    assert ber(bits, rx) == 0.0

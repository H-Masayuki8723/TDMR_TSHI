"""Channel tests: identity (no noise/interference), noise reproducibility, ITI."""

import numpy as np

from tdmr2d.channel import ChannelTaps, map_bits_to_symbols, readback, snr_db_to_sigma


def test_no_noise_no_interference_identity():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(8, 32), dtype=np.uint8)
    symbols = map_bits_to_symbols(bits)
    taps = ChannelTaps(c0=1.0, c_down_prev=0.0, c_down_next=0.0,
                       c_cross_up=0.0, c_cross_down=0.0)
    y = readback(symbols, taps, snr_db=None)  # noiseless
    # y must equal the mapped +/-1 symbols exactly.
    assert np.allclose(y, symbols)
    assert set(np.unique(y)).issubset({-1.0, 1.0})


def test_noise_reproducible_with_seed():
    bits = np.random.default_rng(1).integers(0, 2, size=(16, 64), dtype=np.uint8)
    symbols = map_bits_to_symbols(bits)
    taps = ChannelTaps()
    y1 = readback(symbols, taps, snr_db=12.0, rng=np.random.default_rng(7))
    y2 = readback(symbols, taps, snr_db=12.0, rng=np.random.default_rng(7))
    assert np.array_equal(y1, y2)  # identical noise for identical seed


def test_noise_differs_with_seed():
    bits = np.zeros((16, 64), dtype=np.uint8)
    symbols = map_bits_to_symbols(bits)
    taps = ChannelTaps()
    y1 = readback(symbols, taps, snr_db=12.0, rng=np.random.default_rng(1))
    y2 = readback(symbols, taps, snr_db=12.0, rng=np.random.default_rng(2))
    assert not np.allclose(y1, y2)


def test_iti_increases_neighbor_influence():
    # Center track is all -1 with both neighbours all +1; only cross taps active.
    x = np.array([[+1.0] * 8, [-1.0] * 8, [+1.0] * 8])
    center = 1  # middle track index
    means = []
    for iti in (0.0, 0.1, 0.2, 0.3):
        taps = ChannelTaps(c0=1.0, c_down_prev=0.0, c_down_next=0.0,
                           c_cross_up=iti, c_cross_down=iti)
        y = readback(x, taps, snr_db=None)
        means.append(float(y[center].mean()))
    # Larger ITI pulls the (-1) center track upward (toward the +1 neighbours).
    assert means[0] == -1.0
    assert means[1] < means[2] < means[3]
    assert all(means[i] < means[i + 1] for i in range(len(means) - 1))


def test_snr_to_sigma_convention():
    assert snr_db_to_sigma(None, 1.0) == 0.0
    # sigma^2 = c0^2 * 10^(-snr/10)
    assert np.isclose(snr_db_to_sigma(10.0, 1.0) ** 2, 10 ** (-1.0))
    assert np.isclose(snr_db_to_sigma(20.0, 2.0) ** 2, (2.0 ** 2) * 10 ** (-2.0))

"""LDPC code + BP decoder + LDPC evaluation-track tests."""

import numpy as np

from tdmr2d.config import Config
from tdmr2d.decoder import LDPCDecoder
from tdmr2d.experiments import run_ldpc
from tdmr2d.io import REQUIRED_COLUMNS
from tdmr2d.ldpc import LDPCCode


def _code():
    return LDPCCode.gallager(n=120, dv=3, dc=6, seed=1)


def test_gallager_regularity():
    code = _code()
    assert code.n == 120 and code.m == 60
    # Regular weights: every column has dv=3 ones, every row has dc=6 ones.
    assert np.all(code.H.sum(axis=0) == 3)
    assert np.all(code.H.sum(axis=1) == 6)
    # k = n - GF(2) rank(H); rank <= m = 60, so k >= 60 and rate >= 0.5.
    assert code.k >= 60 and code.rate >= 0.5


def test_parity_check_satisfied():
    code = _code()
    rng = np.random.default_rng(0)
    U = rng.integers(0, 2, size=(40, code.k), dtype=np.uint8)
    C = code.encode(U)
    assert np.all(code.syndrome(C) == 0)


def test_systematic_roundtrip():
    code = _code()
    rng = np.random.default_rng(1)
    U = rng.integers(0, 2, size=(40, code.k), dtype=np.uint8)
    assert np.array_equal(code.info_bits(code.encode(U)), U)


def test_noiseless_decode_recovers_codeword():
    code = _code()
    rng = np.random.default_rng(2)
    U = rng.integers(0, 2, size=(30, code.k), dtype=np.uint8)
    C = code.encode(U)
    s = 2.0 * C - 1.0
    sigma = 0.05
    y = s + rng.normal(0.0, sigma, size=s.shape)
    llr = 2.0 * y / (sigma * sigma)          # positive -> bit 1
    dec = code.decode_llr(llr, max_iters=20, method="minsum", scale=0.75)
    assert np.array_equal(dec, C)


def test_decode_with_posterior_returns_project_convention_llr():
    code = _code()
    rng = np.random.default_rng(22)
    U = rng.integers(0, 2, size=(8, code.k), dtype=np.uint8)
    C = code.encode(U)
    llr = np.where(C == 1, 12.0, -12.0)

    dec, posterior = code.decode_llr_with_posterior(llr, max_iters=10)

    assert np.array_equal(dec, C)
    assert posterior.shape == llr.shape
    assert np.array_equal((posterior >= 0.0).astype(np.uint8), C)


def test_decode_provides_coding_gain():
    code = LDPCCode.gallager(n=300, dv=3, dc=6, seed=1)
    rng = np.random.default_rng(3)
    U = rng.integers(0, 2, size=(120, code.k), dtype=np.uint8)
    C = code.encode(U)
    s = 2.0 * C - 1.0
    sigma = 0.70
    y = s + rng.normal(0.0, sigma, size=s.shape)
    raw_ber = (( y >= 0).astype(np.uint8) != C).mean()
    llr = 2.0 * y / (sigma * sigma)
    dec = code.decode_llr(llr, max_iters=30, method="minsum", scale=0.75)
    post_ber = (code.info_bits(dec) != U).mean()
    assert post_ber < raw_ber / 5.0          # substantial gain


def test_ldpc_decoder_wrapper():
    code = _code()
    rng = np.random.default_rng(4)
    C = code.encode(rng.integers(0, 2, size=(5, code.k), dtype=np.uint8))
    llr = 2.0 * (2.0 * C - 1.0) / (0.1 ** 2)  # near-certain LLRs
    dec = LDPCDecoder(code=code, max_iters=10).decode(llr)
    assert np.array_equal(dec, C)
    # Without a code, the decoder is a redirect stub.
    raised = False
    try:
        LDPCDecoder().decode(llr)
    except NotImplementedError:
        raised = True
    assert raised


def _ldpc_cfg(snr=7.0, iti=0.0):
    return Config.from_dict({
        "experiment": {"name": "ldpc_t", "family": "uncoded", "seed": 0},
        "code": {"type": "uncoded", "block_shape": [8, 2]},
        "channel": {"snr_db": snr, "boundary": "zero"},
        "detector": {"type": "soft_awgn", "threshold": 0.0, "llr_clip": 20.0},
        "metrics": {"target_ber": 1.0e-2},
        "output": {"dir": "outputs/runs"},
        "sweep": {"snr_db": [snr], "iti_coeffs": [iti]},
    })


def test_run_ldpc_schema_and_gain():
    rows, meta = run_ldpc(_ldpc_cfg(snr=7.0, iti=0.0),
                          {"n": 120, "dv": 3, "dc": 6, "num_frames": 40, "max_iters": 20})
    assert len(rows) == 1
    r = rows[0]
    for col in REQUIRED_COLUMNS:
        assert col in r
    assert r["family"] == "ldpc"
    assert "pre_ecc_ber" in r
    # At a decent SNR the decoder should not be worse than the raw channel.
    assert r["BER"] <= r["pre_ecc_ber"] + 1e-12


def test_run_ldpc_requires_soft_detector():
    cfg = Config.from_dict({
        "experiment": {"family": "uncoded"},
        "code": {"type": "uncoded"},
        "detector": {"type": "hard_threshold"},
        "sweep": {"snr_db": [7], "iti_coeffs": [0.0]},
    })
    raised = False
    try:
        run_ldpc(cfg, {"n": 120, "num_frames": 10})
    except ValueError:
        raised = True
    assert raised

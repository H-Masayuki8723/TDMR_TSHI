"""Soft-decision / LLR + decoder-interface tests (stage 2)."""

import numpy as np

from tdmr2d.channel import map_bits_to_symbols
from tdmr2d.config import Config, ConfigError
from tdmr2d.decoder import HardDecisionDecoder, LDPCDecoder, make_decoder
from tdmr2d.detector import HardThresholdDetector, SoftAWGNDetector
from tdmr2d.experiments import run_single


def _soft_cfg(tmp_cache, decoder_type="hard", snr_db=12.0):
    return Config.from_dict({
        "experiment": {"name": "soft", "family": "uncoded", "seed": 0,
                       "num_tracks": 8, "bits_per_track": 64},
        "code": {"type": "uncoded", "block_shape": [8, 2]},
        "channel": {"snr_db": snr_db, "boundary": "zero"},
        "detector": {"type": "soft_awgn", "threshold": 0.0, "llr_clip": 20.0},
        "decoder": {"type": decoder_type, "params": {}},
        "metrics": {"target_ber": 1.0e-2},
        "output": {"dir": "outputs/runs"},
    })


def test_llr_sign_matches_hard_decision():
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 1.0, size=(4, 16))
    det = SoftAWGNDetector(threshold=0.0, llr_clip=50.0)
    llr = det.soft_llr(y, sigma=0.5, amplitude=1.0)
    # Positive LLR <=> hard decision is bit 1.
    assert np.array_equal((llr >= 0).astype(np.uint8), det.detect(y))


def test_llr_formula_no_clip():
    # LLR = 2 * a * y / sigma^2 ; choose values that stay within the clip.
    det = SoftAWGNDetector(llr_clip=100.0)
    y = np.array([[0.4, -0.8, 0.1, -0.2]])
    llr = det.soft_llr(y, sigma=2.0, amplitude=1.0)  # = 2*y/4 = y/2
    assert np.allclose(llr, y / 2.0)


def test_llr_clipping():
    det = SoftAWGNDetector(llr_clip=5.0)
    y = np.array([[10.0, -10.0]])
    llr = det.soft_llr(y, sigma=0.5, amplitude=1.0)  # huge -> clipped
    assert llr.max() <= 5.0 and llr.min() >= -5.0
    assert np.array_equal(np.abs(llr), np.array([[5.0, 5.0]]))


def test_llr_noiseless_saturates():
    det = SoftAWGNDetector(llr_clip=20.0)
    y = np.array([[1.0, -1.0, 0.5]])
    llr = det.soft_llr(y, sigma=0.0, amplitude=1.0)
    assert np.array_equal(llr, np.array([[20.0, -20.0, 20.0]]))


def test_hard_detector_has_no_soft_output():
    det = HardThresholdDetector()
    raised = False
    try:
        det.soft_llr(np.zeros((2, 2)), sigma=1.0)
    except NotImplementedError:
        raised = True
    assert raised


def test_hard_decision_decoder_identity():
    dec = HardDecisionDecoder()
    llr = np.array([[3.0, -1.0, 0.0, -5.0]])
    assert np.array_equal(dec.decode(llr), np.array([[1, 0, 1, 0]], dtype=np.uint8))


def test_make_decoder_variants():
    assert make_decoder(None) is None
    from tdmr2d.config import DecoderConfig
    assert make_decoder(DecoderConfig(type="none")) is None
    assert isinstance(make_decoder(DecoderConfig(type="hard")), HardDecisionDecoder)
    assert isinstance(make_decoder(DecoderConfig(type="ldpc")), LDPCDecoder)


def test_ldpc_decoder_is_stub():
    raised = False
    try:
        LDPCDecoder().decode(np.zeros((2, 8)))
    except NotImplementedError:
        raised = True
    assert raised


def test_decoder_requires_soft_detector():
    raised = False
    try:
        Config.from_dict({
            "experiment": {"family": "uncoded", "num_tracks": 8, "bits_per_track": 64},
            "code": {"type": "uncoded"},
            "detector": {"type": "hard_threshold"},
            "decoder": {"type": "hard"},
        })
    except ConfigError:
        raised = True
    assert raised


def test_soft_run_post_decode_equals_pre_ecc(tmp_path):
    row = run_single(_soft_cfg(tmp_path / "cb"), cache_dir=str(tmp_path / "cb"))
    assert row["detector"] == "soft_awgn"
    assert "mean_abs_llr" in row and row["mean_abs_llr"] >= 0.0
    # Hard passthrough decoder -> post-decode BER must equal pre-ECC BER.
    assert row["decoder"] == "hard"
    assert row["post_decode_BER"] == row["BER"]

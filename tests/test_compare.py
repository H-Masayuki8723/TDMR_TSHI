"""Multi-family comparison tests."""

from tdmr2d.experiments import run_compare
from tdmr2d.io import REQUIRED_COLUMNS


def _raw(tmp_cache):
    return {
        "experiment": {"name": "cmp", "family": "uncoded", "seed": 0,
                       "num_tracks": 8, "bits_per_track": 64},
        "code": {"type": "uncoded", "block_shape": [8, 2]},
        "channel": {"snr_db": 12, "boundary": "zero"},
        "detector": {"type": "hard_threshold", "threshold": 0.0},
        "metrics": {"target_ber": 1.0e-2},
        "output": {"dir": "outputs/runs"},
        "sweep": {"snr_db": [10, 14], "iti_coeffs": [0.0, 0.2]},
        "compare": {
            "families": [
                {"family": "uncoded", "code": {"type": "uncoded"}},
                {"family": "mtr1d", "code": {"type": "mtr1d", "K1": 7, "down_mtr": 3}},
                {"family": "mtr2d_8x2", "code": {"type": "mtr2d_8x2", "K": 14,
                                                 "down_mtr": 3, "max_checker_run": 0}},
            ],
            "slice_snr_db": 14,
            "slice_iti": 0.0,
        },
    }


def test_run_compare_three_families(tmp_path):
    rows, meta = run_compare(_raw(tmp_path / "cb"), cache_dir=str(tmp_path / "cb"))
    families = {r["family"] for r in rows}
    assert families == {"uncoded", "mtr1d", "mtr2d_8x2"}
    # 3 families x (2 snr x 2 iti) = 12 rows.
    assert len(rows) == 12
    assert meta["slice_snr_db"] == 14 and meta["slice_iti"] == 0.0
    for r in rows:
        for col in REQUIRED_COLUMNS:
            assert col in r


def test_compare_rates(tmp_path):
    rows, _ = run_compare(_raw(tmp_path / "cb"), cache_dir=str(tmp_path / "cb"))
    rate = {r["family"]: r["rate"] for r in rows}
    assert rate["uncoded"] == 1.0
    assert abs(rate["mtr1d"] - 0.875) < 1e-9
    assert abs(rate["mtr2d_8x2"] - 0.875) < 1e-9


def test_compare_requires_families(tmp_path):
    raw = _raw(tmp_path / "cb")
    del raw["compare"]["families"]
    raised = False
    try:
        run_compare(raw, cache_dir=str(tmp_path / "cb"))
    except (ValueError, KeyError):
        raised = True
    assert raised

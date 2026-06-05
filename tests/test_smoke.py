"""Smoke / integration tests: CLI self-check, output files, codebooks, schema."""

import numpy as np
import pandas as pd

from tdmr2d import cli, codebook
from tdmr2d.config import Config
from tdmr2d.experiments import run_single
from tdmr2d.io import REQUIRED_COLUMNS


def _cfg(family, tmp_cache, **over):
    base = {
        "experiment": {"name": f"t_{family}", "family": family, "seed": 0,
                       "num_tracks": 8, "bits_per_track": 64},
        "code": {"type": family, "block_shape": [8, 2]},
        "channel": {"snr_db": over.pop("snr_db", 12.0), "boundary": "zero"},
        "detector": {"type": "hard_threshold", "threshold": 0.0},
        "metrics": {"target_ber": 1.0e-2},
        "output": {"dir": "outputs/runs"},
    }
    if family == "mtr1d":
        base["code"].update(K1=7, down_mtr=3)
    if family == "mtr2d_8x2":
        base["code"].update(K=14, down_mtr=3, max_checker_run=0)
    base["code"].update(over)
    return Config.from_dict(base)


def test_smoke_cli_creates_outputs(tmp_path):
    rc = cli.main(["smoke", "--output-root", str(tmp_path),
                   "--cache-dir", str(tmp_path / "cb")])
    assert rc == 0
    csvs = list((tmp_path / "runs").rglob("results.csv"))
    jsons = list((tmp_path / "runs").rglob("results.json"))
    assert csvs and jsons
    df = pd.read_csv(csvs[0])
    for col in REQUIRED_COLUMNS:
        assert col in df.columns


def test_run_single_reproducible(tmp_path):
    cfg = _cfg("uncoded", tmp_path / "cb")
    a = run_single(cfg, cache_dir=str(tmp_path / "cb"))
    b = run_single(cfg, cache_dir=str(tmp_path / "cb"))
    assert a["bit_errors"] == b["bit_errors"]
    assert a["BER"] == b["BER"]


def test_each_family_has_required_columns(tmp_path):
    for fam in ("uncoded", "mtr1d", "mtr2d_8x2"):
        row = run_single(_cfg(fam, tmp_path / "cb"), cache_dir=str(tmp_path / "cb"))
        for col in REQUIRED_COLUMNS:
            assert col in row, f"{fam} missing {col}"


def test_codebook_rates(tmp_path):
    cb1 = codebook.get_1d_codebook(K1=7, length=8, down_mtr=3, cache_dir=str(tmp_path / "cb"))
    assert cb1.size == 128 and abs(cb1.rate - 0.875) < 1e-9

    cb14 = codebook.get_2d_codebook(K=14, down_mtr=3, max_checker_run=0,
                                    cache_dir=str(tmp_path / "cb"))
    assert cb14.size == 16384 and abs(cb14.rate - 0.875) < 1e-9

    cb15 = codebook.get_2d_codebook(K=15, down_mtr=3, max_checker_run=1,
                                    cache_dir=str(tmp_path / "cb"))
    assert cb15.size == 32768 and abs(cb15.rate - 0.9375) < 1e-9


def test_codewords_satisfy_constraints(tmp_path):
    from tdmr2d import constraints as C
    cb = codebook.get_2d_codebook(K=14, down_mtr=3, max_checker_run=0,
                                  cache_dir=str(tmp_path / "cb"))
    # Spot-check a sample of codewords against the 2D-MTR predicate.
    rng = np.random.default_rng(0)
    for k in rng.integers(0, cb.size, size=200):
        block = cb.words[k].reshape(cb.cross, cb.down)
        assert C.satisfies_mtr2d_block(block, down_mtr=3, max_checker_run_allowed=0)


def test_noiseless_run_is_error_free(tmp_path):
    cfg = _cfg("uncoded", tmp_path / "cb", snr_db=None)
    row = run_single(cfg, cache_dir=str(tmp_path / "cb"))
    assert row["BER"] == 0.0

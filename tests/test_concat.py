"""Concatenated LDPC + constrained-code path tests."""

import numpy as np

from tdmr2d import codebook
from tdmr2d.codebook import Codebook
from tdmr2d.concat import (decode_inner_llr, encode_inner_grid, indices_to_bits,
                           soft_demapper_llr, trellis_pruned_demapper_llr,
                           trellis_pruned_viterbi_indices)
from tdmr2d.config import Config
from tdmr2d.experiments import (run_boundary_scan, run_channel_metrics,
                                run_concatenated, run_iti_calibration,
                                run_rate_plan)


def _cfg() -> Config:
    return Config.from_dict({
        "experiment": {
            "name": "concat_test",
            "family": "mtr2d_8x2",
            "seed": 0,
            "num_tracks": 4,
            "bits_per_track": 64,
        },
        "code": {
            "type": "mtr2d_8x2",
            "block_shape": [8, 2],
            "K": 14,
            "down_mtr": 3,
            "max_checker_run": 0,
        },
        "channel": {
            "model": "linear_2d_awgn",
            "c0": 1.0,
            "c_down_prev": 0.0,
            "c_down_next": 0.0,
            "c_cross_up": 0.0,
            "c_cross_down": 0.0,
            "snr_db": None,
            "boundary": "zero",
        },
        "detector": {"type": "soft_awgn", "threshold": 0.0, "llr_clip": 20.0},
        "metrics": {"target_ber": 1.0e-2},
        "output": {"dir": "outputs/runs"},
    })


def _cfg_stateful_k13() -> Config:
    raw = _cfg().to_dict()
    raw["code"].update({
        "K": 13,
        "inner_encoder": "stateful_trellis",
        "trellis_candidate_K": 14,
        "boundary_down_mtr": 6,
        "boundary_max_checker_run": 1,
    })
    return Config.from_dict(raw)


def test_soft_demapper_recovers_codebook_index_bits(tmp_path):
    cb = codebook.get_1d_codebook(K1=7, length=8, down_mtr=3, cache_dir=tmp_path)
    idx = np.array([0, 1, 5, 127])
    words = cb.encode_indices(idx)
    llr = np.where(words == 1, 20.0, -20.0)

    out = soft_demapper_llr(llr, cb)
    assert np.array_equal((out >= 0).astype(np.uint8), indices_to_bits(idx, cb.K))


def test_soft_demapper_apriori_is_removed_from_extrinsic(tmp_path):
    cb = codebook.get_1d_codebook(K1=7, length=8, down_mtr=3, cache_dir=tmp_path)
    channel = np.zeros((2, cb.block_len))
    apriori = np.array([
        [2.0, -1.0, 0.5, -0.5, 1.5, -2.0, 0.25],
        [-0.25, 2.0, -1.5, 1.0, -0.5, 0.75, -2.5],
    ])

    posterior = soft_demapper_llr(channel, cb, apriori_llr=apriori, extrinsic=False)
    extrinsic = soft_demapper_llr(channel, cb, apriori_llr=apriori, extrinsic=True)

    assert np.allclose(posterior, apriori)
    assert np.allclose(extrinsic, 0.0)


def test_trellis_pruned_demapper_cuts_impossible_boundary_transition():
    cb = Codebook(
        name="toy_mtr1d_L3_K1",
        kind="mtr1d",
        K=1,
        block_len=3,
        cross=1,
        down=3,
        words=np.array([[0, 1, 0], [1, 0, 1]], dtype=np.uint8),
    )
    llr = np.array([
        [-10.0, 10.0, -10.0],  # strongly favours row 0 -> bit 0
        [4.0, -4.0, 4.0],      # independently favours row 1 -> bit 1
    ])

    blockwise = soft_demapper_llr(llr, cb, extrinsic=False)
    pruned, meta = trellis_pruned_demapper_llr(
        llr, cb, down_mtr=2, apriori_llr=None, extrinsic=False,
    )
    path = trellis_pruned_viterbi_indices(llr, cb, down_mtr=2)

    assert blockwise[1, 0] > 0.0
    assert pruned.shape == (1, 2, 1)
    assert pruned[0, 1, 0] < 0.0
    assert np.array_equal(path, np.array([0, 0]))
    assert meta["inner_demapper"] == "trellis_pruned"


def test_inner_2dmtr_encode_decode_roundtrip_with_strong_llr(tmp_path):
    cfg = _cfg()
    rng = np.random.default_rng(0)
    outer = rng.integers(0, 2, size=(4, 60), dtype=np.uint8)

    grid, meta = encode_inner_grid(outer, cfg, cache_dir=tmp_path)
    llr_grid = np.where(grid == 1, 20.0, -20.0)
    llr_outer, demap = decode_inner_llr(llr_grid, cfg, outer.shape, cache_dir=tmp_path)

    assert meta["inner_code"].startswith("mtr2d_2x8_K14")
    assert demap["inner_demapper"] == "exact_codebook"
    assert np.array_equal((llr_outer >= 0).astype(np.uint8), outer)


def test_inner_2dmtr_accepts_trellis_pruned_demapper(tmp_path):
    cfg = _cfg()
    rng = np.random.default_rng(1)
    outer = rng.integers(0, 2, size=(4, 60), dtype=np.uint8)

    grid, _ = encode_inner_grid(outer, cfg, cache_dir=tmp_path)
    llr_grid = np.where(grid == 1, 20.0, -20.0)
    llr_outer, demap = decode_inner_llr(
        llr_grid, cfg, outer.shape, cache_dir=tmp_path,
        inner_demapper="trellis_pruned",
    )

    assert demap["inner_demapper"] == "trellis_pruned"
    assert "trellis_state_transition_density" in demap
    assert llr_outer.shape == outer.shape


def test_inner_2dmtr_accepts_hard_codebook_demapper(tmp_path):
    cfg = _cfg()
    rng = np.random.default_rng(4)
    outer = rng.integers(0, 2, size=(4, 60), dtype=np.uint8)

    grid, _ = encode_inner_grid(outer, cfg, cache_dir=tmp_path)
    llr_grid = np.where(grid == 1, 20.0, -20.0)
    llr_outer, demap = decode_inner_llr(
        llr_grid, cfg, outer.shape, cache_dir=tmp_path,
        inner_demapper="hard_codebook",
    )

    assert demap["inner_demapper"] == "hard_codebook"
    assert np.array_equal((llr_outer >= 0).astype(np.uint8), outer)


def test_stateful_k13_encoder_is_matched_to_boundary_trellis(tmp_path):
    from tdmr2d.concat import inner_transition_stats

    cfg = _cfg_stateful_k13()
    rng = np.random.default_rng(2)
    outer = rng.integers(0, 2, size=(4, 60), dtype=np.uint8)

    grid, meta = encode_inner_grid(outer, cfg, cache_dir=tmp_path)
    stats = inner_transition_stats(grid, cfg, cache_dir=tmp_path)
    llr_grid = np.where(grid == 1, 20.0, -20.0)
    llr_outer, demap = decode_inner_llr(
        llr_grid, cfg, outer.shape, cache_dir=tmp_path,
        inner_demapper="stateful_trellis",
    )

    assert meta["inner_encoder"] == "stateful_trellis"
    assert meta["stateful_info_bits"] == 13
    assert abs(meta["inner_rate"] - 13 / 16) < 1e-12
    assert stats["tx_trellis_transition_violations"] == 0
    assert demap["inner_demapper"] == "stateful_trellis"
    assert np.array_equal((llr_outer >= 0).astype(np.uint8), outer)


def test_concatenated_noiseless_2dmtr_ldpc_is_error_free(tmp_path):
    cfg = _cfg()
    rows, meta = run_concatenated(
        cfg,
        {
            "n": 60,
            "dv": 3,
            "dc": 6,
            "num_frames": 4,
            "max_iters": 10,
            "turbo_iterations": 2,
        },
        cache_dir=tmp_path,
    )

    assert meta["inner_family"] == "mtr2d_8x2"
    assert meta["turbo_iterations"] == 2
    assert len(rows) == 1
    assert rows[0]["turbo_iterations"] == 2
    assert rows[0]["inner_channel_ber"] == 0.0
    assert rows[0]["pre_ecc_ber"] == 0.0
    assert rows[0]["final_ldpc_input_ber"] == 0.0
    assert rows[0]["BER"] == 0.0
    assert rows[0]["block_error_rate"] == 0.0
    assert "tx_trellis_transition_violation_rate" in rows[0]


def test_concatenated_reports_sector_fsr_with_auto_frame_count(tmp_path):
    cfg = _cfg()
    rows, meta = run_concatenated(
        cfg,
        {
            "n": 60,
            "dv": 3,
            "dc": 6,
            "sector_bits": 64,
            "sector_count_target": 2,
            "max_iters": 10,
        },
        cache_dir=tmp_path,
    )

    assert meta["sector_bits"] == 64
    assert meta["sector_count_target"] == 2
    assert meta["sector_count_observed"] >= 2
    assert rows[0]["sector_count"] >= 2
    assert rows[0]["sector_errors"] == 0
    assert rows[0]["FSR"] == 0.0


def test_concatenated_accepts_bcjr_channel_detector(tmp_path):
    cfg = _cfg()
    rows, _ = run_concatenated(
        cfg,
        {
            "n": 60,
            "dv": 3,
            "dc": 6,
            "num_frames": 4,
            "max_iters": 10,
            "turbo_iterations": 1,
            "channel_detector": "bcjr_2d_equalized",
            "equalizer_iterations": 1,
            "channel_llr_clip": 30.0,
        },
        cache_dir=tmp_path,
    )

    assert rows[0]["channel_detector"] == "bcjr_2d_equalized"
    assert rows[0]["equalizer_iterations"] == 1
    assert rows[0]["inner_channel_ber"] == 0.0
    assert rows[0]["BER"] == 0.0


def test_boundary_scan_reports_density_and_tx_violations(tmp_path):
    cfg = _cfg()
    rows, meta = run_boundary_scan(
        cfg,
        {"n": 60, "dv": 3, "dc": 6, "num_frames": 4},
        boundary_down_values=[3, 8],
        boundary_checker_values=[0, 8],
        num_trials=1,
        cache_dir=tmp_path,
    )

    assert meta["family"] == "mtr2d_8x2"
    assert len(rows) == 4
    for row in rows:
        assert "tx_trellis_transition_violation_rate" in row
        assert "trellis_state_transition_density" in row
    relaxed = [r for r in rows if r["boundary_down_mtr"] == 8 and r["boundary_max_checker_run"] == 8][0]
    assert relaxed["tx_trellis_transition_violations"] == 0
    assert relaxed["structurally_no_pruning"] is True


def test_rate_plan_estimates_k15_high_rate_without_running_channel(tmp_path):
    raw = _cfg().to_dict()
    raw["code"].update({"K": 15, "max_checker_run": 1})
    raw["ldpc"] = {
        "n": 1800,
        "dv": 3,
        "dc": 75,
        "sector_bits": 32768,
        "sector_count_target": 10,
    }
    raw["rate_plan"] = {
        "target_total_rate": 0.90,
        "sector_bits": 32768,
        "sector_count_target": 10,
        "check_codebook": False,
    }

    rows, meta = run_rate_plan(raw, cache_dir=tmp_path)

    assert meta["required_outer_ldpc_rate"] < 1.0
    assert abs(rows[0]["inner_rate"] - 15 / 16) < 1e-12
    assert abs(rows[0]["design_total_rate"] - 0.9) < 1e-12
    assert rows[0]["recommended_num_frames"] % 2 == 0


def test_rate_plan_accepts_1dmtr_k15_high_rate_baseline(tmp_path):
    raw = _cfg().to_dict()
    raw["experiment"].update({"family": "mtr1d"})
    raw["code"] = {
        "type": "mtr1d",
        "block_shape": [16, 1],
        "K1": 15,
        "down_mtr": 3,
        "boundary_down_mtr": 8,
        "boundary_max_checker_run": 8,
    }
    raw["ldpc"] = {
        "n": 1800,
        "dv": 3,
        "dc": 75,
        "sector_bits": 32768,
        "sector_count_target": 10,
    }
    raw["rate_plan"] = {
        "target_total_rate": 0.90,
        "sector_bits": 32768,
        "sector_count_target": 10,
        "check_codebook": True,
    }

    rows, meta = run_rate_plan(raw, cache_dir=tmp_path)

    assert abs(meta["inner_rate"] - 15 / 16) < 1e-12
    assert rows[0]["codebook_status"] == "ok"
    assert rows[0]["codebook_num_valid"] >= 2 ** 15
    assert abs(rows[0]["design_total_rate"] - 0.9) < 1e-12


def test_channel_metrics_reports_effective_interference_for_ldpc_inner_grid(tmp_path):
    raw = _cfg().to_dict()
    raw["experiment"]["name"] = "channel_metrics_test"
    raw["channel"].update({"c_cross_up": 0.2, "c_cross_down": 0.2, "snr_db": 12.0})
    raw["ldpc"] = {"n": 60, "dv": 3, "dc": 6, "construction_seed": 1}
    raw["channel_metrics"] = {"use_ldpc_inner_grid": True, "sample_frames": 4, "chunk_tracks": 2}
    raw["sweep"] = {"snr_db": [12.0], "iti_coeffs": [0.0, 0.2]}

    rows, meta = run_channel_metrics(raw, cache_dir=tmp_path, chunk_tracks=2)

    assert len(rows) == 2
    assert meta["use_ldpc_inner_grid"] is True
    assert rows[0]["tap_crosstrack_energy"] == 0.0
    assert rows[1]["tap_crosstrack_energy"] > 0.0
    assert "effective_interference_variance" in rows[1]


def test_iti_calibration_marks_one_best_per_snr(tmp_path):
    raw = {
        "experiment": {"name": "iti_cal_t", "family": "uncoded", "seed": 0,
                       "num_tracks": 4, "bits_per_track": 64},
        "code": {"type": "uncoded", "block_shape": [8, 2]},
        "channel": {"model": "linear_2d_awgn", "snr_db": 12.0,
                    "c_cross_up": 0.0, "c_cross_down": 0.0},
        "detector": {"type": "hard_threshold", "threshold": 0.0},
        "metrics": {"target_ber": 1.0e-2},
        "output": {"dir": "outputs/runs"},
        "iti_calibration": {
            "target_ber": 1.0e-2,
            "snr_db": [12.0],
            "iti_coeffs": [0.0, 0.2],
            "families": [{"label": "uncoded_ref", "family": "uncoded",
                          "code": {"type": "uncoded"}}],
        },
    }

    rows, best, meta = run_iti_calibration(raw, cache_dir=tmp_path)

    assert meta["target_ber"] == 1.0e-2
    assert len(rows) == 2
    assert len(best) == 1
    assert sum(1 for r in rows if r["calibration_best_match"]) == 1

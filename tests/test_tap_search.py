import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tdmr2d import tap_search as search
from tdmr2d.config import Config
from tdmr2d.experiments import run_concatenated


PLAN = Path(__file__).resolve().parents[1] / "configs/phase2d_trueiti035_tap_search.yaml"


@pytest.fixture
def plan():
    result = search.load_plan(PLAN)
    result.update(chunks=5, snr_db=[20.0, 21.0], detector_taps=[.08, .10, .35],
                  bootstrap_replicates=200)
    result["ldpc"]["detector_iti_coeffs"] = result["detector_taps"]
    return result


def fake_row(plan, chunk, si, ti, errors=0):
    tap = plan["detector_taps"][ti]
    return dict(seed=search.point_seed(plan, chunk, si),
                snr_db=plan["snr_db"][si], iti_coeff=plan["true_iti"],
                detector_iti_coeff=tap, sector_bits=32768,
                sector_count=plan["sectors_per_chunk"], sector_errors=int(errors),
                FSR=errors / plan["sectors_per_chunk"], rate=1730/1920,
                channel_detector="bcjr_2d_equalized", n=1800, k=1730,
                equalizer_iterations=plan["ldpc"]["equalizer_iterations"],
                configured_inner_demapper="hard_codebook", turbo_iterations=0,
                true_tap_c_cross_up=.35, true_tap_c_cross_down=.35,
                detector_tap_c_cross_up=tap, detector_tap_c_cross_down=tap,
                runtime_sec=.1)


def put(path, manifest, chunk, si, ti, errors=0):
    search.atomic_json(search.point_path(path, chunk, si, ti),
                       {"fingerprint": manifest["fingerprint"], "point": [chunk, si, ti],
                        "row": fake_row(manifest["plan"], chunk, si, ti, errors)})


def test_full_plan_is_balanced_highrate_and_memory_bounded():
    plan = search.load_plan(PLAN)
    assert len(plan["detector_taps"]) == 14
    assert len(plan["snr_db"]) == 7
    assert plan["sectors_per_chunk"] == 100
    assert plan["chunks"] * plan["sectors_per_chunk"] == 10000
    assert plan["ldpc"]["sector_count_target"] == 100
    assert plan["reference_tap"] == .10
    assert .35 in plan["detector_taps"]


def test_seed_depends_on_coordinates_not_tap_order(plan):
    value = search.point_seed(plan, 0, 0)
    shuffled = copy.deepcopy(plan)
    shuffled["detector_taps"].reverse()
    assert value == search.point_seed(shuffled, 0, 0)
    assert value != search.point_seed(plan, 0, 1)
    assert value != search.point_seed(plan, 1, 0)
    shuffled["snr_db"].reverse()
    assert value == search.point_seed(shuffled, 0, 1)


def test_prepare_rejects_changed_plan_or_code(tmp_path, plan, monkeypatch):
    manifest = search.prepare(tmp_path, plan)
    assert search.prepare(tmp_path, plan) == manifest
    changed = copy.deepcopy(plan)
    changed["chunks"] += 1
    with pytest.raises(ValueError, match="changed"):
        search.prepare(tmp_path, changed)
    monkeypatch.setattr(search, "implementation_hash", lambda: "different")
    with pytest.raises(ValueError, match="changed"):
        search.prepare(tmp_path, plan)


def test_resume_checks_identity_counts_and_never_trusts_name_alone(tmp_path, plan):
    manifest = search.prepare(tmp_path, plan)
    put(tmp_path, manifest, 0, 0, 0)
    assert search.missing_taps(tmp_path, manifest, 0, 0) == [1, 2]
    path = search.point_path(tmp_path, 0, 0, 0)
    record = json.loads(path.read_text())
    record["row"]["sector_count"] = 99
    search.atomic_json(path, record)
    with pytest.raises(ValueError, match="sector_count"):
        search.missing_taps(tmp_path, manifest, 0, 0)
    put(tmp_path, manifest, 0, 0, 0)
    record = json.loads(path.read_text())
    record["fingerprint"] = "wrong"
    search.atomic_json(path, record)
    with pytest.raises(ValueError, match="identity"):
        search.missing_taps(tmp_path, manifest, 0, 0)


def test_interrupted_job_replays_only_missing_taps_with_same_seed(tmp_path, plan, monkeypatch):
    manifest = search.prepare(tmp_path, plan)
    calls = []

    def interrupted(cfg, params, **kwargs):
        calls.append((cfg.experiment.seed, params["detector_iti_coeffs"]))
        kwargs["result_callback"](fake_row(plan, 0, 0, 0))
        raise KeyboardInterrupt

    monkeypatch.setattr(search, "run_concatenated", interrupted)
    with pytest.raises(KeyboardInterrupt):
        search.run_job(tmp_path, manifest, 0, 0)
    assert search.missing_taps(tmp_path, manifest, 0, 0) == [1, 2]

    def resumed(cfg, params, **kwargs):
        calls.append((cfg.experiment.seed, params["detector_iti_coeffs"]))
        for ti in [1, 2]:
            kwargs["result_callback"](fake_row(plan, 0, 0, ti))
        return [], {}

    monkeypatch.setattr(search, "run_concatenated", resumed)
    search.run_job(tmp_path, manifest, 0, 0)
    assert calls[0][0] == calls[1][0]
    assert calls[1][1] == [.10, .35]
    assert search.missing_taps(tmp_path, manifest, 0, 0) == []
    search.run_job(tmp_path, manifest, 0, 0)
    assert len(calls) == 2


def test_summary_ignores_partial_rounds_and_preserves_zero_counts(tmp_path, plan):
    plan["chunks"] = 6
    manifest = search.prepare(tmp_path, plan)
    for chunk in range(5):
        for si in range(2):
            for ti in range(3):
                errors = 0 if ti == 0 else (chunk + 1) * ti
                put(tmp_path, manifest, chunk, si, ti, errors)
    put(tmp_path, manifest, 5, 0, 0, 99)
    status = search.summarize(tmp_path, manifest)
    assert status["saved_points"] == 31
    assert status["complete_chunks"] == 5
    assert not status["complete"]
    agg = pd.read_csv(tmp_path / "summaries/aggregate.csv")
    assert len(agg) == 6
    assert (agg.sector_count == 500).all()
    zeros = agg[agg.detector_iti_coeff == .08]
    assert (zeros.FSR == 0).all()
    assert (zeros.zero_fail_upper95_iid_exact > 0).all()
    pairs = pd.read_csv(tmp_path / "summaries/paired_vs_reference.csv")
    better = pairs[pairs.detector_iti_coeff == .08]
    assert (better.paired_chunk_bootstrap95_high < 0).all()
    reference = pairs[pairs.detector_iti_coeff == .10]
    assert (reference.delta_FSR_vs_reference == 0).all()
    assert reference.degenerate_chunk_differences.all()


def test_zero_failure_bracket_is_not_a_log_extrapolation():
    g = pd.DataFrame({"snr_db": [20., 21.], "FSR": [.05, 0.], "sector_errors": [5, 0]})
    result = search.observed_crossing(g, .001)
    assert result["status"] == "empirical_bracket_only"
    assert result["bracket_low_snr_db"] == 20
    assert np.isnan(result["interpolated_snr_db"])
    g["FSR"] = [.1, .01]
    g["sector_errors"] = [100, 10]
    result = search.observed_crossing(g, np.sqrt(.001))
    assert result["interpolated_snr_db"] == pytest.approx(20.5)


def test_os_lock_prevents_duplicate_workers(tmp_path):
    with search.campaign_lock(tmp_path):
        with pytest.raises(ValueError, match="Another process"):
            with search.campaign_lock(tmp_path):
                pass
    with search.campaign_lock(tmp_path):
        pass


def test_real_callback_and_resumed_subset_preserve_common_random_numbers(tmp_path):
    raw = search.load_plan(PLAN)["base"]
    raw["experiment"].update(seed=812, num_tracks=4, bits_per_track=64)
    raw["code"]["K"] = 14
    raw["sweep"] = {"snr_db": [7.0], "iti_coeffs": [.35]}
    cfg = Config.from_dict(raw)
    params = {"n": 168, "dv": 3, "dc": 6, "num_frames": 4, "max_iters": 2,
              "channel_detector": "bcjr_2d_equalized", "equalizer_iterations": 2,
              "inner_demapper": "hard_codebook", "detector_iti_coeffs": [.08, .10]}
    saved = []
    full, _ = run_concatenated(cfg, params, cache_dir=str(tmp_path), result_callback=saved.append)
    assert len(saved) == len(full) == 2
    assert saved == full
    params["detector_iti_coeffs"] = [.10]
    subset, _ = run_concatenated(cfg, params, cache_dir=str(tmp_path))
    for key in full[1]:
        if key != "runtime_sec":
            assert subset[0][key] == full[1][key], key

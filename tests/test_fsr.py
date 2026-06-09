"""FSR aggregation and extrapolation tests."""

from pathlib import Path

import pandas as pd

from tdmr2d.fsr import aggregate_fsr, extrapolate_fsr_targets, load_fsr_rows


def test_fsr_aggregate_and_extrapolate_targets(tmp_path):
    run_a = tmp_path / "outputs" / "runs" / "a"
    run_b = tmp_path / "outputs" / "runs" / "b"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)

    rows = [
        {"name": "highrate_k15_snrscan_iti0_500sector", "snr_db": 14, "iti_coeff": 0.0,
         "sector_errors": 500, "sector_count": 1000, "bit_errors": 5000, "num_bits": 100000},
        {"name": "highrate_k15_snrscan_iti0_500sector", "snr_db": 16, "iti_coeff": 0.0,
         "sector_errors": 50, "sector_count": 1000, "bit_errors": 500, "num_bits": 100000},
        {"name": "highrate_k15_snrscan_iti0_500sector", "snr_db": 18, "iti_coeff": 0.0,
         "sector_errors": 5, "sector_count": 1000, "bit_errors": 50, "num_bits": 100000},
    ]
    pd.DataFrame(rows).to_csv(run_a / "results.csv", index=False)
    pd.DataFrame(rows).to_csv(run_b / "results.csv", index=False)

    raw = load_fsr_rows([tmp_path / "outputs" / "runs"], name_filter="highrate_k15")
    agg = aggregate_fsr(raw, ["iti_coeff"])
    targets, fit_points = extrapolate_fsr_targets(
        agg, ["iti_coeff"], [1e-8, 1e-9, 1e-10], max_fit_fsr=0.8,
    )

    assert len(raw) == 6
    assert len(agg) == 3
    assert agg.loc[agg["snr_db"] == 14, "sector_count"].iloc[0] == 2000
    assert len(fit_points) == 3
    assert set(targets["target_fsr"]) == {1e-8, 1e-9, 1e-10}
    assert targets["estimated_snr_db"].notna().all()
    assert targets["estimated_snr_db"].min() > agg["snr_db"].max()


def test_fsr_fit_reports_insufficient_points():
    agg = pd.DataFrame([
        {"snr_db": 14.0, "iti_coeff": 0.0, "sector_errors": 1000,
         "sector_count": 1000, "FSR": 1.0},
        {"snr_db": 16.0, "iti_coeff": 0.0, "sector_errors": 0,
         "sector_count": 1000, "FSR": 0.0},
    ])

    targets, fit_points = extrapolate_fsr_targets(
        agg, ["iti_coeff"], [1e-8], max_fit_fsr=0.8,
    )

    assert fit_points.empty
    assert targets["estimated_snr_db"].isna().all()
    assert "insufficient" in targets["fit_note"].iloc[0]

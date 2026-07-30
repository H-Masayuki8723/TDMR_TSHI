#!/usr/bin/env python3
"""Aggregate detector-tap sweeps and select the best tap per true ITI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdmr2d.fsr import (  # noqa: E402
    aggregate_fsr,
    extrapolate_fsr_targets,
    load_fsr_rows,
    parse_column_list,
    parse_float_list,
    plot_fsr_extrapolation,
    select_best_detector_map,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sources",
        nargs="+",
        help="outputs/runs directory or one or more results.csv files",
    )
    parser.add_argument("--name-filter", required=True, help="pandas regex applied to experiment.name")
    parser.add_argument("--targets", default="1e-2,1e-3,1e-4")
    parser.add_argument("--group-by", default="iti_coeff,detector_iti_coeff")
    parser.add_argument("--max-fit-fsr", type=float, default=0.3)
    parser.add_argument("--min-fit-fsr", type=float, default=0.0)
    parser.add_argument("--min-points", type=int, default=3)
    parser.add_argument("--out-dir", default="outputs/summaries")
    parser.add_argument("--prefix", required=True)
    parser.add_argument(
        "--allow-duplicate-names",
        action="store_true",
        help="allow one experiment.name to occur in multiple results.csv files",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = load_fsr_rows(args.sources, name_filter=args.name_filter)
    if raw.empty:
        raise SystemExit(f"no FSR rows matched name filter {args.name_filter!r}")

    manifest_agg = {
        "source_files": ("source", "nunique"),
        "raw_rows": ("source", "size"),
    }
    for column in ("seed", "iti_coeff"):
        if column in raw.columns:
            manifest_agg[column] = (column, "first")
    if "sector_count" in raw.columns:
        manifest_agg["sector_count_sum"] = ("sector_count", "sum")
    manifest = raw.groupby("name", as_index=False, dropna=False).agg(**manifest_agg)
    manifest_path = out_dir / f"{args.prefix}_raw_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    duplicates = manifest[manifest["source_files"] > 1]
    if not duplicates.empty and not args.allow_duplicate_names:
        print(duplicates.to_string(index=False), file=sys.stderr)
        raise SystemExit(
            "duplicate experiment.name values found in multiple results.csv files; "
            "remove/exclude duplicates or pass --allow-duplicate-names intentionally"
        )

    group_cols = parse_column_list(args.group_by)
    aggregate = aggregate_fsr(raw, group_cols)
    count_keys = [c for c in group_cols if c in raw.columns] + ["snr_db"]
    diagnostics = {"raw_rows": ("source", "size")}
    if "seed" in raw.columns:
        diagnostics["seed_count"] = ("seed", "nunique")
    if "source" in raw.columns:
        diagnostics["source_files"] = ("source", "nunique")
    counts = raw.groupby(count_keys, as_index=False, dropna=False).agg(**diagnostics)
    aggregate = aggregate.merge(counts, on=count_keys, how="left")

    targets, fit_points = extrapolate_fsr_targets(
        aggregate,
        group_cols,
        parse_float_list(args.targets),
        max_fit_fsr=args.max_fit_fsr,
        min_fit_fsr=args.min_fit_fsr,
        min_points=args.min_points,
    )
    best = select_best_detector_map(targets)

    aggregate_path = out_dir / f"{args.prefix}_aggregate.csv"
    fit_path = out_dir / f"{args.prefix}_fit_points.csv"
    target_path = out_dir / f"{args.prefix}_target_snr.csv"
    best_path = out_dir / f"{args.prefix}_best_detector_map.csv"
    plot_path = out_dir / f"{args.prefix}.png"
    summary_path = out_dir / f"{args.prefix}_summary.json"

    aggregate.to_csv(aggregate_path, index=False)
    fit_points.to_csv(fit_path, index=False)
    targets.to_csv(target_path, index=False)
    best.to_csv(best_path, index=False)
    if not targets.empty:
        plot_fsr_extrapolation(aggregate, targets, group_cols, plot_path)

    summary = {
        "name_filter": args.name_filter,
        "raw_rows": int(len(raw)),
        "experiment_names": int(raw["name"].nunique()) if "name" in raw.columns else None,
        "aggregate_rows": int(len(aggregate)),
        "fit_point_rows": int(len(fit_points)),
        "target_rows": int(len(targets)),
        "best_rows": int(len(best)),
        "duplicate_experiment_names": duplicates["name"].astype(str).tolist(),
        "group_by": group_cols,
        "targets": parse_float_list(args.targets),
        "max_fit_fsr": args.max_fit_fsr,
        "min_fit_fsr": args.min_fit_fsr,
        "min_points": args.min_points,
        "files": {
            "manifest": str(manifest_path),
            "aggregate": str(aggregate_path),
            "fit_points": str(fit_path),
            "target_snr": str(target_path),
            "best_detector_map": str(best_path),
            "plot": str(plot_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(
        f"Detector-map aggregation: {len(raw)} raw rows, "
        f"{raw['name'].nunique() if 'name' in raw.columns else '?'} experiment name(s)"
    )
    print()
    print(manifest.to_string(index=False))
    print()
    print(f"  manifest: {manifest_path}")
    print(f"  aggregate: {aggregate_path}")
    print(f"  target SNR: {target_path}")
    print(f"  best map: {best_path}")
    print(f"  plot: {plot_path}")
    if not best.empty:
        columns = [
            "iti_coeff",
            "target_fsr",
            "detector_iti_coeff",
            "estimated_snr_db",
            "matched_snr_db",
            "gain_vs_matched_db",
            "best_at_lower_boundary",
            "best_at_upper_boundary",
            "fit_points",
            "r2_log10_fsr",
        ]
        columns = [column for column in columns if column in best.columns]
        print()
        print(best[columns].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

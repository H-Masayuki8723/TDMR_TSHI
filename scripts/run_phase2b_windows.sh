#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

configs=(
  "configs/concat_ldpc_2dmtr_highrate_k15_eqpilot_bcjr2_phase2b_trueiti035_lowtap_pilot_seed601_100sector.yaml"
  "configs/concat_ldpc_2dmtr_highrate_k15_eqpilot_bcjr2_phase2b_trueiti020_tail_seed501_500sector.yaml"
  "configs/concat_ldpc_2dmtr_highrate_k15_eqpilot_bcjr2_phase2b_trueiti020_tail_seed502_500sector.yaml"
  "configs/concat_ldpc_2dmtr_highrate_k15_eqpilot_bcjr2_phase2b_trueiti020_tail_seed503_500sector.yaml"
  "configs/concat_ldpc_2dmtr_highrate_k15_eqpilot_bcjr2_phase2b_trueiti025_band_seed511_500sector.yaml"
  "configs/concat_ldpc_2dmtr_highrate_k15_eqpilot_bcjr2_phase2b_trueiti025_band_seed512_500sector.yaml"
  "configs/concat_ldpc_2dmtr_highrate_k15_eqpilot_bcjr2_phase2b_trueiti025_band_seed513_500sector.yaml"
)

experiment_name() {
  awk '
    /^experiment:/ { in_experiment = 1; next }
    in_experiment && /^[^[:space:]]/ { in_experiment = 0 }
    in_experiment && $1 == "name:" { print $2; exit }
  ' "$1"
}

is_complete() {
  local name="$1"
  local result
  while IFS= read -r result; do
    if grep -Fq -- ",${name}," "$result"; then
      return 0
    fi
  done < <(find outputs/runs -name results.csv -type f 2>/dev/null)
  return 1
}

for config in "${configs[@]}"; do
  name="$(experiment_name "$config")"
  if [[ -z "$name" ]]; then
    echo "ERROR: experiment.name not found in $config" >&2
    exit 2
  fi
  if is_complete "$name"; then
    echo "SKIP completed: $name"
    continue
  fi

  echo
  echo "RUN: $name"
  if ! tdmr2d concat "$config"; then
    echo "STOP: failed or interrupted at $name" >&2
    exit 1
  fi
done

echo
echo "Phase 2B queue completed."

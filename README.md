# TDMR_2D_eval — pre-ECC BER evaluation of 2D constrained codes

A Docker-based, reproducible research harness for measuring the **pre-ECC bit
error rate (BER)** of 2D / constrained codes sent through a simplified **2D TDMR
readback channel** with inter-track interference (ITI) and AWGN.

It compares an uncoded baseline, a 1D down-track MTR code, and a 2D-MTR 8×2 code
under identical channel conditions, sweeping SNR and ITI. The initial reference
operating point is **BER ≈ 1×10⁻²**.

---

## 1. What this is (and is not)

- **This system evaluates the pre-ECC BER of 2D coding.** BER is measured by
  comparing the hard-decided received channel bits directly against the
  transmitted channel bits — no error correction, no erasure handling, no
  decode-back. It quantifies how 1D/2D modulation constraints reshape the raw
  channel error behaviour under ITI.
- **Hard-decision pre-ECC BER is the default and headline metric.** LDPC and
  concatenated LDPC+MTR tracks are included as stage-2 extensions, while the
  Pattern A/B/C modulation study remains pre-ECC.
- **A real LDPC decoder is now included (stage 2).** The `soft_awgn` detector
  produces per-bit LLRs and `ldpc.py` implements a regular Gallager code with a
  systematic encoder and a normalized **min-sum belief-propagation** decoder. The
  `tdmr2d ldpc` command reports **post-ECC BER and FER** over the 2D channel,
  alongside the pre-ECC reference. The hard-decision pre-ECC BER of the modulation
  study (Patterns A/B/C) is unchanged and remains the headline pre-ECC metric.
- **A turbo/SISO concatenated LDPC + constrained-code path is included.** The
  `tdmr2d concat` command encodes LDPC codeword bits through an inner 1D/2D-MTR
  codebook, sends the constrained channel bits through the same 2D channel, then
  performs exact blockwise SISO demapping back to LDPC-bit LLRs before BP decode.
  With `ldpc.turbo_iterations > 0`, LDPC extrinsic LLRs are fed back to the inner
  demapper as a-priori information for turbo-style iterations.
- **`TDMR_8x2` denotes the 8×2 block geometry** — 8 down-track bits across 2
  tracks (16 cells per block) — that the 2D-MTR codebook constrains.
- **This is a different project from the earlier ECC comparison.** The previous
  work compared error-correction/detection schemes on 16-bit blocks. This harness
  is separate and deliberately ECC-free: it studies the *channel-level* benefit of
  2D coding before any ECC is applied. Do not conflate the two.

---

## 2. Quickstart

### Docker (intended entry point)

```bash
docker compose build
docker compose run --rm sim tdmr2d smoke
docker compose run --rm sim tdmr2d run     configs/baseline_uncoded.yaml
docker compose run --rm sim tdmr2d sweep   configs/sweep_iti_snr.yaml
docker compose run --rm sim tdmr2d compare configs/compare_three_families.yaml
docker compose run --rm sim tdmr2d ldpc    configs/ldpc_eval.yaml
docker compose run --rm sim tdmr2d concat  configs/concat_ldpc_2dmtr.yaml
docker compose run --rm sim tdmr2d concat  configs/concat_ldpc_2dmtr_bcjr.yaml
```

`outputs/`, `configs/`, and `data/` are bind-mounted, so results and the codebook
cache appear on the host and config edits take effect without a rebuild.

### Local (no Docker)

```bash
pip install -e .            # or: pip install -e ".[dev]" for pytest
tdmr2d smoke
tdmr2d run     configs/baseline_uncoded.yaml
tdmr2d run     configs/baseline_uncoded_soft.yaml   # soft detector (LLRs) + hard decoder
tdmr2d sweep   configs/sweep_iti_snr.yaml
tdmr2d compare configs/compare_three_families.yaml  # uncoded vs 1D vs 2D overlay
tdmr2d ldpc    configs/ldpc_eval.yaml               # post-ECC BER/FER (min-sum BP)
tdmr2d concat  configs/concat_ldpc_2dmtr.yaml        # turbo LDPC + inner 2D-MTR
tdmr2d concat  configs/concat_ldpc_2dmtr_bcjr.yaml   # + BCJR/2D-equalized channel SISO
tdmr2d summarize outputs/runs
```

Runtime dependencies: `numpy`, `pyyaml`, `pandas`, `matplotlib`. Config
validation uses stdlib `dataclasses` and the CLI uses stdlib `argparse` (the spec
allows pydantic/typer *or* dataclass/argparse; the lighter choice keeps the
harness reproducible from a clean Python install).

---

## 3. Evaluation patterns

| Pattern | family | Rate | Purpose |
|---|---|---|---|
| **A — Uncoded baseline** | `uncoded` | `1.0` | Reference BER vs SNR/ITI with no 2D coding. |
| **B — 1D / MTR baseline** | `mtr1d` | `K1/8` (default `7/8 = 0.875`) | Down-track-only constraint; intermediate reference. |
| **C — 2D-MTR 8×2** | `mtr2d_8x2` | `K/16` (`K=14 → 0.875`, `K=15 → 0.9375`) | Suppresses ITI-vulnerable 2D patterns. |

Patterns B and C default to the **same rate (0.875)** so the 1D-vs-2D comparison
is at equal overhead.

**Constraint definitions**

- *Down-track MTR* (`down_mtr`): limits the maximum run of consecutive
  *transitions* (adjacent differing bits) along a track. Length-8 / `MTR≤3` admits
  216 valid words, so `K1=7` (128 words) fits.
- *2D-MTR 8×2*: down-track MTR per track **plus** a cross-track rule forbidding
  long 2×2 checkerboard runs (`max_checker_run`). The anti-diagonal checkerboard
  is the 2×2 pattern most vulnerable to ITI in TDMR.
  - `down_mtr=3, max_checker_run=0` → 22 818 valid words → supports up to `K=14`.
  - `down_mtr=3, max_checker_run=1` → 42 388 valid words → supports up to `K=15`.

Codebooks are enumerated deterministically (sorted by integer value, first `2^K`
kept) and cached under `data/codebooks/`. To attach an **external** hand-designed
K14/K15 8×2 table, drop a matching `.npz` (same name) under `data/codebooks/`; it
is loaded automatically — this is the codebook-connection extension point.

---

## 4. Physical / signal model

Bit mapping: `0 → −1`, `1 → +1`.

2D readback for recorded symbols `x[i, j]` (track `i`, down-track `j`):

```
y[i,j] = c0          * x[i,   j]
       + c_down_prev  * x[i,   j-1]
       + c_down_next  * x[i,   j+1]
       + c_cross_up   * x[i-1, j]
       + c_cross_down * x[i+1, j]
       + noise
```

The cross-track taps (`c_cross_up`, `c_cross_down`) model ITI. Default taps:
`c0=1.0`, down `0.15/0.15`, cross `0.10/0.10`.

**Boundaries.** `zero` (default) pads missing neighbours with 0 (no contribution).
`periodic` and `edge` are also implemented for future use.

**SNR / noise convention.** `noise ~ N(0, σ²)` with

```
σ² = c0² · 10^(−SNR_dB / 10)
```

i.e. SNR is defined relative to the main-tap symbol energy `c0²` (symbols are
±1). `snr_db: null` ⇒ noiseless (`σ=0`). This is a documented modelling
convention, not a calibrated recording-channel SNR.

**Sweeps.** `sweep.snr_db` overrides `channel.snr_db`; `sweep.iti_coeffs`
overrides *both* cross-track taps (`c_cross_up = c_cross_down = iti`) at each grid
point. Defaults:

```yaml
sweep:
  snr_db:     [8, 10, 12, 14, 16, 18, 20]
  iti_coeffs: [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
```

---

## 5. Detector and soft-decision stage

Default detector is a memoryless **hard threshold** (`hard_threshold`):

```
rx_bit = 1 if y >= threshold else 0   (threshold = 0.0)
```

With the ±1 mapping this recovers the symbol sign and drives the pre-ECC BER.

**Soft-decision foundation (`soft_awgn`).** Setting `detector.type: soft_awgn`
additionally generates per-bit LLRs under the main-tap AWGN model:

```
LLR(bit) = 2 * c0 * y / sigma^2     (clipped to ± llr_clip; positive favours bit 1)
```

The LLR sign equals the hard decision (consistency) and its magnitude grows with
SNR. An optional **decoder** then consumes the LLRs (`decoder.py`):

- `decoder.type: hard` — `HardDecisionDecoder`, a no-op that hard-decides the LLR
  (post-decode BER == pre-ECC BER; a wiring sanity check, not an ECC gain).
- `decoder.type: ldpc` — `LDPCDecoder`, usable when constructed with an
  `LDPCCode`; the general `run` decoder slot raises without one because ordinary
  modulation grids are not LDPC-framed. Use `tdmr2d ldpc` or `tdmr2d concat` for
  full encoded-frame evaluations.

A decoder requires `detector.type: soft_awgn`. The hard-decision pre-ECC BER is
always reported regardless of the soft stage. An equalizer can later be inserted
ahead of either detector behind the `Detector` interface.

**Channel-aware SISO detector (`channel_detector: bcjr_2d_equalized`).** The LDPC
and concatenated tracks can replace the main-tap AWGN LLR with a heavier detector
implemented in `siso.py`: neighbouring tracks are converted to soft symbol
estimates and subtracted as ITI, then each track is decoded by exact BCJR over the
down-track 3-tap ISI model. This is exact for the down-track ISI channel with
`boundary: zero` and approximate for the full 2D channel because ITI is cancelled
rather than jointly trellised over all tracks.

### LDPC track — post-ECC BER / FER (`tdmr2d ldpc`)

`ldpc.py` provides a self-contained LDPC code and decoder:

- **Code**: a regular `(dv, dc)` **Gallager** parity-check matrix (`dv` ones per
  column, `dc` per row, `m = n·dv/dc` checks; `n` divisible by `dc`). A GF(2)
  Gaussian elimination yields a **systematic encoder**; `k = n − rank(H)` info
  bits per length-`n` codeword, so `rate = k/n` (≈ `1 − dv/dc`).
- **Decoder**: flooding **belief propagation** in the LLR domain — normalized
  **min-sum** (default) or sum-product — batched over frames with syndrome-based
  early stop.

Geometry: **one codeword per track**, so inter-track interference couples
neighbouring codewords and down-track ISI acts within a codeword. The `soft_awgn`
detector feeds per-bit LLRs to the decoder. Output rows reuse the standard schema
with `BER` = **post-ECC information BER** and `block_error_rate` = **FER** (frame /
message error rate), plus a `pre_ecc_ber` reference column. Figures:
`ldpc_ber_vs_snr.png` (post-ECC vs pre-ECC BER) and `ldpc_fer_vs_snr.png`.

Example (default `(3,6)`, n=300, rate≈0.5): at ITI=0 the pre-ECC channel BER of
~9×10⁻² at 3 dB collapses to post-ECC BER ≈ 2×10⁻³ and to 0 by 4 dB; raising ITI
to 0.2 shifts the waterfall ~2 dB to higher SNR — the expected coding-gain and
ITI-penalty behaviour.

This LDPC track runs over the **uncoded** modulation path.

### Concatenated LDPC + constrained code (`tdmr2d concat`)

`tdmr2d concat` evaluates the outer LDPC code together with an inner modulation
codebook (`uncoded`, `mtr1d`, or `mtr2d_8x2`). For 2D-MTR, pairs of LDPC tracks
are grouped into 8×2 codebook blocks. The receiver scores every candidate
codebook row from the channel-bit LLRs plus optional LDPC a-priori LLRs, then
marginalizes those scores back to the K index bits.

This gives a real LDPC+2D-MTR concatenated evaluation path and writes
`concat_ber_vs_snr.png` / `concat_fer_vs_snr.png`. `turbo_iterations` controls
the number of LDPC-to-demapper extrinsic feedback passes; `0` recovers the
original non-iterative concatenated path. Set
`ldpc.channel_detector: bcjr_2d_equalized` to feed the inner demapper with
channel-aware BCJR/equalized LLRs instead of memoryless AWGN LLRs.

---

## 6. Configuration reference

```yaml
experiment:
  name: uncoded_baseline   # label stored in results
  family: uncoded          # uncoded | mtr1d | mtr2d_8x2 (must match code.type)
  seed: 0                  # base RNG seed (reproducibility)
  num_tracks: 64           # grid tracks  (must be divisible by block cross-size = 2)
  bits_per_track: 4096     # grid columns (must be divisible by block down-size = 8)
code:
  type: uncoded            # uncoded | mtr1d | mtr2d_8x2
  block_shape: [8, 2]      # [down_track, cross_track]; tiling unit for blocks
  K: 14                    # (mtr2d_8x2) info bits per 16-bit block -> rate K/16
  K1: 7                    # (mtr1d) info bits per 8-bit block -> rate K1/8
  down_mtr: 3              # max down-track transition run
  max_checker_run: 0       # (mtr2d_8x2) max 2x2 checkerboard run (0 = forbid any)
channel:
  model: linear_2d_awgn
  c0: 1.0
  c_down_prev: 0.15
  c_down_next: 0.15
  c_cross_up: 0.10
  c_cross_down: 0.10
  snr_db: 14               # number, or null for noiseless
  boundary: zero           # zero | periodic | edge
detector:
  type: hard_threshold     # hard_threshold | soft_awgn
  threshold: 0.0
  llr_clip: 20.0           # (soft_awgn) LLR saturation level
decoder:                   # optional; requires detector.type == soft_awgn
  type: none               # none | hard | ldpc
  params: {}
metrics:
  target_ber: 1.0e-2
output:
  dir: outputs/runs
sweep:                     # optional; required by `tdmr2d sweep` / `compare`
  snr_db: [8, 10, 12, 14, 16, 18, 20]
  iti_coeffs: [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
compare:                   # optional; used by `tdmr2d compare`
  families:
    - {family: uncoded,   code: {type: uncoded}}
    - {family: mtr1d,     code: {type: mtr1d, K1: 7}}
    - {family: mtr2d_8x2, code: {type: mtr2d_8x2, K: 14}}
  slice_snr_db: 14
  slice_iti: 0.10
```

Unknown keys are rejected with a clear error to catch typos early.

---

## 7. CLI

| Command | Description |
|---|---|
| `tdmr2d smoke` | Minimal self-check: noiseless ⇒ BER 0, fixed seed reproduces, writes CSV/JSON. |
| `tdmr2d run CONFIG.yaml` | Single-condition BER evaluation. |
| `tdmr2d sweep CONFIG.yaml` | SNR × ITI sweep; emits BER curves. |
| `tdmr2d compare CONFIG.yaml` | Run several families over one grid; overlay BER curves by family. |
| `tdmr2d ldpc CONFIG.yaml` | LDPC post-ECC BER/FER over the 2D channel (min-sum BP); pre-ECC reference. |
| `tdmr2d concat CONFIG.yaml` | Turbo LDPC + inner constrained-code evaluation with exact SISO demapping. |
| `tdmr2d summarize OUTPUT_DIR` | Aggregate every `results.csv` under a directory into one summary CSV. |

---

## 8. Outputs

```
outputs/
  runs/<timestamp>/
    config.resolved.yaml   # fully-resolved config actually used
    results.csv            # one row per operating point
    results.json           # config + results
    ber_vs_snr.png         # BER vs SNR, one curve per ITI
    ber_vs_iti.png         # BER vs ITI, one curve per SNR
    run.log                # run log
  summaries/               # summary_<timestamp>.csv from `summarize`
  figures/                 # reserved for ad-hoc figures
  reports/                 # reserved for reports
```

### CSV columns

The required columns (in order) are:

```
family,rate,snr_db,iti_coeff,num_bits,bit_errors,BER,block_errors,block_error_rate,seed,runtime_sec
```

| Column | Meaning |
|---|---|
| `family` | `uncoded` / `mtr1d` / `mtr2d_8x2`. |
| `rate` | Code rate (info bits / channel bits). |
| `snr_db` | SNR in dB for this point (`null`/empty = noiseless). |
| `iti_coeff` | Cross-track coupling magnitude used (`c_cross_up`). |
| `num_bits` | Channel bits evaluated (`num_tracks × bits_per_track`). |
| `bit_errors` | Hard-decision bit errors vs the transmitted channel bits. |
| `BER` | `bit_errors / num_bits` (the pre-ECC BER). |
| `block_errors` | 8×2 blocks containing ≥1 bit error. |
| `block_error_rate` | `block_errors / num_blocks`. |
| `seed` | Base RNG seed. |
| `runtime_sec` | Wall-clock seconds for the point. |

Extra diagnostic columns also written: `name`, `num_blocks`, `num_tracks`,
`bits_per_track`, `boundary`, `detector`, `threshold`, `sigma`, `target_ber`,
`hit_target`, `codebook`. With `detector: soft_awgn` you also get `mean_abs_llr`,
and with a decoder `decoder`, `post_decode_BER`, `post_decode_block_error_rate`.

`tdmr2d compare` writes a combined `results.csv` (one `family` per row) plus
`compare_ber_vs_snr.png` and `compare_ber_vs_iti.png`, overlaid by family at the
configured `slice_iti` / `slice_snr_db`.

`tdmr2d ldpc` writes `results.csv` (`family = ldpc`, where `BER` is the post-ECC
information BER and `block_error_rate` is the FER, with extra `pre_ecc_ber`,
`n`, `k`, `num_frames`, `method`, `scale`, `max_iters`) plus `ldpc_ber_vs_snr.png`
and `ldpc_fer_vs_snr.png`. Because the schema matches, `summarize` aggregates LDPC
runs together with the modulation families.

`tdmr2d concat` writes `family = ldpc+<inner_family>`, where `BER` is post-ECC
information BER and `block_error_rate` is FER. Extra columns include
`inner_channel_ber`, `pre_ecc_ber` (LDPC-input hard BER after inner demapping),
`final_ldpc_input_ber`, `outer_ldpc_rate`, `inner_rate`, `inner_code`,
`inner_demapper`, `channel_detector`, `equalizer_iterations`,
`turbo_iterations`, and `turbo_rounds`.

---

## 9. Reproducibility

All randomness flows through a `numpy.random.Generator` seeded from
`experiment.seed`. A single run is deterministic for a given seed; a sweep derives
independent, deterministic per-point streams via `SeedSequence.spawn`. The
fully-resolved config (including the seed) is written to every run directory.
Codebooks are pure functions of their parameters and cached deterministically.

---

## 10. Known limitations

```
- The initial channel model is a simple linear 2D + AWGN model and does NOT
  fully represent a real HDD readback chain.
- The concatenated LDPC + MTR path uses exact blockwise codebook SISO demapping,
  not a full 2D trellis over all tracks.
- The BCJR/equalized detector cancels ITI from soft neighbouring-track estimates;
  it is not a full joint 2D BCJR detector.
- Real media nonlinearity, media noise, head response, and timing/jitter are
  not implemented.
- Therefore the initial results are for RELATIVE comparison only and are not an
  absolute performance guarantee.
```

Additional notes: the SNR convention is main-tap-referenced (not calibrated to a
physical channel); the hard-threshold detector is symbol-by-symbol and not
ISI-aware; codebook selection (first `2^K` by integer value) is a valid but not
rate-optimal constrained code.

---

## 11. Roadmap (future extensions)

- **Soft detector + LLR generator** — done (`soft_awgn`).
- **LDPC BP/min-sum decoder, post-ECC BER/FER** — done (`tdmr2d ldpc`).
- **Joint LDPC + 2D-MTR (concatenated coding)** — done (`tdmr2d concat` with
  exact codebook SISO demapping).
- **Iterative SISO / turbo LDPC + 2D-MTR** — done (`turbo_iterations` feeds LDPC
  extrinsic LLRs back to the constrained-code demapper).
- **Trellis/BCJR channel detector** — done for the down-track 3-tap ISI channel
  (`channel_detector: bcjr_2d_equalized`).
- **2D equalizer ahead of detection** — done as soft ITI cancellation feeding
  the BCJR ISI detector.
- **Full joint 2D BCJR / graph detector** — next: jointly infer neighbouring
  tracks instead of cancelling ITI from soft estimates.
- **Richer channels**: media/jitter noise, nonlinear transition shift, head
  response, periodic/edge boundaries in sweeps.
- **Performance**: swap the numpy channel/detector for numba or cupy (the
  modules are isolated for this).
- **Better 2D codebooks**: capacity-approaching / hand-optimized K14/K15 tables
  loadable from `data/codebooks/`.

---

## 12. Testing

```bash
pip install -e ".[dev]"
pytest                     # or: docker compose run --rm sim pytest
```

Covers the channel (identity, noise reproducibility, ITI monotonicity, SNR
convention), the detector (sign→bit, BER=0 on a clean channel), the BCJR/SISO
channel detector (down-track ISI, soft ITI cancellation), the metrics
(bit/block error counting), the soft stage (LLR formula / clipping / hard
consistency, decoder interface, post-decode == pre-ECC), the family comparison,
the LDPC code/decoder (Gallager regularity, parity, systematic roundtrip,
noiseless decode, coding gain, posterior LLRs, post-ECC schema), the concatenated LDPC+MTR path,
and an end-to-end smoke check
that the CLI writes CSV/JSON and that the codebooks reach the expected K14/K15
sizes. **49 tests total.**

---

## 13. Project layout

```
TDMR_2D_eval/
├── Dockerfile, docker-compose.yml, pyproject.toml, README.md
├── configs/      baseline_uncoded | baseline_mtr | baseline_2dmtr_8x2 |
│                 baseline_uncoded_soft | sweep_iti_snr |
│                 compare_three_families | ldpc_eval | concat_ldpc_2dmtr |
│                 concat_ldpc_2dmtr_bcjr
├── src/tdmr2d/   cli, config, patterns, constraints, codebook, channel,
│                 detector, decoder, ldpc, concat, siso, metrics,
│                 experiments, reports, io
├── tests/        test_channel | test_detector | test_metrics | test_smoke |
│                 test_soft | test_compare | test_ldpc | test_concat |
│                 test_siso
├── data/         codebooks/ (generated cache), examples/
└── outputs/      runs/ summaries/ figures/ reports/
```

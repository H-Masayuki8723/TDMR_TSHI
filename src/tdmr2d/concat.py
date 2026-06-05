"""Concatenated LDPC + constrained-code helpers.

The outer code is an LDPC codeword stream. The inner code is one of the existing
modulation codebooks (uncoded, 1D-MTR, or 2D-MTR 8x2). For constrained codebooks
the receiver uses exact blockwise soft demapping: every codebook row is scored
from the channel-bit LLRs, then marginalized back to the K information bits that
formed the codebook index.

The same demapper also supports turbo-style iterations: LDPC extrinsic LLRs can
be supplied as a-priori information for the codebook index bits, and the demapper
returns extrinsic LLRs back to the LDPC decoder.

The optional ``trellis_pruned`` demapper extends the blockwise marginalizer with
block-to-block MTR/checkerboard transition pruning. It is a SISO/log-MAP block
trellis: impossible transitions are assigned zero probability while still
returning soft LLRs for LDPC/turbo iterations.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from . import codebook as codebook_mod
from . import constraints
from .config import Config


_TRELLIS_CACHE: Dict[Tuple[str, int, int, int], Dict] = {}
_ROW_BITS_CACHE: Dict[Tuple[str, int, int], np.ndarray] = {}
_STATEFUL_TRELLIS_CACHE: Dict[Tuple[str, int, int, int, int], Dict] = {}
_HARD_DEMAP_CACHE: Dict[Tuple[str, int, int], np.ndarray] = {}


def bits_to_indices(groups: np.ndarray) -> np.ndarray:
    """Convert ``(..., K)`` LSB-first bit groups to integer codebook indices."""
    bits = np.asarray(groups, dtype=np.uint8)
    if bits.ndim == 1:
        bits = bits[None, :]
    K = bits.shape[-1]
    weights = (1 << np.arange(K, dtype=np.uint64))
    return (bits.astype(np.uint64) @ weights).astype(np.int64)


def indices_to_bits(indices: np.ndarray, K: int) -> np.ndarray:
    """Convert integer codebook indices to LSB-first ``(..., K)`` bit groups."""
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    return np.stack([constraints.bits_down(int(v), K) for v in idx]).astype(np.uint8)


def _logsumexp(a: np.ndarray, axis: int = 1) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    m = np.max(arr, axis=axis, keepdims=True)
    finite = np.isfinite(m)
    with np.errstate(invalid="ignore"):
        shifted = np.where(finite, arr - m, -np.inf)
    out = np.log(np.exp(shifted).sum(axis=axis)) + np.squeeze(m, axis=axis)
    return np.where(np.squeeze(finite, axis=axis), out, -np.inf)


def _logsumexp1d(v: np.ndarray) -> float:
    arr = np.asarray(v, dtype=np.float64)
    if arr.size == 0:
        return -np.inf
    m = float(np.max(arr))
    if not np.isfinite(m):
        return -np.inf
    return float(m + np.log(np.exp(arr - m).sum()))


def _prefix_suffix_run(flags: np.ndarray) -> Tuple[int, int]:
    vals = np.asarray(flags, dtype=bool)
    prefix = 0
    for v in vals:
        if not v:
            break
        prefix += 1
    suffix = 0
    for v in vals[::-1]:
        if not v:
            break
        suffix += 1
    return int(prefix), int(suffix)


def _checker_flags(block: np.ndarray) -> np.ndarray:
    b = np.asarray(block, dtype=np.uint8)
    if b.shape[0] != 2 or b.shape[1] < 2:
        return np.zeros((0,), dtype=bool)
    a = b[0, :-1]
    c = b[0, 1:]
    d = b[1, :-1]
    e = b[1, 1:]
    return (a != d) & (a != c) & (e != d) & (e != c) & (a == e) & (d == c)


def _boundary_checker(prev_last: Tuple[int, ...], curr_first: Tuple[int, ...]) -> bool:
    if len(prev_last) != 2 or len(curr_first) != 2:
        return False
    a, d = prev_last
    c, e = curr_first
    return bool(a != d and a != c and e != d and e != c and a == e and d == c)


def _word_boundary_state(word: np.ndarray, cb: codebook_mod.Codebook) -> Tuple[Tuple, Tuple]:
    block = np.asarray(word, dtype=np.uint8).reshape(cb.cross, cb.down)
    first = tuple(int(v) for v in block[:, 0])
    last = tuple(int(v) for v in block[:, -1])
    prefix_mtr: List[int] = []
    suffix_mtr: List[int] = []
    for t in range(cb.cross):
        tr = block[t, 1:] != block[t, :-1]
        p, s = _prefix_suffix_run(tr)
        prefix_mtr.append(p)
        suffix_mtr.append(s)
    checker = _checker_flags(block)
    checker_prefix, checker_suffix = _prefix_suffix_run(checker)
    left = (first, tuple(prefix_mtr), int(checker_prefix))
    right = (last, tuple(suffix_mtr), int(checker_suffix))
    return left, right


def _transition_allowed(left: Tuple, right: Tuple, down_mtr: int,
                        max_checker_run_allowed: int) -> bool:
    """Return True when a previous right state may precede a current left state."""
    prev_last, prev_suffix_mtr, prev_checker_suffix = right
    curr_first, curr_prefix_mtr, curr_checker_prefix = left

    for a, b, s_prev, p_curr in zip(prev_last, curr_first, prev_suffix_mtr, curr_prefix_mtr):
        if a != b and int(s_prev) + 1 + int(p_curr) > int(down_mtr):
            return False

    if _boundary_checker(prev_last, curr_first):
        run = int(prev_checker_suffix) + 1 + int(curr_checker_prefix)
        if run > int(max_checker_run_allowed):
            return False
    return True


def _transition_trellis(cb: codebook_mod.Codebook, down_mtr: int,
                        max_checker_run_allowed: int = 0) -> Dict:
    key = (cb.name, cb.size, int(down_mtr), int(max_checker_run_allowed))
    if key in _TRELLIS_CACHE:
        return _TRELLIS_CACHE[key]

    left_keys: List[Tuple] = []
    right_keys: List[Tuple] = []
    left_groups: Dict[Tuple, List[int]] = {}
    right_groups: Dict[Tuple, List[int]] = {}

    for row, word in enumerate(cb.words):
        left, right = _word_boundary_state(word, cb)
        left_keys.append(left)
        right_keys.append(right)
        left_groups.setdefault(left, []).append(row)
        right_groups.setdefault(right, []).append(row)

    left_groups_np = {k: np.asarray(v, dtype=np.int64) for k, v in left_groups.items()}
    right_groups_np = {k: np.asarray(v, dtype=np.int64) for k, v in right_groups.items()}

    allowed_right_by_left: Dict[Tuple, List[Tuple]] = {}
    allowed_left_by_right: Dict[Tuple, List[Tuple]] = {}
    for lk in left_groups_np:
        allowed = [rk for rk in right_groups_np
                   if _transition_allowed(lk, rk, down_mtr, max_checker_run_allowed)]
        allowed_right_by_left[lk] = allowed
    for rk in right_groups_np:
        allowed = [lk for lk in left_groups_np
                   if _transition_allowed(lk, rk, down_mtr, max_checker_run_allowed)]
        allowed_left_by_right[rk] = allowed

    possible = sum(len(v) for v in allowed_right_by_left.values())
    total = len(left_groups_np) * len(right_groups_np)
    trellis = {
        "left_keys": left_keys,
        "right_keys": right_keys,
        "left_groups": left_groups_np,
        "right_groups": right_groups_np,
        "allowed_right_by_left": allowed_right_by_left,
        "allowed_left_by_right": allowed_left_by_right,
        "num_left_states": len(left_groups_np),
        "num_right_states": len(right_groups_np),
        "state_transition_count": int(possible),
        "state_transition_density": float(possible / total) if total else 1.0,
    }
    _TRELLIS_CACHE[key] = trellis
    return trellis


def _aggregate_logsum(values: np.ndarray, groups: Dict[Tuple, np.ndarray]) -> Dict[Tuple, float]:
    return {key: _logsumexp1d(values[idx]) for key, idx in groups.items()}


def _row_bits(cb: codebook_mod.Codebook) -> np.ndarray:
    key = (cb.name, cb.size, cb.K)
    if key not in _ROW_BITS_CACHE:
        _ROW_BITS_CACHE[key] = indices_to_bits(np.arange(cb.size), cb.K).astype(bool)
    return _ROW_BITS_CACHE[key]


def _score_blocks(llr_blocks: np.ndarray, cb: codebook_mod.Codebook,
                  apriori_llr: np.ndarray) -> np.ndarray:
    row_bits_f = _row_bits(cb).astype(np.float64)
    return np.asarray(llr_blocks, dtype=np.float64) @ cb.words.astype(np.float64).T + (
        np.asarray(apriori_llr, dtype=np.float64) @ row_bits_f.T
    )


def _marginal_llrs(row_metrics: np.ndarray, cb: codebook_mod.Codebook,
                   apriori_llr: np.ndarray, extrinsic: bool) -> np.ndarray:
    bits = _row_bits(cb)
    out = np.empty((row_metrics.shape[0], cb.K), dtype=np.float64)
    for k in range(cb.K):
        one = bits[:, k]
        posterior = _logsumexp(row_metrics[:, one], axis=1) - _logsumexp(row_metrics[:, ~one], axis=1)
        out[:, k] = posterior - apriori_llr[:, k] if extrinsic else posterior
    return out


def soft_demapper_llr(llr_blocks: np.ndarray, cb: codebook_mod.Codebook,
                      apriori_llr: Optional[np.ndarray] = None,
                      chunk: int = 64, extrinsic: bool = True,
                      llr_clip: Optional[float] = None) -> np.ndarray:
    """Map channel-bit LLR blocks back to codebook-index LLRs.

    ``llr_blocks`` has shape ``(num_blocks, cb.block_len)`` and uses the project
    convention positive => bit 1. ``apriori_llr`` optionally has shape
    ``(num_blocks, cb.K)`` and represents LDPC extrinsic information from the
    previous turbo iteration. The return value has shape ``(num_blocks, cb.K)``.

    By default the return value is extrinsic, i.e. posterior minus the supplied
    a-priori LLR. With no a-priori input this equals the posterior LLR.
    """
    llr = np.asarray(llr_blocks, dtype=np.float64)
    if llr.ndim != 2 or llr.shape[1] != cb.block_len:
        raise ValueError(f"llr_blocks must have shape (N, {cb.block_len})")
    if apriori_llr is None:
        apriori = np.zeros((llr.shape[0], cb.K), dtype=np.float64)
    else:
        apriori = np.asarray(apriori_llr, dtype=np.float64)
        if apriori.shape != (llr.shape[0], cb.K):
            raise ValueError(f"apriori_llr must have shape (N, {cb.K})")

    row_bits = indices_to_bits(np.arange(cb.size), cb.K).astype(bool)
    words = cb.words.astype(np.float64)
    row_bits_f = row_bits.astype(np.float64)
    out = np.empty((llr.shape[0], cb.K), dtype=np.float64)

    for s in range(0, llr.shape[0], chunk):
        e = min(s + chunk, llr.shape[0])
        scores = llr[s:e] @ words.T
        scores += apriori[s:e] @ row_bits_f.T
        for k in range(cb.K):
            one = row_bits[:, k]
            posterior = _logsumexp(scores[:, one]) - _logsumexp(scores[:, ~one])
            out[s:e, k] = posterior - apriori[s:e, k] if extrinsic else posterior
    if llr_clip is not None and llr_clip > 0:
        out = np.clip(out, -float(llr_clip), float(llr_clip))
    return out


def _codebook_word_ints(cb: codebook_mod.Codebook) -> np.ndarray:
    weights = (1 << np.arange(cb.block_len, dtype=np.uint64))
    return (cb.words.astype(np.uint64) @ weights).astype(np.int64)


def _masks_by_weight(nbits: int) -> List[np.ndarray]:
    masks: List[np.ndarray] = []
    weights = np.array([int(v) for v in (1 << np.arange(nbits, dtype=np.uint64))], dtype=np.int64)
    for dist in range(nbits + 1):
        vals = []
        for mask in range(1 << nbits):
            if int(mask).bit_count() == dist:
                vals.append(mask)
        masks.append(np.asarray(vals, dtype=np.int64))
    return masks


def _nearest_hard_row_table(cb: codebook_mod.Codebook) -> np.ndarray:
    """Map every hard channel word to the nearest codebook row by Hamming distance."""
    key = (cb.name, cb.size, cb.block_len)
    if key in _HARD_DEMAP_CACHE:
        return _HARD_DEMAP_CACHE[key]
    if cb.block_len > 20:
        raise ValueError("hard_codebook demapper table is intended for short block codebooks")

    table = np.full(1 << cb.block_len, -1, dtype=np.int64)
    word_ints = _codebook_word_ints(cb)
    rows = np.arange(cb.size, dtype=np.int64)
    for masks in _masks_by_weight(cb.block_len):
        for mask in masks:
            pat = word_ints ^ int(mask)
            missing = table[pat] < 0
            if np.any(missing):
                table[pat[missing]] = rows[missing]
        if np.all(table >= 0):
            break
    if np.any(table < 0):  # pragma: no cover - exhaustive masks should fill all words.
        raise ValueError("failed to build complete hard-demapper nearest-row table")
    _HARD_DEMAP_CACHE[key] = table
    return table


def hard_codebook_demapper_llr(llr_blocks: np.ndarray, cb: codebook_mod.Codebook,
                               llr_clip: Optional[float] = None) -> np.ndarray:
    """Fast hard-decision nearest-codebook demapper.

    This is a practical high-rate baseline, not a SISO demapper. It ignores
    a-priori feedback and assigns fixed-magnitude LLRs to the selected codebook
    index bits.
    """
    llr = np.asarray(llr_blocks, dtype=np.float64)
    if llr.ndim != 2 or llr.shape[1] != cb.block_len:
        raise ValueError(f"llr_blocks must have shape (N, {cb.block_len})")
    hard = (llr >= 0.0).astype(np.uint8)
    patterns = bits_to_indices(hard)
    table = _nearest_hard_row_table(cb)
    rows = table[patterns]
    bits = indices_to_bits(rows, cb.K)
    mag = float(llr_clip if llr_clip is not None and llr_clip > 0 else 20.0)
    return np.where(bits == 1, mag, -mag).astype(np.float64)


def trellis_pruned_viterbi_indices(llr_blocks: np.ndarray, cb: codebook_mod.Codebook,
                                   down_mtr: int,
                                   max_checker_run_allowed: int = 0,
                                   apriori_llr: Optional[np.ndarray] = None) -> np.ndarray:
    """Hard Viterbi path over codebook blocks with invalid transitions pruned.

    This is mainly a diagnostic/baseline. The LDPC/turbo path should usually use
    :func:`trellis_pruned_demapper_llr`, which keeps soft output.
    """
    llr = np.asarray(llr_blocks, dtype=np.float64)
    if llr.ndim != 2 or llr.shape[1] != cb.block_len:
        raise ValueError(f"llr_blocks must have shape (N, {cb.block_len})")
    if apriori_llr is None:
        apriori = np.zeros((llr.shape[0], cb.K), dtype=np.float64)
    else:
        apriori = np.asarray(apriori_llr, dtype=np.float64)
        if apriori.shape != (llr.shape[0], cb.K):
            raise ValueError(f"apriori_llr must have shape (N, {cb.K})")
    if llr.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)

    scores = _score_blocks(llr, cb, apriori)
    trellis = _transition_trellis(cb, down_mtr, max_checker_run_allowed)
    left_groups = trellis["left_groups"]
    right_groups = trellis["right_groups"]
    allowed_right_by_left = trellis["allowed_right_by_left"]

    delta = scores[0].copy()
    back = np.zeros((llr.shape[0], cb.size), dtype=np.int64)
    for b in range(1, llr.shape[0]):
        best_by_right: Dict[Tuple, Tuple[float, int]] = {}
        for rk, rows in right_groups.items():
            local = delta[rows]
            pos = int(np.argmax(local))
            best_by_right[rk] = (float(local[pos]), int(rows[pos]))

        next_delta = np.full(cb.size, -np.inf, dtype=np.float64)
        for lk, rows in left_groups.items():
            candidates = [best_by_right[rk] for rk in allowed_right_by_left[lk]
                          if np.isfinite(best_by_right[rk][0])]
            if not candidates:
                continue
            best_val, best_row = max(candidates, key=lambda x: x[0])
            next_delta[rows] = scores[b, rows] + best_val
            back[b, rows] = best_row
        delta = next_delta

    path = np.empty(llr.shape[0], dtype=np.int64)
    path[-1] = int(np.argmax(delta))
    for b in range(llr.shape[0] - 1, 0, -1):
        path[b - 1] = back[b, path[b]]
    return path


def trellis_pruned_demapper_llr(llr_sequences: np.ndarray, cb: codebook_mod.Codebook,
                                down_mtr: int,
                                max_checker_run_allowed: int = 0,
                                apriori_llr: Optional[np.ndarray] = None,
                                extrinsic: bool = True,
                                llr_clip: Optional[float] = None) -> Tuple[np.ndarray, Dict]:
    """SISO block-trellis demapper with MTR/checkerboard transition pruning.

    ``llr_sequences`` has shape ``(num_sequences, blocks_per_sequence, block_len)``.
    The return value has shape ``(num_sequences, blocks_per_sequence, K)``. Invalid
    block-to-block transitions are excluded from the forward/backward sums.
    """
    llr = np.asarray(llr_sequences, dtype=np.float64)
    if llr.ndim == 2:
        llr = llr[None, :, :]
    if llr.ndim != 3 or llr.shape[2] != cb.block_len:
        raise ValueError(f"llr_sequences must have shape (S, B, {cb.block_len})")
    if apriori_llr is None:
        apriori = np.zeros((llr.shape[0], llr.shape[1], cb.K), dtype=np.float64)
    else:
        apriori = np.asarray(apriori_llr, dtype=np.float64)
        if apriori.ndim == 2:
            apriori = apriori[None, :, :]
        if apriori.shape != (llr.shape[0], llr.shape[1], cb.K):
            raise ValueError(f"apriori_llr must have shape (S, B, {cb.K})")

    trellis = _transition_trellis(cb, down_mtr, max_checker_run_allowed)
    left_groups = trellis["left_groups"]
    right_groups = trellis["right_groups"]
    allowed_right_by_left = trellis["allowed_right_by_left"]
    allowed_left_by_right = trellis["allowed_left_by_right"]

    S, B, _ = llr.shape
    out = np.empty((S, B, cb.K), dtype=np.float64)

    for s in range(S):
        scores = _score_blocks(llr[s], cb, apriori[s])
        alpha = np.full((B, cb.size), -np.inf, dtype=np.float64)
        beta = np.full((B, cb.size), -np.inf, dtype=np.float64)
        alpha[0] = scores[0]
        beta[-1].fill(0.0)

        for b in range(1, B):
            prev_by_right = _aggregate_logsum(alpha[b - 1], right_groups)
            prev_for_left = {
                lk: _logsumexp1d(np.asarray([prev_by_right[rk] for rk in allowed_right_by_left[lk]],
                                            dtype=np.float64))
                for lk in left_groups
            }
            for lk, rows in left_groups.items():
                alpha[b, rows] = scores[b, rows] + prev_for_left[lk]

        for b in range(B - 2, -1, -1):
            future = scores[b + 1] + beta[b + 1]
            future_by_left = _aggregate_logsum(future, left_groups)
            future_for_right = {
                rk: _logsumexp1d(np.asarray([future_by_left[lk] for lk in allowed_left_by_right[rk]],
                                            dtype=np.float64))
                for rk in right_groups
            }
            for rk, rows in right_groups.items():
                beta[b, rows] = future_for_right[rk]

        out[s] = _marginal_llrs(alpha + beta, cb, apriori[s], extrinsic=extrinsic)

    if llr_clip is not None and llr_clip > 0:
        out = np.clip(out, -float(llr_clip), float(llr_clip))
    meta = {
        "inner_demapper": "trellis_pruned",
        "trellis_num_left_states": trellis["num_left_states"],
        "trellis_num_right_states": trellis["num_right_states"],
        "trellis_state_transition_count": trellis["state_transition_count"],
        "trellis_state_transition_density": trellis["state_transition_density"],
    }
    return out, meta


def trellis_transition_meta(cb: codebook_mod.Codebook, down_mtr: int,
                            max_checker_run_allowed: int = 0) -> Dict:
    """Return structural state-transition counts for a boundary-pruned trellis."""
    trellis = _transition_trellis(cb, down_mtr, max_checker_run_allowed)
    return {
        "trellis_num_left_states": trellis["num_left_states"],
        "trellis_num_right_states": trellis["num_right_states"],
        "trellis_state_transition_count": trellis["state_transition_count"],
        "trellis_state_transition_density": trellis["state_transition_density"],
    }


def _rows_from_right_to_safe(trellis: Dict, prev_right: Tuple, safe_right: set) -> List[int]:
    rows: List[int] = []
    right_keys = trellis["right_keys"]
    for lk in trellis["allowed_left_by_right"][prev_right]:
        for row in trellis["left_groups"][lk]:
            row_i = int(row)
            if right_keys[row_i] in safe_right:
                rows.append(row_i)
    rows.sort()
    return rows


def _stateful_trellis_maps(cb: codebook_mod.Codebook, info_K: int, down_mtr: int,
                           max_checker_run_allowed: int = 0) -> Dict:
    """Build deterministic K-bit/state -> codeword maps for a matched encoder."""
    key = (cb.name, cb.size, int(info_K), int(down_mtr), int(max_checker_run_allowed))
    if key in _STATEFUL_TRELLIS_CACHE:
        return _STATEFUL_TRELLIS_CACHE[key]

    need = 1 << int(info_K)
    if cb.size < need:
        raise ValueError(f"candidate codebook has {cb.size} words but K={info_K} needs {need}")

    trellis = _transition_trellis(cb, down_mtr, max_checker_run_allowed)
    all_right = set(trellis["right_groups"].keys())
    safe = set(all_right)
    while True:
        next_safe = {
            rk for rk in all_right
            if len(_rows_from_right_to_safe(trellis, rk, safe)) >= need
        }
        if next_safe == safe:
            break
        safe = next_safe
        if not safe:
            raise ValueError(
                f"no safe stateful trellis remains for K={info_K}, "
                f"boundary_down_mtr={down_mtr}, boundary_max_checker_run={max_checker_run_allowed}"
            )

    state_keys = sorted(safe)
    state_index = {rk: i for i, rk in enumerate(state_keys)}
    right_keys = trellis["right_keys"]

    init_rows_all = [row for row in range(cb.size) if right_keys[row] in safe]
    init_rows_all.sort()
    if len(init_rows_all) < need:
        raise ValueError(
            f"initial state has {len(init_rows_all)} safe words but K={info_K} needs {need}"
        )
    init_rows = np.asarray(init_rows_all[:need], dtype=np.int64)
    init_next = np.asarray([state_index[right_keys[int(row)]] for row in init_rows], dtype=np.int64)

    rows_by_state = np.empty((len(state_keys), need), dtype=np.int64)
    next_by_state = np.empty((len(state_keys), need), dtype=np.int64)
    min_allowed = None
    for si, rk in enumerate(state_keys):
        rows = _rows_from_right_to_safe(trellis, rk, safe)
        if len(rows) < need:
            raise ValueError(f"safe state unexpectedly has only {len(rows)} allowed words")
        min_allowed = len(rows) if min_allowed is None else min(min_allowed, len(rows))
        chosen = rows[:need]
        rows_by_state[si] = np.asarray(chosen, dtype=np.int64)
        next_by_state[si] = np.asarray([state_index[right_keys[int(row)]] for row in chosen], dtype=np.int64)

    label_bits = indices_to_bits(np.arange(need), int(info_K)).astype(bool)
    maps = {
        "candidate_codebook": cb.name,
        "info_K": int(info_K),
        "need": need,
        "state_keys": state_keys,
        "state_index": state_index,
        "init_rows": init_rows,
        "init_next": init_next,
        "rows_by_state": rows_by_state,
        "next_by_state": next_by_state,
        "label_bits": label_bits,
        "num_safe_states": len(state_keys),
        "min_allowed_words_per_state": int(min_allowed if min_allowed is not None else 0),
        "init_safe_words": int(len(init_rows_all)),
        "trellis": trellis,
    }
    _STATEFUL_TRELLIS_CACHE[key] = maps
    return maps


def _stateful_candidate_codebook(cfg: Config, cache_dir: str) -> codebook_mod.Codebook:
    if cfg.experiment.family != "mtr2d_8x2":
        raise ValueError("stateful_trellis candidate codebook is currently supported for mtr2d_8x2 only")
    candidate_K = int(
        cfg.code.trellis_candidate_K
        if cfg.code.trellis_candidate_K is not None
        else cfg.code.K
    )
    return codebook_mod.get_2d_codebook(
        K=candidate_K, cross=cfg.code.block_cross, down=cfg.code.block_down,
        down_mtr=cfg.code.down_mtr, max_checker_run=cfg.code.max_checker_run,
        cache_dir=cache_dir,
    )


def _encode_stateful_trellis_sequences(groups: np.ndarray, cb: codebook_mod.Codebook,
                                       info_K: int, down_mtr: int,
                                       max_checker_run_allowed: int = 0) -> Tuple[np.ndarray, Dict]:
    idx = bits_to_indices(np.asarray(groups, dtype=np.uint8).reshape(-1, info_K)).reshape(groups.shape[0], groups.shape[1])
    maps = _stateful_trellis_maps(cb, info_K, down_mtr, max_checker_run_allowed)
    rows = np.empty(idx.shape, dtype=np.int64)
    for s in range(idx.shape[0]):
        first = int(idx[s, 0])
        row = int(maps["init_rows"][first])
        state = int(maps["init_next"][first])
        rows[s, 0] = row
        for b in range(1, idx.shape[1]):
            label = int(idx[s, b])
            row = int(maps["rows_by_state"][state, label])
            state = int(maps["next_by_state"][state, label])
            rows[s, b] = row
    words = cb.words[rows]
    meta = {
        "stateful_info_bits": int(info_K),
        "stateful_candidate_code": cb.name,
        "stateful_candidate_K": int(cb.K),
        "stateful_safe_states": maps["num_safe_states"],
        "stateful_min_allowed_words_per_state": maps["min_allowed_words_per_state"],
        "stateful_init_safe_words": maps["init_safe_words"],
    }
    return words.astype(np.uint8), meta


def stateful_trellis_demapper_llr(llr_sequences: np.ndarray, cb: codebook_mod.Codebook,
                                  info_K: int, down_mtr: int,
                                  max_checker_run_allowed: int = 0,
                                  apriori_llr: Optional[np.ndarray] = None,
                                  extrinsic: bool = True,
                                  llr_clip: Optional[float] = None) -> Tuple[np.ndarray, Dict]:
    """SISO demapper for the matched stateful K-bit trellis encoder."""
    llr = np.asarray(llr_sequences, dtype=np.float64)
    if llr.ndim == 2:
        llr = llr[None, :, :]
    if llr.ndim != 3 or llr.shape[2] != cb.block_len:
        raise ValueError(f"llr_sequences must have shape (S, B, {cb.block_len})")
    if apriori_llr is None:
        apriori = np.zeros((llr.shape[0], llr.shape[1], int(info_K)), dtype=np.float64)
    else:
        apriori = np.asarray(apriori_llr, dtype=np.float64)
        if apriori.ndim == 2:
            apriori = apriori[None, :, :]
        if apriori.shape != (llr.shape[0], llr.shape[1], int(info_K)):
            raise ValueError(f"apriori_llr must have shape (S, B, {int(info_K)})")

    maps = _stateful_trellis_maps(cb, info_K, down_mtr, max_checker_run_allowed)
    init_rows = maps["init_rows"]
    init_next = maps["init_next"]
    rows_by_state = maps["rows_by_state"]
    next_by_state = maps["next_by_state"]
    label_bits = maps["label_bits"]
    label_bits_f = label_bits.astype(np.float64)
    M = int(maps["num_safe_states"])
    need = int(maps["need"])
    S, B, _ = llr.shape
    out = np.empty((S, B, int(info_K)), dtype=np.float64)

    for s in range(S):
        row_scores = llr[s] @ cb.words.astype(np.float64).T
        prior_scores = apriori[s] @ label_bits_f.T

        alpha = np.full((B, M), -np.inf, dtype=np.float64)
        vals0 = row_scores[0, init_rows] + prior_scores[0]
        np.logaddexp.at(alpha[0], init_next, vals0)

        for b in range(1, B):
            new = np.full(M, -np.inf, dtype=np.float64)
            for state in range(M):
                if not np.isfinite(alpha[b - 1, state]):
                    continue
                vals = alpha[b - 1, state] + row_scores[b, rows_by_state[state]] + prior_scores[b]
                np.logaddexp.at(new, next_by_state[state], vals)
            alpha[b] = new

        beta = np.full((B, M), -np.inf, dtype=np.float64)
        beta[-1].fill(0.0)
        for b in range(B - 2, -1, -1):
            for state in range(M):
                vals = (
                    row_scores[b + 1, rows_by_state[state]]
                    + prior_scores[b + 1]
                    + beta[b + 1, next_by_state[state]]
                )
                beta[b, state] = _logsumexp1d(vals)

        for b in range(B):
            metric_by_label = np.full(need, -np.inf, dtype=np.float64)
            if b == 0:
                metric_by_label = row_scores[0, init_rows] + prior_scores[0] + beta[0, init_next]
            else:
                for state in range(M):
                    if not np.isfinite(alpha[b - 1, state]):
                        continue
                    vals = (
                        alpha[b - 1, state]
                        + row_scores[b, rows_by_state[state]]
                        + prior_scores[b]
                        + beta[b, next_by_state[state]]
                    )
                    metric_by_label = np.logaddexp(metric_by_label, vals)
            for k in range(int(info_K)):
                one = label_bits[:, k]
                posterior = _logsumexp1d(metric_by_label[one]) - _logsumexp1d(metric_by_label[~one])
                out[s, b, k] = posterior - apriori[s, b, k] if extrinsic else posterior

    if llr_clip is not None and llr_clip > 0:
        out = np.clip(out, -float(llr_clip), float(llr_clip))
    meta = {
        "inner_demapper": "stateful_trellis",
        "stateful_info_bits": int(info_K),
        "stateful_candidate_code": cb.name,
        "stateful_candidate_K": int(cb.K),
        "stateful_safe_states": maps["num_safe_states"],
        "stateful_min_allowed_words_per_state": maps["min_allowed_words_per_state"],
        "stateful_init_safe_words": maps["init_safe_words"],
        "trellis_num_left_states": maps["trellis"]["num_left_states"],
        "trellis_num_right_states": maps["trellis"]["num_right_states"],
        "trellis_state_transition_count": maps["trellis"]["state_transition_count"],
        "trellis_state_transition_density": maps["trellis"]["state_transition_density"],
    }
    return out, meta


def count_trellis_transition_violations(block_sequences: np.ndarray, cb: codebook_mod.Codebook,
                                        down_mtr: int,
                                        max_checker_run_allowed: int = 0) -> Tuple[int, int]:
    """Count adjacent block transitions that violate the global trellis rule."""
    blocks = np.asarray(block_sequences, dtype=np.uint8)
    if blocks.ndim == 2:
        blocks = blocks[None, :, :]
    if blocks.ndim != 3 or blocks.shape[2] != cb.block_len:
        raise ValueError(f"block_sequences must have shape (S, B, {cb.block_len})")
    if blocks.shape[1] < 2:
        return 0, 0
    violations = 0
    total = 0
    for seq in blocks:
        states = [_word_boundary_state(word, cb) for word in seq]
        for (_, right), (left, _) in zip(states[:-1], states[1:]):
            total += 1
            if not _transition_allowed(left, right, down_mtr, max_checker_run_allowed):
                violations += 1
    return int(violations), int(total)


def _inner_codebook(cfg: Config, cache_dir: str) -> codebook_mod.Codebook:
    fam = cfg.experiment.family
    if fam == "mtr1d":
        return codebook_mod.get_1d_codebook(
            K1=cfg.code.K1, length=cfg.code.block_down,
            down_mtr=cfg.code.down_mtr, cache_dir=cache_dir,
        )
    if fam == "mtr2d_8x2":
        return codebook_mod.get_2d_codebook(
            K=cfg.code.K, cross=cfg.code.block_cross, down=cfg.code.block_down,
            down_mtr=cfg.code.down_mtr, max_checker_run=cfg.code.max_checker_run,
            cache_dir=cache_dir,
        )
    raise ValueError(f"no constrained codebook for family {fam!r}")


def encode_inner_grid(ldpc_bits: np.ndarray, cfg: Config,
                      cache_dir: str = "data/codebooks") -> Tuple[np.ndarray, Dict]:
    """Encode LDPC codeword bits through the configured inner modulation code."""
    bits = np.asarray(ldpc_bits, dtype=np.uint8)
    if bits.ndim != 2:
        raise ValueError("ldpc_bits must have shape (num_frames, n)")
    num_frames, n = bits.shape
    fam = cfg.experiment.family

    if fam == "uncoded":
        return bits.copy(), {
            "inner_code": "uncoded",
            "inner_rate": 1.0,
            "outer_bits_per_frame": n,
            "inner_bits_per_track": n,
            "pad_bits": 0,
        }

    cb = _stateful_candidate_codebook(cfg, cache_dir) if cfg.code.inner_encoder == "stateful_trellis" else _inner_codebook(cfg, cache_dir)
    if fam == "mtr1d":
        pad = (-n) % cb.K
        padded = np.pad(bits, ((0, 0), (0, pad)), constant_values=0)
        groups = padded.reshape(num_frames, -1, cb.K)
        idx = bits_to_indices(groups.reshape(-1, cb.K))
        words = cb.encode_indices(idx).reshape(num_frames, -1, cb.block_len)
        grid = words.reshape(num_frames, -1).astype(np.uint8)
        return grid, {
            "inner_code": cb.name,
            "inner_rate": cb.rate,
            "outer_bits_per_frame": n,
            "inner_bits_per_track": grid.shape[1],
            "blocks_per_frame": groups.shape[1],
            "pad_bits": int(pad * num_frames),
            "pad_bits_per_frame": int(pad),
        }

    if fam == "mtr2d_8x2":
        if cfg.code.inner_encoder == "stateful_trellis":
            cb = _stateful_candidate_codebook(cfg, cache_dir)
            info_K = int(cfg.code.K)
        else:
            info_K = int(cb.K)
        cross = cb.cross
        down = cb.down
        if num_frames % cross != 0:
            raise ValueError(f"num_frames ({num_frames}) must be divisible by inner cross size ({cross})")
        pairs = num_frames // cross
        pair_bits = bits.reshape(pairs, cross, n).reshape(pairs, cross * n)
        pad = (-(cross * n)) % info_K
        padded = np.pad(pair_bits, ((0, 0), (0, pad)), constant_values=0)
        groups = padded.reshape(pairs, -1, info_K)
        if cfg.code.inner_encoder == "stateful_trellis":
            words, stateful_meta = _encode_stateful_trellis_sequences(
                groups, cb, info_K=info_K,
                down_mtr=cfg.code.trellis_boundary_down_mtr,
                max_checker_run_allowed=cfg.code.trellis_boundary_max_checker_run,
            )
            blocks = words.reshape(pairs, -1, cross, down)
            inner_code = (
                f"stateful_mtr2d_{cross}x{down}_K{info_K}_from_{cb.name}"
                f"_bd{cfg.code.trellis_boundary_down_mtr}"
                f"_bc{cfg.code.trellis_boundary_max_checker_run}"
            )
        else:
            idx = bits_to_indices(groups.reshape(-1, info_K))
            blocks = cb.encode_indices(idx).reshape(pairs, -1, cross, down)
            inner_code = cb.name
            stateful_meta = {}
        grid = blocks.transpose(0, 2, 1, 3).reshape(num_frames, -1).astype(np.uint8)
        meta = {
            "inner_code": inner_code,
            "inner_encoder": cfg.code.inner_encoder,
            "inner_rate": float(info_K / cb.block_len),
            "outer_bits_per_frame": n,
            "inner_bits_per_track": grid.shape[1],
            "blocks_per_track_pair": groups.shape[1],
            "pad_bits": int(pad * pairs),
            "pad_bits_per_track_pair": int(pad),
        }
        meta.update(stateful_meta)
        return grid, meta

    raise ValueError(f"unsupported inner family {fam!r}")


def inner_transition_stats(tx_grid: np.ndarray, cfg: Config,
                           cache_dir: str = "data/codebooks") -> Dict:
    """Measure block-boundary transitions that a global trellis would prune.

    A nonzero violation rate means the current transmitter is still a blockwise
    independent encoder for that constraint; a pruned trellis demapper is then an
    unmatched research diagnostic until the encoder is made stateful too.
    """
    bits = np.asarray(tx_grid, dtype=np.uint8)
    fam = cfg.experiment.family
    if fam == "uncoded":
        return {
            "tx_trellis_transition_violations": 0,
            "tx_trellis_transition_count": 0,
            "tx_trellis_transition_violation_rate": 0.0,
        }

    cb = _stateful_candidate_codebook(cfg, cache_dir) if cfg.code.inner_encoder == "stateful_trellis" else _inner_codebook(cfg, cache_dir)
    if fam == "mtr1d":
        if bits.ndim != 2 or bits.shape[1] % cb.block_len:
            raise ValueError("mtr1d tx grid does not align to codebook blocks")
        seq = bits.reshape(bits.shape[0], bits.shape[1] // cb.block_len, cb.block_len)
    elif fam == "mtr2d_8x2":
        cross = cb.cross
        down = cb.down
        if bits.ndim != 2 or bits.shape[0] % cross or bits.shape[1] % down:
            raise ValueError("mtr2d_8x2 tx grid does not align to 2D codebook blocks")
        pairs = bits.shape[0] // cross
        blocks_per_pair = bits.shape[1] // down
        seq = (
            bits.reshape(pairs, cross, blocks_per_pair, down)
            .transpose(0, 2, 1, 3)
            .reshape(pairs, blocks_per_pair, cb.block_len)
        )
    else:
        raise ValueError(f"unsupported inner family {fam!r}")

    violations, total = count_trellis_transition_violations(
        seq, cb, down_mtr=cfg.code.trellis_boundary_down_mtr,
        max_checker_run_allowed=cfg.code.trellis_boundary_max_checker_run,
    )
    return {
        "tx_trellis_transition_violations": violations,
        "tx_trellis_transition_count": total,
        "tx_trellis_transition_violation_rate": float(violations / total) if total else 0.0,
        "trellis_boundary_down_mtr": cfg.code.trellis_boundary_down_mtr,
        "trellis_boundary_max_checker_run": cfg.code.trellis_boundary_max_checker_run,
    }


def decode_inner_llr(llr_grid: np.ndarray, cfg: Config, outer_shape: Tuple[int, int],
                     cache_dir: str = "data/codebooks",
                     apriori_llr: Optional[np.ndarray] = None,
                     extrinsic: bool = True,
                     llr_clip: Optional[float] = None,
                     chunk: int = 64,
                     inner_demapper: str = "exact_codebook") -> Tuple[np.ndarray, Dict]:
    """Soft-demodulate the inner grid back to LDPC-codeword LLRs.

    ``apriori_llr`` is optional LDPC-to-demapper extrinsic information for the
    outer codeword bits. For constrained inner codebooks this is grouped into
    codebook-index bits before exact marginalization.
    """
    llr = np.asarray(llr_grid, dtype=np.float64)
    num_frames, n = outer_shape
    demapper_name = str(inner_demapper or "exact_codebook")
    if demapper_name not in {"exact_codebook", "hard_codebook", "trellis_pruned", "stateful_trellis"}:
        raise ValueError(
            "inner_demapper must be 'exact_codebook', 'hard_codebook', "
            "'trellis_pruned', or 'stateful_trellis'"
        )
    if apriori_llr is None:
        apriori_outer = np.zeros(outer_shape, dtype=np.float64)
    else:
        apriori_outer = np.asarray(apriori_llr, dtype=np.float64)
        if apriori_outer.shape != outer_shape:
            raise ValueError(f"apriori_llr shape {apriori_outer.shape} != outer shape {outer_shape}")
    fam = cfg.experiment.family

    if fam == "uncoded":
        if llr.shape != outer_shape:
            raise ValueError(f"uncoded LLR grid shape {llr.shape} != outer shape {outer_shape}")
        posterior = llr + apriori_outer
        out = posterior - apriori_outer if extrinsic else posterior
        if llr_clip is not None and llr_clip > 0:
            out = np.clip(out, -float(llr_clip), float(llr_clip))
        return out, {"inner_demapper": "identity"}

    cb = _stateful_candidate_codebook(cfg, cache_dir) if cfg.code.inner_encoder == "stateful_trellis" else _inner_codebook(cfg, cache_dir)
    if fam == "mtr1d":
        if llr.shape[0] != num_frames or llr.shape[1] % cb.block_len:
            raise ValueError("mtr1d LLR grid does not align to codebook blocks")
        blocks_per_frame = llr.shape[1] // cb.block_len
        block_seq = llr.reshape(num_frames, blocks_per_frame, cb.block_len)
        blocks = block_seq.reshape(-1, cb.block_len)
        pad = blocks_per_frame * cb.K - n
        ap = np.pad(apriori_outer, ((0, 0), (0, pad)), constant_values=0.0)
        ap_seq = ap.reshape(num_frames, blocks_per_frame, cb.K)
        if demapper_name == "hard_codebook":
            info = hard_codebook_demapper_llr(
                blocks, cb, llr_clip=llr_clip,
            ).reshape(num_frames, blocks_per_frame * cb.K)
            return info[:, :n], {"inner_demapper": "hard_codebook", "demapped_bits_per_frame": n}
        if demapper_name == "exact_codebook":
            info = soft_demapper_llr(
                blocks, cb, apriori_llr=ap_seq.reshape(-1, cb.K), extrinsic=extrinsic,
                llr_clip=llr_clip, chunk=chunk,
            ).reshape(num_frames, blocks_per_frame * cb.K)
            return info[:, :n], {"inner_demapper": "exact_codebook", "demapped_bits_per_frame": n}
        info_seq, meta = trellis_pruned_demapper_llr(
            block_seq, cb, down_mtr=cfg.code.trellis_boundary_down_mtr,
            max_checker_run_allowed=cfg.code.trellis_boundary_max_checker_run,
            apriori_llr=ap_seq, extrinsic=extrinsic, llr_clip=llr_clip,
        )
        info = info_seq.reshape(num_frames, blocks_per_frame * cb.K)
        meta["demapped_bits_per_frame"] = n
        return info[:, :n], meta

    if fam == "mtr2d_8x2":
        stateful_mode = cfg.code.inner_encoder == "stateful_trellis"
        info_K = int(cfg.code.K) if stateful_mode else int(cb.K)
        if stateful_mode and demapper_name != "stateful_trellis":
            raise ValueError("stateful_trellis inner_encoder requires ldpc.inner_demapper='stateful_trellis'")
        cross = cb.cross
        down = cb.down
        if num_frames % cross != 0 or llr.shape[0] != num_frames or llr.shape[1] % down:
            raise ValueError("mtr2d_8x2 LLR grid does not align to 2D codebook blocks")
        pairs = num_frames // cross
        blocks_per_pair = llr.shape[1] // down
        block_seq = (
            llr.reshape(pairs, cross, blocks_per_pair, down)
            .transpose(0, 2, 1, 3)
            .reshape(pairs, blocks_per_pair, cb.block_len)
        )
        blocks = block_seq.reshape(-1, cb.block_len)
        pad = blocks_per_pair * info_K - cross * n
        ap_pairs = apriori_outer.reshape(pairs, cross, n).reshape(pairs, cross * n)
        ap = np.pad(ap_pairs, ((0, 0), (0, pad)), constant_values=0.0)
        ap_seq = ap.reshape(pairs, blocks_per_pair, info_K)
        if stateful_mode:
            info_seq, meta = stateful_trellis_demapper_llr(
                block_seq, cb, info_K=info_K,
                down_mtr=cfg.code.trellis_boundary_down_mtr,
                max_checker_run_allowed=cfg.code.trellis_boundary_max_checker_run,
                apriori_llr=ap_seq, extrinsic=extrinsic, llr_clip=llr_clip,
            )
            info = info_seq.reshape(pairs, blocks_per_pair * info_K)
            info = info[:, :cross * n].reshape(pairs, cross, n).reshape(num_frames, n)
            meta["demapped_bits_per_frame"] = n
            return info, meta
        if demapper_name == "hard_codebook":
            info = hard_codebook_demapper_llr(
                blocks, cb, llr_clip=llr_clip,
            ).reshape(pairs, blocks_per_pair * cb.K)
            info = info[:, :cross * n].reshape(pairs, cross, n).reshape(num_frames, n)
            return info, {"inner_demapper": "hard_codebook", "demapped_bits_per_frame": n}
        if demapper_name == "exact_codebook":
            info = soft_demapper_llr(
                blocks, cb, apriori_llr=ap_seq.reshape(-1, info_K), extrinsic=extrinsic,
                llr_clip=llr_clip, chunk=chunk,
            ).reshape(pairs, blocks_per_pair * cb.K)
            info = info[:, :cross * n].reshape(pairs, cross, n).reshape(num_frames, n)
            return info, {"inner_demapper": "exact_codebook", "demapped_bits_per_frame": n}
        info_seq, meta = trellis_pruned_demapper_llr(
            block_seq, cb, down_mtr=cfg.code.trellis_boundary_down_mtr,
            max_checker_run_allowed=cfg.code.trellis_boundary_max_checker_run,
            apriori_llr=ap_seq, extrinsic=extrinsic, llr_clip=llr_clip,
        )
        info = info_seq.reshape(pairs, blocks_per_pair * cb.K)
        info = info[:, :cross * n].reshape(pairs, cross, n).reshape(num_frames, n)
        meta["demapped_bits_per_frame"] = n
        return info, meta

    raise ValueError(f"unsupported inner family {fam!r}")

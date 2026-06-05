"""LDPC code: parity-check matrix, systematic encoder, and BP/min-sum decoder.

This is the real decoder behind the stage-2 LDPC track. It is self-contained
(numpy only) and deterministic.

Pieces
------
* **Parity-check matrix** -- a regular ``(dv, dc)`` Gallager construction: ``dv``
  ones per column, ``dc`` per row, ``m = n*dv/dc`` checks (requires ``n`` divisible
  by ``dc``). Built from a seeded RNG so a code is reproducible from its params.
* **Systematic encoder** -- GF(2) Gaussian elimination puts ``H`` in reduced form;
  the non-pivot columns carry the ``k = n - rank(H)`` information bits and the
  pivot columns carry parity. ``encode(U)`` is then a GF(2) matrix product.
* **Decoder** -- flooding **belief propagation** in the LLR domain, either
  normalized **min-sum** (default, robust) or **sum-product**, batched over many
  frames with numpy and stopped early once every frame satisfies its syndrome.

LLR convention: this project uses ``positive LLR => bit 1`` everywhere (see
:mod:`tdmr2d.detector`). Internally the decoder converts to the textbook
``L = log P(0)/P(1)`` convention (``L = -llr``) so the min-sum equations are the
standard ones, then converts the decision back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# GF(2) helpers                                                                #
# --------------------------------------------------------------------------- #
def _gf2_systematic(H: np.ndarray) -> Tuple[List[int], List[int], np.ndarray, int]:
    """Reduce ``H`` over GF(2); return (info_cols, parity_cols, Pmat, rank).

    After reduction each pivot (parity) column holds a single 1; the parity bit
    at pivot row ``i`` equals ``Pmat[i] . u`` where ``u`` are the info bits.
    """
    R = (np.asarray(H, dtype=np.uint8) % 2).copy()
    m, n = R.shape
    pivot_cols: List[int] = []
    row = 0
    for col in range(n):
        if row >= m:
            break
        pivot = None
        for r in range(row, m):
            if R[r, col]:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != row:
            R[[row, pivot]] = R[[pivot, row]]
        for r in range(m):
            if r != row and R[r, col]:
                R[r] ^= R[row]
        pivot_cols.append(col)
        row += 1
    rank = row
    parity_set = set(pivot_cols)
    info_cols = [c for c in range(n) if c not in parity_set]
    Pmat = R[:rank][:, info_cols].astype(np.uint8) if info_cols else np.zeros((rank, 0), np.uint8)
    return info_cols, pivot_cols, Pmat, rank


# --------------------------------------------------------------------------- #
# LDPC code                                                                    #
# --------------------------------------------------------------------------- #
@dataclass
class LDPCCode:
    H: np.ndarray            # (m, n) uint8 parity-check matrix
    info_cols: np.ndarray    # length-k info column indices (systematic)
    parity_cols: np.ndarray  # length-r parity column indices
    Pmat: np.ndarray         # (r, k) parity generator over GF(2)

    @property
    def n(self) -> int:
        return int(self.H.shape[1])

    @property
    def m(self) -> int:
        return int(self.H.shape[0])

    @property
    def k(self) -> int:
        return int(len(self.info_cols))

    @property
    def rate(self) -> float:
        return self.k / self.n

    # ---- construction ----------------------------------------------------
    @classmethod
    def gallager(cls, n: int, dv: int = 3, dc: int = 6, seed: int = 1) -> "LDPCCode":
        """Regular ``(dv, dc)`` Gallager parity-check code (``n`` divisible by ``dc``)."""
        if n % dc != 0:
            raise ValueError(f"n ({n}) must be divisible by dc ({dc}) for a Gallager code")
        rows_per_band = n // dc          # rows in each of the dv bands
        rng = np.random.default_rng(seed)

        base = np.zeros((rows_per_band, n), dtype=np.uint8)
        for r in range(rows_per_band):
            base[r, r * dc:(r + 1) * dc] = 1   # dc consecutive ones per row

        bands = [base]
        for _ in range(dv - 1):
            perm = rng.permutation(n)
            bands.append(base[:, perm])
        H = np.vstack(bands).astype(np.uint8)

        info_cols, parity_cols, Pmat, _ = _gf2_systematic(H)
        return cls(H=H, info_cols=np.asarray(info_cols, dtype=np.int64),
                   parity_cols=np.asarray(parity_cols, dtype=np.int64), Pmat=Pmat)

    # ---- encode / syndrome ----------------------------------------------
    def encode(self, info_bits: np.ndarray) -> np.ndarray:
        """Systematically encode ``(F, k)`` info bits into ``(F, n)`` codewords."""
        U = np.asarray(info_bits, dtype=np.uint8) % 2
        if U.shape[-1] != self.k:
            raise ValueError(f"info width {U.shape[-1]} != k {self.k}")
        F = U.shape[0]
        C = np.zeros((F, self.n), dtype=np.uint8)
        C[:, self.info_cols] = U
        if self.Pmat.shape[0]:
            parity = (U @ self.Pmat.T) % 2
            C[:, self.parity_cols] = parity.astype(np.uint8)
        return C

    def syndrome(self, bits: np.ndarray) -> np.ndarray:
        b = np.asarray(bits, dtype=np.uint8) % 2
        return (b @ self.H.T) % 2

    def info_bits(self, codewords: np.ndarray) -> np.ndarray:
        return np.asarray(codewords)[:, self.info_cols]

    # ---- decode ----------------------------------------------------------
    def decode_llr(self, llr: np.ndarray, max_iters: int = 30, method: str = "minsum",
                   scale: float = 0.75, chunk: int = 64) -> np.ndarray:
        """Decode per-bit LLRs (``positive => bit 1``) to ``(F, n)`` hard bits.

        Processes frames in chunks to bound memory. Uses flooding BP with early
        stop when all frames in a chunk satisfy their syndrome.
        """
        bits, _ = self.decode_llr_with_posterior(
            llr, max_iters=max_iters, method=method, scale=scale, chunk=chunk
        )
        return bits

    def decode_llr_with_posterior(self, llr: np.ndarray, max_iters: int = 30,
                                  method: str = "minsum", scale: float = 0.75,
                                  chunk: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        """Decode LLRs and return ``(hard_bits, posterior_llr)``.

        ``posterior_llr`` uses the project convention positive => bit 1. It lets
        a concatenated/turbo receiver form LDPC extrinsic feedback as
        ``posterior_llr - input_llr``.
        """
        L = np.asarray(llr, dtype=np.float64)
        F, n = L.shape
        if n != self.n:
            raise ValueError(f"llr width {n} != n {self.n}")
        out = np.empty((F, n), dtype=np.uint8)
        posterior = np.empty((F, n), dtype=np.float64)
        for s in range(0, F, chunk):
            e = min(s + chunk, F)
            bits, Ltot = self._decode_batch(
                -L[s:e], max_iters, method, scale, return_ltot=True
            )
            out[s:e] = bits
            posterior[s:e] = -Ltot  # back to project convention: positive => bit 1
        return out, posterior

    def _decode_batch(self, Lch: np.ndarray, max_iters: int, method: str,
                      scale: float, return_ltot: bool = False):
        Hb = (self.H.astype(bool))[None]          # (1, m, n)
        Hbits = self.H.astype(np.uint8)
        B = Lch.shape[0]
        m, n = self.H.shape
        E = np.zeros((B, m, n), dtype=np.float64)  # check -> var messages

        bits = (Lch < 0).astype(np.uint8)
        for _ in range(max_iters):
            colsum = E.sum(axis=1)                              # (B, n)
            M = (Lch + colsum)[:, None, :] - E                 # var -> check (B, m, n)
            M = np.where(Hb, M, 0.0)

            if method == "minsum":
                absM = np.where(Hb, np.abs(M), np.inf)
                sign = np.where(Hb, np.sign(M), 1.0)
                sign = np.where(sign == 0.0, 1.0, sign)
                tsign = np.prod(sign, axis=2)                  # (B, m)
                min1 = absM.min(axis=2)                        # (B, m)
                amin = absM.argmin(axis=2)
                tmp = absM.copy()
                np.put_along_axis(tmp, amin[..., None], np.inf, axis=2)
                min2 = tmp.min(axis=2)                         # (B, m)
                mag = np.where(absM == min1[..., None], min2[..., None], min1[..., None])
                E = np.where(Hb, (tsign[..., None] * sign) * mag * scale, 0.0)
            else:  # sum-product (tanh rule)
                t = np.where(Hb, np.tanh(np.clip(M / 2.0, -30, 30)), 1.0)
                tprod = np.prod(t, axis=2)[..., None]          # (B, m, 1)
                ratio = np.divide(tprod, t, out=np.zeros_like(t), where=(np.abs(t) > 1e-12))
                ratio = np.clip(ratio, -1 + 1e-12, 1 - 1e-12)
                E = np.where(Hb, 2.0 * np.arctanh(ratio), 0.0)

            Ltot = Lch + E.sum(axis=1)
            bits = (Ltot < 0).astype(np.uint8)
            if not (bits @ Hbits.T % 2).any():
                break

        Ltot = Lch + E.sum(axis=1)
        bits = (Ltot < 0).astype(np.uint8)
        if return_ltot:
            return bits, Ltot
        return bits

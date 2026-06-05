"""Constrained-code codebooks (Pattern B = 1D-MTR, Pattern C = 2D-MTR 8x2).

A codebook maps ``K`` information bits to one constraint-satisfying channel block:

* **1D (mtr1d)** -- a length-``L`` down-track block, applied independently per
  track. ``rate = K1 / L``.
* **2D (mtr2d_8x2)** -- a ``cross x down`` (default 2x8 = 16-bit) block.
  ``rate = K / 16``.

Codewords are enumerated deterministically (sorted by integer value) and the
first ``2**K`` valid blocks are selected, so a codebook is a pure function of its
parameters and is safe to cache. Cached codebooks live under
``data/codebooks/`` as ``.npz`` (+ a human-readable ``.json`` sidecar).

External codebooks (e.g. a hand-designed high-rate K14/K15 8x2 table) can be
attached via :func:`load_codebook` without changing any call site -- this is the
"connect a K14/K15 codebook" extension point.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from . import constraints as C

DEFAULT_CACHE_DIR = "data/codebooks"


class CodebookError(ValueError):
    """Raised when a codebook cannot satisfy the requested size."""


@dataclass
class Codebook:
    name: str
    kind: str                 # "mtr1d" | "mtr2d_8x2"
    K: int                    # information bits per block
    block_len: int            # channel bits per block (L for 1D, cross*down for 2D)
    cross: int                # tracks spanned by a block (1 for 1D, 2 for 8x2)
    down: int                 # down-track bits spanned by a block
    words: np.ndarray         # (2**K, block_len) uint8, track-major for 2D
    meta: Dict = field(default_factory=dict)

    @property
    def size(self) -> int:
        return int(self.words.shape[0])

    @property
    def rate(self) -> float:
        return self.K / self.block_len

    def encode_indices(self, idx: np.ndarray) -> np.ndarray:
        """Map an array of info indices in ``[0, 2**K)`` to codeword rows."""
        return self.words[idx]

    # --- persistence ------------------------------------------------------
    def save(self, cache_dir: str | Path = DEFAULT_CACHE_DIR) -> Path:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        npz = cache_dir / f"{self.name}.npz"
        meta = dict(self.meta)
        meta.update(
            name=self.name, kind=self.kind, K=self.K, block_len=self.block_len,
            cross=self.cross, down=self.down, size=self.size, rate=self.rate,
        )
        np.savez_compressed(npz, words=self.words, meta=json.dumps(meta))
        with open(cache_dir / f"{self.name}.json", "w") as fh:
            json.dump(meta, fh, indent=2)
        return npz


def load_codebook(path: str | Path) -> Codebook:
    """Load a codebook from a ``.npz`` produced by :meth:`Codebook.save`."""
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    return Codebook(
        name=meta["name"], kind=meta["kind"], K=int(meta["K"]),
        block_len=int(meta["block_len"]), cross=int(meta["cross"]),
        down=int(meta["down"]), words=data["words"].astype(np.uint8), meta=meta,
    )


# --------------------------------------------------------------------------- #
# builders                                                                     #
# --------------------------------------------------------------------------- #
def build_1d_codebook(K1: int, length: int = 8, down_mtr: int = 3) -> Codebook:
    """Enumerate valid length-``length`` 1D-MTR blocks; keep the first ``2**K1``."""
    valid = [v for v in range(1 << length) if C.satisfies_mtr_1d(C.bits_down(v, length), down_mtr)]
    valid.sort()
    need = 1 << K1
    if len(valid) < need:
        raise CodebookError(
            f"1D-MTR(length={length}, down_mtr={down_mtr}) has {len(valid)} valid words "
            f"but K1={K1} needs {need}"
        )
    chosen = valid[:need]
    words = np.stack([C.bits_down(v, length) for v in chosen]).astype(np.uint8)
    name = f"mtr1d_L{length}_K{K1}_m{down_mtr}"
    meta = {"num_valid": len(valid), "down_mtr": down_mtr, "length": length}
    return Codebook(name=name, kind="mtr1d", K=K1, block_len=length, cross=1,
                    down=length, words=words, meta=meta)


def build_2d_codebook(K: int, cross: int = 2, down: int = 8, down_mtr: int = 3,
                      max_checker_run: int = 0) -> Codebook:
    """Enumerate valid ``cross x down`` 2D-MTR blocks; keep the first ``2**K``.

    Words are stored track-major flattened so ``word.reshape(cross, down)`` gives
    the block in ``(track, down-track)`` orientation.
    """
    block_len = cross * down
    valid = []
    for v in range(1 << block_len):
        b = C.block_of(v, cross, down)
        if C.satisfies_mtr2d_block(b, down_mtr, max_checker_run):
            valid.append(v)
    valid.sort()
    need = 1 << K
    if len(valid) < need:
        raise CodebookError(
            f"2D-MTR({cross}x{down}, down_mtr={down_mtr}, max_checker_run={max_checker_run}) "
            f"has {len(valid)} valid words but K={K} needs {need}. "
            f"Relax max_checker_run (e.g. 1) or down_mtr to raise the count."
        )
    chosen = valid[:need]
    words = np.stack([C.block_of(v, cross, down).reshape(-1) for v in chosen]).astype(np.uint8)
    name = f"mtr2d_{cross}x{down}_K{K}_m{down_mtr}_c{max_checker_run}"
    meta = {"num_valid": len(valid), "down_mtr": down_mtr,
            "max_checker_run": max_checker_run, "cross": cross, "down": down}
    return Codebook(name=name, kind="mtr2d_8x2", K=K, block_len=block_len, cross=cross,
                    down=down, words=words, meta=meta)


# --------------------------------------------------------------------------- #
# cached accessors (used by experiments)                                       #
# --------------------------------------------------------------------------- #
def get_1d_codebook(K1: int, length: int = 8, down_mtr: int = 3,
                    cache_dir: str | Path = DEFAULT_CACHE_DIR) -> Codebook:
    name = f"mtr1d_L{length}_K{K1}_m{down_mtr}"
    npz = Path(cache_dir) / f"{name}.npz"
    if npz.exists():
        return load_codebook(npz)
    cb = build_1d_codebook(K1=K1, length=length, down_mtr=down_mtr)
    cb.save(cache_dir)
    return cb


def get_2d_codebook(K: int, cross: int = 2, down: int = 8, down_mtr: int = 3,
                    max_checker_run: int = 0,
                    cache_dir: str | Path = DEFAULT_CACHE_DIR) -> Codebook:
    name = f"mtr2d_{cross}x{down}_K{K}_m{down_mtr}_c{max_checker_run}"
    npz = Path(cache_dir) / f"{name}.npz"
    if npz.exists():
        return load_codebook(npz)
    cb = build_2d_codebook(K=K, cross=cross, down=down, down_mtr=down_mtr,
                           max_checker_run=max_checker_run)
    cb.save(cache_dir)
    return cb

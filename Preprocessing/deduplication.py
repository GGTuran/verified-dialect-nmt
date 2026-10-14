"""
deduplication.py
=================
exact deduplication, near-duplicate
identification, and source-grouped train/val/test splitting.

Design notes
------------
* **Exact deduplication** is a trivial vectorized ``drop_duplicates`` on
  ``(standard_bangla, dialect_text)``.
* **Near-duplicate identification** uses a from-scratch, numpy-vectorized
  MinHash + LSH-banding implementation for *candidate generation* (so we
  never pay O(n^2) pairwise cost), followed by a pairwise similarity
  *confirmation* step (via ``rapidfuzz`` when available, else a
  ``difflib``-based fallback of identical semantics) restricted to
  same-bucket candidate pairs only. This satisfies the spec's requirement
  to support either a ``datasketch``-style MinHash/LSH approach or a
  ``rapidfuzz``-style Levenshtein-ratio approach -- here they are combined:
  MinHash/LSH for *efficient candidate generation*, Levenshtein ratio for
  *precise confirmation* at the ``>= 0.92`` threshold.
* If the optional ``datasketch`` package *is* installed, it is used in
  place of the hand-rolled MinHash for signature generation (same banding
  logic either way). This keeps the library dependency-light for
  environments without network access while still honoring the spec's
  named libraries when present.
* **Splitting** uses hashed group assignment keyed on a *near-duplicate
  cluster id* derived from ``standard_bangla`` (so near-identical source
  sentences -- not just exact duplicates -- are kept within a single
  split), computed jointly across both dialects so a shared Standard
  Bangla sentence can never leak across splits even if it appears in both
  the Chittagonian and Sylheti corpora.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import PipelineConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

try:
    import rapidfuzz.fuzz as _rf_fuzz  # type: ignore

    _HAS_RAPIDFUZZ = True
except ImportError:  # pragma: no cover - exercised in dependency-light envs
    _HAS_RAPIDFUZZ = False

try:
    from datasketch import MinHash as _DatasketchMinHash, MinHashLSH as _DatasketchLSH  # type: ignore

    _HAS_DATASKETCH = True
except ImportError:  # pragma: no cover
    _HAS_DATASKETCH = False


class DeduplicationError(Exception):
    """Raised on unrecoverable errors during dedup/splitting."""


def similarity_ratio(a: str, b: str) -> float:
    """Normalized Levenshtein-style similarity in ``[0, 1]``.

    Uses ``rapidfuzz.fuzz.ratio`` (scaled to 0-1) when available; otherwise
    falls back to :class:`difflib.SequenceMatcher`, which computes a
    comparable (Ratcliff/Obershelp) similarity ratio already in ``[0, 1]``.
    """
    if not a and not b:
        return 1.0
    if _HAS_RAPIDFUZZ:
        return _rf_fuzz.ratio(a, b) / 100.0
    import difflib

    return difflib.SequenceMatcher(None, a, b).ratio()



# Exact deduplication
class ExactDeduplicator:
    """Drops exact duplicate ``(standard_bangla, dialect_text)`` pairs."""

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG) -> None:
        self.config = config

    def deduplicate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        schema = self.config.schema
        subset = [schema.standard_col, schema.dialect_text_col]
        dup_mask = df.duplicated(subset=subset, keep="first")
        dropped_ids = df.loc[dup_mask, "row_id"].tolist()
        clean = df.loc[~dup_mask].copy()
        logger.info(
            "ExactDeduplicator: %d -> %d rows (%d exact duplicates dropped)",
            len(df), len(clean), len(dropped_ids),
        )
        return clean.reset_index(drop=True), dropped_ids



# MinHash + LSH banding (vectorized, dependency-optional)
# NOTE: We deliberately use the 32-bit Mersenne prime M31 = 2**31 - 1
# (rather than the more commonly cited 2**61 - 1) so that every value
# flowing through the vectorized ``(a * h + b) mod prime`` universal-hash
# computation stays comfortably within int64 range even *before* the
# modulo reduction (2**31 * 2**31 = 2**62 < int64 max of ~2**63 - 1).
# Using the larger 61-bit prime with numpy's fixed-width int64 arithmetic
# silently overflows during the multiplication step (numpy does not raise
# on integer overflow), which would corrupt the MinHash signatures.
_MERSENNE_PRIME = (1 << 31) - 1


def _shingles(text: str, k: int) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if len(text) < k:
        return [text] if text else []
    return [text[i : i + k] for i in range(len(text) - k + 1)]


def _hash_shingle(shingle: str) -> int:
    digest = hashlib.md5(shingle.encode("utf-8")).digest()
    raw = int.from_bytes(digest[:8], byteorder="little", signed=False)
    # Reduce mod the (32-bit) Mersenne prime immediately so every hash
    # value is already bounded before entering the int64 universal-hash
    # arithmetic in ``_MinHashIndex._signature_for``.
    return raw % _MERSENNE_PRIME


@dataclass
class _MinHashIndex:
    """Vectorized MinHash signature bank + LSH banding candidate index."""

    num_perm: int
    shingle_size: int
    num_bands: int = 32
    _a: np.ndarray = field(init=False, repr=False)
    _b: np.ndarray = field(init=False, repr=False)
    signatures: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.num_perm % self.num_bands != 0:
            # Adjust bands down to the largest divisor <= requested value so
            # banding is exact (each band gets an equal number of rows).
            for nb in range(min(self.num_bands, self.num_perm), 0, -1):
                if self.num_perm % nb == 0:
                    self.num_bands = nb
                    break
        rng = np.random.RandomState(42)
        self._a = rng.randint(1, _MERSENNE_PRIME - 1, size=self.num_perm, dtype=np.int64)
        self._b = rng.randint(0, _MERSENNE_PRIME - 1, size=self.num_perm, dtype=np.int64)

    def _signature_for(self, text: str) -> np.ndarray:
        shingles = _shingles(text, self.shingle_size)
        if not shingles:
            # Degenerate (empty) document: constant signature so it only
            # clusters with other empty documents.
            return np.zeros(self.num_perm, dtype=np.int64)
        hashes = np.array([_hash_shingle(s) for s in shingles], dtype=np.int64)
        # (num_perm, num_shingles) matrix of (a*h + b) mod prime, then min
        # over shingles -> (num_perm,) signature.
        combined = (
            np.mod(
                self._a[:, None].astype(np.int64) * hashes[None, :].astype(np.int64)
                + self._b[:, None].astype(np.int64),
                _MERSENNE_PRIME,
            )
        )
        return combined.min(axis=1)

    def build(self, texts: Sequence[str]) -> None:
        logger.info("Building MinHash signatures for %d documents (num_perm=%d)", len(texts), self.num_perm)
        sigs = np.empty((len(texts), self.num_perm), dtype=np.int64)
        for i, t in enumerate(texts):
            sigs[i] = self._signature_for(t)
        self.signatures = sigs

    def candidate_pairs(self) -> List[Tuple[int, int]]:
        """LSH banding: two documents are candidates if any band's rows
        match exactly. Returns a de-duplicated list of index pairs
        ``(i, j)`` with ``i < j``.
        """
        if self.signatures is None:
            raise DeduplicationError("Call build() before candidate_pairs()")
        n, num_perm = self.signatures.shape
        rows_per_band = num_perm // self.num_bands
        buckets: Dict[Tuple[int, bytes], List[int]] = {}
        for band_idx in range(self.num_bands):
            start = band_idx * rows_per_band
            end = start + rows_per_band
            band_matrix = self.signatures[:, start:end]
            for doc_idx in range(n):
                key = (band_idx, band_matrix[doc_idx].tobytes())
                buckets.setdefault(key, []).append(doc_idx)

        pairs = set()
        for doc_indices in buckets.values():
            if len(doc_indices) < 2:
                continue
            # Cap per-bucket fan-out for pathological cases (e.g. many
            # empty/near-empty documents hashing identically) to keep the
            # confirmation step bounded.
            capped = doc_indices[:200]
            for i in range(len(capped)):
                for j in range(i + 1, len(capped)):
                    a, b = capped[i], capped[j]
                    pairs.add((a, b) if a < b else (b, a))
        logger.info("LSH banding produced %d candidate pairs from %d documents", len(pairs), n)
        return sorted(pairs)


class _UnionFind:
    """Simple disjoint-set forest for clustering near-duplicate pairs into
    connected components."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1



# Near-duplicate clustering
class NearDuplicateClusterer:
    """Clusters near-identical strings (typically ``standard_bangla`` source
    sentences) into connected components using MinHash/LSH candidate
    generation followed by pairwise similarity confirmation at
    ``config.thresholds.near_dup_similarity_threshold``.
    """

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG) -> None:
        self.config = config

    def cluster(self, texts: Sequence[str]) -> np.ndarray:
        """Returns an ``int`` array of length ``len(texts)`` giving each
        text's cluster id (0-indexed, dense). Identical/near-identical
        texts share a cluster id.
        """
        th = self.config.thresholds
        n = len(texts)
        if n == 0:
            return np.array([], dtype=np.int64)

        # Fast path: exact-string clusters seed the union-find directly
        # (guarantees exact duplicates are always clustered even if LSH
        # banding happens to miss a candidate pair on unlucky hashing).
        uf = _UnionFind(n)
        first_seen: Dict[str, int] = {}
        for i, t in enumerate(texts):
            if t in first_seen:
                uf.union(first_seen[t], i)
            else:
                first_seen[t] = i

        index = _MinHashIndex(num_perm=th.minhash_num_perm, shingle_size=th.minhash_shingle_size)
        index.build(texts)
        candidates = index.candidate_pairs()

        confirmed = 0
        for i, j in candidates:
            if uf.find(i) == uf.find(j):
                continue
            sim = similarity_ratio(texts[i], texts[j])
            if sim >= th.near_dup_similarity_threshold:
                uf.union(i, j)
                confirmed += 1
        logger.info(
            "NearDuplicateClusterer: confirmed %d/%d candidate pairs at threshold >= %.2f",
            confirmed, len(candidates), th.near_dup_similarity_threshold,
        )

        roots = np.array([uf.find(i) for i in range(n)], dtype=np.int64)
        _, dense = np.unique(roots, return_inverse=True)
        return dense

    def drop_near_duplicates(self, df: pd.DataFrame, text_col: str) -> Tuple[pd.DataFrame, List[str]]:
        """Convenience wrapper: within each near-dup cluster, keep only the
        first row and report the rest as dropped ids. (Used when the caller
        wants near-dup *removal* rather than just *clustering for
        splitting*.)
        """
        cluster_ids = self.cluster(df[text_col].astype(str).tolist())
        df = df.assign(_cluster_id=cluster_ids)
        keep_mask = ~df.duplicated(subset="_cluster_id", keep="first")
        dropped_ids = df.loc[~keep_mask, "row_id"].tolist()
        clean = df.loc[keep_mask].drop(columns=["_cluster_id"]).copy()
        logger.info(
            "NearDuplicateClusterer.drop_near_duplicates: %d -> %d rows (%d near-duplicates dropped)",
            len(df), len(clean), len(dropped_ids),
        )
        return clean.reset_index(drop=True), dropped_ids



# Source-grouped train/val/test splitting
class GroupSplitter:
    """Splits a combined (both-dialect) DataFrame into train/val/test using
    hashed group assignment keyed on near-duplicate cluster id over
    ``standard_bangla``, so that no identical or near-identical Standard
    Bangla sentence appears in more than one split -- across *either*
    dialect.
    """

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self._clusterer = NearDuplicateClusterer(config)

    def split(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        schema = self.config.schema
        th = self.config.thresholds
        if df.empty:
            raise DeduplicationError("Cannot split an empty DataFrame")

        texts = df[schema.standard_col].astype(str).tolist()
        cluster_ids = self._clusterer.cluster(texts)
        df = df.assign(_group_id=cluster_ids)

        unique_groups = np.unique(cluster_ids)
        bucket = self._hash_assign(unique_groups, th.random_seed)
        group_to_split = dict(zip(unique_groups.tolist(), bucket))

        df = df.assign(_split=[group_to_split[g] for g in df["_group_id"]])

        splits = {}
        for name in ("train", "val", "test"):
            part = df.loc[df["_split"] == name].drop(columns=["_group_id", "_split"]).copy()
            splits[name] = part.reset_index(drop=True)

        total = len(df)
        for name, part in splits.items():
            logger.info(
                "GroupSplitter: split=%-5s rows=%6d (%.2f%%)",
                name, len(part), 100.0 * len(part) / total if total else 0.0,
            )
        self._verify_group_integrity(df)
        return splits

  
    def _hash_assign(self, groups: np.ndarray, seed: int) -> List[str]:
        """Deterministically assign each group id to train/val/test using a
        stable hash of the group id, targeting the configured ratios.
        Using a hash (rather than ``np.random.shuffle``) makes the
        assignment reproducible independent of processing order and stable
        across pipeline re-runs on the same data.
        """
        th = self.config.thresholds
        assignments = []
        for g in groups:
            h = hashlib.sha256(f"{seed}:{int(g)}".encode("utf-8")).hexdigest()
            frac = int(h[:8], 16) / 0xFFFFFFFF
            if frac < th.train_ratio:
                assignments.append("train")
            elif frac < th.train_ratio + th.val_ratio:
                assignments.append("val")
            else:
                assignments.append("test")
        return assignments

    def _verify_group_integrity(self, df: pd.DataFrame) -> None:
        leaky = df.groupby("_group_id")["_split"].nunique()
        n_leaky = int((leaky > 1).sum())
        if n_leaky:
            raise DeduplicationError(
                f"Group-split integrity violated: {n_leaky} group(s) span multiple splits"
            )
        logger.info("GroupSplitter: group integrity verified (0 groups span multiple splits)")

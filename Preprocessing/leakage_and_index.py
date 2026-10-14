"""
leakage_and_index.py
=====================
train/test contamination auditing and
BM25 retrieval-index construction, plus export of cleaned splits and
indices to Parquet/JSONL.

Both the leakage audit and the BM25 index avoid full O(n_train * n_test)
or O(n^2) pairwise work by using an inverted n-gram index for candidate
generation, mirroring the MinHash/LSH candidate-generation strategy used
in :mod:`deduplication`.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import PipelineConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi as _RankBM25Okapi  # type: ignore

    _HAS_RANK_BM25 = True
except ImportError:  # pragma: no cover
    _HAS_RANK_BM25 = False

try:
    import pyarrow  # noqa: F401  # type: ignore

    _HAS_PARQUET_ENGINE = True
except ImportError:  # pragma: no cover
    try:
        import fastparquet  # noqa: F401  # type: ignore

        _HAS_PARQUET_ENGINE = True
    except ImportError:
        _HAS_PARQUET_ENGINE = False


class LeakageAuditError(Exception):
    """Raised when the leakage audit detects a hard contamination violation
    (currently: any nonzero exact-match leakage between train and test)."""


class IndexError_(Exception):
    """Raised on retrieval-index construction failures."""



# Tokenization / n-gram helpers
_WORD_SPLIT_RE = re.compile(r"\s+")


def word_tokenize(text: str) -> List[str]:
    text = str(text).strip()
    if not text:
        return []
    return [t for t in _WORD_SPLIT_RE.split(text) if t]


def char_ngrams(text: str, n: int) -> List[str]:
    text = re.sub(r"\s+", " ", str(text).strip())
    if len(text) < n:
        return [text] if text else []
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def word_ngrams(tokens: Sequence[str], n: int) -> List[str]:
    if len(tokens) < n:
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)




# Contamination audit
@dataclass
class LeakageReport:
    exact_match_leakage_count: int
    exact_match_leakage_pct: float
    exact_match_offending_ids: List[str]
    char_ngram_jaccard: np.ndarray  # per-test-row max similarity to train
    word_ngram_jaccard: np.ndarray
    flagged_test_ids: List[str]  # jaccard > threshold (char OR word)
    n_test: int
    n_train: int

    def summary(self) -> str:
        lines = [
            "Leakage audit summary",
            "----------------------",
            f"train rows: {self.n_train}",
            f"test rows:  {self.n_test}",
            f"exact-match leakage: {self.exact_match_leakage_count} "
            f"({self.exact_match_leakage_pct:.4f}%)",
            f"char-ngram jaccard: mean={float(np.mean(self.char_ngram_jaccard)):.4f} "
            f"p95={float(np.percentile(self.char_ngram_jaccard, 95)) if len(self.char_ngram_jaccard) else 0:.4f} "
            f"max={float(np.max(self.char_ngram_jaccard)) if len(self.char_ngram_jaccard) else 0:.4f}",
            f"word-ngram jaccard: mean={float(np.mean(self.word_ngram_jaccard)):.4f} "
            f"p95={float(np.percentile(self.word_ngram_jaccard, 95)) if len(self.word_ngram_jaccard) else 0:.4f} "
            f"max={float(np.max(self.word_ngram_jaccard)) if len(self.word_ngram_jaccard) else 0:.4f}",
            f"flagged test rows (jaccard > threshold): {len(self.flagged_test_ids)}",
        ]
        return "\n".join(lines)


class LeakageAuditor:
    """Audits a train/test split pair for exact-match and near-match
    (n-gram Jaccard) contamination."""

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG) -> None:
        self.config = config

    def audit(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        text_col: Optional[str] = None,
        strict: bool = True,
    ) -> LeakageReport:
        schema = self.config.schema
        th = self.config.thresholds
        text_col = text_col or schema.standard_col

        train_texts = train_df[text_col].astype(str).tolist()
        test_texts = test_df[text_col].astype(str).tolist()
        test_ids = test_df["row_id"].astype(str).tolist()

        exact_count, exact_ids = self._exact_match_leakage(train_texts, test_texts, test_ids)
        exact_pct = 100.0 * exact_count / len(test_texts) if test_texts else 0.0

        char_sims, word_sims, flagged_ids = self._ngram_jaccard_leakage(
            train_texts, test_texts, test_ids, th.char_ngram_n, th.word_ngram_n, th.leakage_jaccard_threshold
        )

        report = LeakageReport(
            exact_match_leakage_count=exact_count,
            exact_match_leakage_pct=exact_pct,
            exact_match_offending_ids=exact_ids,
            char_ngram_jaccard=char_sims,
            word_ngram_jaccard=word_sims,
            flagged_test_ids=flagged_ids,
            n_test=len(test_texts),
            n_train=len(train_texts),
        )
        logger.info("\n%s", report.summary())

        if strict and exact_count > 0:
            raise LeakageAuditError(
                f"Exact-match leakage detected: {exact_count} test row(s) "
                f"({exact_pct:.4f}%) exactly match a train row. Row ids: "
                f"{exact_ids[:10]}{'...' if len(exact_ids) > 10 else ''}"
            )
        return report

    
    def _exact_match_leakage(
        self, train_texts: List[str], test_texts: List[str], test_ids: List[str]
    ) -> Tuple[int, List[str]]:
        train_set = set(train_texts)
        offending = [tid for t, tid in zip(test_texts, test_ids) if t in train_set]
        return len(offending), offending

    def _ngram_jaccard_leakage(
        self,
        train_texts: List[str],
        test_texts: List[str],
        test_ids: List[str],
        char_n: int,
        word_n: int,
        threshold: float,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        # Precompute train n-gram sets + inverted index (ngram -> train row
        # indices) for candidate generation, so we never do a full
        # n_train * n_test comparison.
        train_char_sets = [set(char_ngrams(t, char_n)) for t in train_texts]
        train_word_sets = [set(word_ngrams(word_tokenize(t), word_n)) for t in train_texts]

        char_inv: Dict[str, List[int]] = defaultdict(list)
        for idx, s in enumerate(train_char_sets):
            for g in s:
                char_inv[g].append(idx)

        char_sims = np.zeros(len(test_texts), dtype=np.float64)
        word_sims = np.zeros(len(test_texts), dtype=np.float64)
        flagged: List[str] = []

        for i, (t, tid) in enumerate(zip(test_texts, test_ids)):
            test_char_set = set(char_ngrams(t, char_n))
            test_word_set = set(word_ngrams(word_tokenize(t), word_n))

            candidate_idx = set()
            for g in test_char_set:
                candidate_idx.update(char_inv.get(g, ()))

            best_char = 0.0
            best_word = 0.0
            for c_idx in candidate_idx:
                cs = jaccard(test_char_set, train_char_sets[c_idx])
                if cs > best_char:
                    best_char = cs
                ws = jaccard(test_word_set, train_word_sets[c_idx])
                if ws > best_word:
                    best_word = ws

            char_sims[i] = best_char
            word_sims[i] = best_word
            if best_char > threshold or best_word > threshold:
                flagged.append(tid)

        return char_sims, word_sims, flagged


#  BM25 retrieval index
class _FallbackBM25:
    """Minimal, dependency-free BM25 (Okapi) implementation used when
    ``rank_bm25`` is not installed. API-compatible subset of
    ``rank_bm25.BM25Okapi`` (``get_scores``, ``get_top_n``).
    """

    def __init__(self, corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus_tokens = corpus_tokens
        self.doc_lens = np.array([len(doc) for doc in corpus_tokens], dtype=np.float64)
        self.avgdl = float(self.doc_lens.mean()) if len(corpus_tokens) else 0.0
        self.n_docs = len(corpus_tokens)

        self.doc_freqs: List[Counter] = [Counter(doc) for doc in corpus_tokens]
        df: Counter = Counter()
        for doc in corpus_tokens:
            for term in set(doc):
                df[term] += 1
        self.idf: Dict[str, float] = {}
        for term, freq in df.items():
            # BM25 idf with +1 smoothing to keep values non-negative.
            self.idf[term] = np.log(1 + (self.n_docs - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query_tokens: List[str]) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=np.float64)
        for term in query_tokens:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for i in range(self.n_docs):
                f = self.doc_freqs[i].get(term, 0)
                if f == 0:
                    continue
                dl = self.doc_lens[i]
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                scores[i] += idf * (f * (self.k1 + 1)) / (denom or 1.0)
        return scores

    def get_top_n(self, query_tokens: List[str], documents: List[str], n: int = 5) -> List[str]:
        scores = self.get_scores(query_tokens)
        top_idx = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_idx]


@dataclass
class BM25FieldIndex:
    field_name: str
    tokens: List[List[str]]
    model: object  # BM25Okapi or _FallbackBM25
    row_ids: List[str]

    def query(self, text: str, top_n: int = 5) -> List[Tuple[str, float]]:
        q_tokens = word_tokenize(text)
        scores = self.model.get_scores(q_tokens)
        order = np.argsort(scores)[::-1][:top_n]
        return [(self.row_ids[i], float(scores[i])) for i in order]


class BM25Indexer:
    """Builds BM25 retrieval indices over the training ``standard_bangla``
    and ``dialect_text`` fields, to support translation-memory / RAG-style
    retrieval for dialect translation.
    """

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self.indices: Dict[str, BM25FieldIndex] = {}

    def build(self, train_df: pd.DataFrame) -> Dict[str, BM25FieldIndex]:
        schema = self.config.schema
        th = self.config.thresholds
        for field_name in (schema.standard_col, schema.dialect_text_col):
            texts = train_df[field_name].astype(str).tolist()
            row_ids = train_df["row_id"].astype(str).tolist()
            tokens = [word_tokenize(t) for t in texts]

            if _HAS_RANK_BM25:
                model = _RankBM25Okapi(tokens, k1=th.bm25_k1, b=th.bm25_b)
            else:
                model = _FallbackBM25(tokens, k1=th.bm25_k1, b=th.bm25_b)

            self.indices[field_name] = BM25FieldIndex(
                field_name=field_name, tokens=tokens, model=model, row_ids=row_ids
            )
            logger.info(
                "BM25Indexer: built index for field=%s over %d docs (engine=%s)",
                field_name, len(texts), "rank_bm25" if _HAS_RANK_BM25 else "fallback",
            )
        return self.indices

    def query(self, field_name: str, text: str, top_n: int = 5) -> List[Tuple[str, float]]:
        if field_name not in self.indices:
            raise IndexError_(f"No BM25 index built for field {field_name!r}")
        return self.indices[field_name].query(text, top_n=top_n)



# Export: partitioned Parquet + JSONL
class ArtifactExporter:
    """Exports cleaned splits and retrieval indices to disk as partitioned
    Parquet (when a parquet engine is available) and JSONL (always)."""

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG) -> None:
        self.config = config

    def export_splits(self, splits: Dict[str, pd.DataFrame], dialect_name: str) -> None:
        out_dir = self.config.paths.outputs_dir / "splits"
        out_dir.mkdir(parents=True, exist_ok=True)
        for split_name, df in splits.items():
            base = out_dir / f"{dialect_name}_{split_name}"
            self._write_jsonl(df, base.with_suffix(".jsonl"))
            self._write_parquet(df, base.with_suffix(".parquet"), partition_label=f"{dialect_name}/{split_name}")

    def export_bm25_index(self, indexer: BM25Indexer, dialect_name: str) -> None:
        out_dir = self.config.paths.outputs_dir / "index"
        out_dir.mkdir(parents=True, exist_ok=True)
        for field_name, field_index in indexer.indices.items():
            payload = {
                "field": field_name,
                "dialect": dialect_name,
                "engine": "rank_bm25" if _HAS_RANK_BM25 else "fallback",
                "row_ids": field_index.row_ids,
                "tokens": field_index.tokens,
            }
            path = out_dir / f"{dialect_name}_{field_name}_bm25.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            logger.info("Exported BM25 index artifact: %s", path)

    
    def _write_jsonl(self, df: pd.DataFrame, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for record in df.to_dict(orient="records"):
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Exported JSONL: %s (%d rows)", path, len(df))

    def _write_parquet(self, df: pd.DataFrame, path: Path, partition_label: str) -> None:
        if not _HAS_PARQUET_ENGINE:
            logger.warning(
                "No parquet engine (pyarrow/fastparquet) available in this "
                "environment; skipping Parquet export for %s. JSONL export "
                "already covers this partition (%s).",
                path.name, partition_label,
            )
            return
        try:
            df.to_parquet(path, index=False)
            logger.info("Exported Parquet: %s (%d rows)", path, len(df))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Parquet export failed for %s: %s", path, exc)

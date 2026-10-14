"""
preprocessor.py
================

Two public classes:

* :class:`DatasetIngestor` -- loads a raw ``.xlsx`` workbook, validates its
  schema, casts to UTF-8 strings, strips control characters and renames
  columns into the canonical schema defined in :mod:`config`.
* :class:`QualityFilter` -- drops/flags empty rows, machine-translation
  artifacts, untranslated-Bengali-in-English rows, identity pairs and
  pairs whose dialect/standard length ratio falls outside the configured
  guardrail. Produces a full diagnostic :class:`AuditLog`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import PipelineConfig, DEFAULT_CONFIG
from normalizer import BengaliTextNormalizer

logger = logging.getLogger(__name__)



# Custom exceptions
class SchemaValidationError(Exception):
    """Raised when a raw workbook does not contain the expected columns."""


class IngestionError(Exception):
    """Raised when a raw workbook cannot be loaded or parsed."""


class FilteringError(Exception):
    """Raised when the quality-filtering stage encounters an unrecoverable
    error (e.g. an empty DataFrame where a non-empty one is required)."""



# Audit log
@dataclass
class AuditLog:
    """Diagnostic audit trail tracking rows dropped/flagged at each stage of
    the pipeline. Row ids are preserved (not just counts) so that a
    reviewer can trace exactly which sentence pairs were removed and why.
    """

    dialect_name: str
    stage_dropped_ids: Dict[str, List[str]] = field(default_factory=dict)
    stage_flagged_ids: Dict[str, List[str]] = field(default_factory=dict)

    def log_dropped(self, stage: str, row_ids: List[str]) -> None:
        if not row_ids:
            logger.debug("[%s] stage=%s dropped 0 rows", self.dialect_name, stage)
            return
        self.stage_dropped_ids.setdefault(stage, [])
        self.stage_dropped_ids[stage].extend(row_ids)
        logger.info("[%s] stage=%s dropped %d rows", self.dialect_name, stage, len(row_ids))

    def log_flagged(self, stage: str, row_ids: List[str]) -> None:
        if not row_ids:
            return
        self.stage_flagged_ids.setdefault(stage, [])
        self.stage_flagged_ids[stage].extend(row_ids)
        logger.info("[%s] stage=%s flagged %d rows", self.dialect_name, stage, len(row_ids))

    def total_dropped(self) -> int:
        return sum(len(v) for v in self.stage_dropped_ids.values())

    def summary_table(self) -> pd.DataFrame:
        rows = []
        for stage, ids in self.stage_dropped_ids.items():
            rows.append({"stage": stage, "kind": "dropped", "count": len(ids)})
        for stage, ids in self.stage_flagged_ids.items():
            rows.append({"stage": stage, "kind": "flagged", "count": len(ids)})
        if not rows:
            return pd.DataFrame(columns=["stage", "kind", "count"])
        return pd.DataFrame(rows).sort_values(["kind", "stage"]).reset_index(drop=True)

    def to_jsonl(self, path: Path) -> None:
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for stage, ids in self.stage_dropped_ids.items():
                f.write(json.dumps({"stage": stage, "kind": "dropped", "row_ids": ids}, ensure_ascii=False) + "\n")
            for stage, ids in self.stage_flagged_ids.items():
                f.write(json.dumps({"stage": stage, "kind": "flagged", "row_ids": ids}, ensure_ascii=False) + "\n")



# Ingestion & structural validation
class DatasetIngestor:
    """Loads and structurally validates a single dialect's raw Excel
    workbook, producing a DataFrame in the canonical Kothon schema:
    ``['row_id', 'standard_bangla', 'translated_english', 'dialect_text',
    'dialect_name']``.
    """

    # Control characters \x00-\x1F excluding \n (\x0A) and \t (\x09).
    _CONTROL_CHARS = "".join(chr(c) for c in range(0x00, 0x20) if c not in (0x09, 0x0A))

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG) -> None:
        self.config = config

    def load(self, path: Path, dialect_key: str) -> pd.DataFrame:
        """Load one workbook (``dialect_key`` in {"chittagonian", "sylheti"})
        and return a canonically-schema'd, UTF-8-clean DataFrame.
        """
        path = Path(path)
        if not path.exists():
            raise IngestionError(f"Workbook not found: {path}")

        try:
            raw_df = pd.read_excel(path, engine="openpyxl")
        except Exception as exc:
            raise IngestionError(f"Failed to read Excel workbook {path}: {exc}") from exc

        logger.info("Loaded %s: %d rows, %d columns", path.name, *raw_df.shape)

        self._validate_schema(raw_df, dialect_key, path)
        df = self._rename_to_canonical(raw_df, dialect_key)
        df = self._enforce_utf8_and_strip_control_chars(df)

        dialect_name = self.config.schema.raw_dialect_cols.get(dialect_key, dialect_key).split()[0]
        df[self.config.schema.dialect_name_col] = dialect_key
        df.insert(0, "row_id", [f"{dialect_key}_{i}" for i in range(len(df))])
        return df.reset_index(drop=True)

    
    def _validate_schema(self, df: pd.DataFrame, dialect_key: str, path: Path) -> None:
        schema = self.config.schema
        expected_dialect_col = schema.raw_dialect_cols.get(dialect_key)
        if expected_dialect_col is None:
            raise SchemaValidationError(
                f"Unknown dialect_key {dialect_key!r}; expected one of "
                f"{list(schema.raw_dialect_cols)}"
            )

        # Case-insensitive / whitespace-tolerant matching against a small
        # set of accepted aliases so minor header drift doesn't hard-fail
        # ingestion.
        normalized_cols = {c.strip().lower(): c for c in df.columns}
        required = {
            "standard_bangla": [schema.raw_standard_col.lower()],
            "translated_english": [schema.raw_english_col.lower()],
            "dialect_text": [expected_dialect_col.lower()],
        }
        missing = []
        resolved: Dict[str, str] = {}
        for canonical_key, aliases in required.items():
            found = None
            for alias in aliases:
                if alias in normalized_cols:
                    found = normalized_cols[alias]
                    break
            if found is None:
                missing.append((canonical_key, aliases))
            else:
                resolved[canonical_key] = found

        if missing:
            raise SchemaValidationError(
                f"{path.name}: missing expected column(s) "
                f"{[m[0] for m in missing]} (looked for aliases {missing}); "
                f"found columns were {list(df.columns)}"
            )
        self._resolved_columns = resolved  # stashed for _rename_to_canonical

    def _rename_to_canonical(self, df: pd.DataFrame, dialect_key: str) -> pd.DataFrame:
        schema = self.config.schema
        rename_map = {
            self._resolved_columns["standard_bangla"]: schema.standard_col,
            self._resolved_columns["translated_english"]: schema.english_col,
            self._resolved_columns["dialect_text"]: schema.dialect_text_col,
        }
        out = df.rename(columns=rename_map)
        keep = [schema.standard_col, schema.english_col, schema.dialect_text_col]
        return out[keep].copy()

    def _enforce_utf8_and_strip_control_chars(self, df: pd.DataFrame) -> pd.DataFrame:
        control_table = str.maketrans("", "", self._CONTROL_CHARS)

        def _clean(value) -> str:
            if pd.isna(value):
                return np.nan
            s = str(value)
            # Enforce UTF-8 round-trip; replace undecodable bytes rather than
            # raising, since source material spans OCR/scan pipelines that
            # can introduce mojibake.
            s = s.encode("utf-8", errors="replace").decode("utf-8")
            return s.translate(control_table)

        for col in df.columns:
            df[col] = df[col].map(_clean)
        return df



# apply the BengaliTextNormalizer across the ingested frame
def apply_normalization(
    df: pd.DataFrame,
    normalizer: BengaliTextNormalizer,
    text_columns: Tuple[str, ...] = ("standard_bangla", "dialect_text"),
) -> pd.DataFrame:
    """Applies :class:`BengaliTextNormalizer` to the given text columns of
    ``df`` in place (returns the same, mutated DataFrame for chaining).
    English-only columns are intentionally excluded by default since the
    Bengali-specific Nukta/Khanda-Ta rules do not apply to Latin script.
    """
    df = df.copy()
    for col in text_columns:
        if col not in df.columns:
            continue
        df[col] = normalizer.normalize_series(df[col].astype("object"))
    return df



# Parallel quality & anomaly filtering
class QualityFilter:
    """Applies the Phase-3 quality/anomaly filters to a canonically-schema'd,
    normalized DataFrame, producing a cleaned DataFrame plus a full
    :class:`AuditLog` of everything dropped or flagged, and why.
    """

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG) -> None:
        self.config = config

    def filter(self, df: pd.DataFrame, dialect_name: str) -> Tuple[pd.DataFrame, AuditLog, pd.DataFrame]:
        """Run every Phase-3 filter in sequence.

        Returns
        -------
        clean_df:
            The surviving rows.
        audit:
            :class:`AuditLog` with per-stage dropped/flagged row ids.
        identity_df:
            Rows where ``standard_bangla == dialect_text`` (returned
            separately per the "detect and separate" requirement, rather
            than silently discarded).
        """
        if df.empty:
            raise FilteringError(f"[{dialect_name}] Cannot filter an empty DataFrame")

        schema = self.config.schema
        audit = AuditLog(dialect_name=dialect_name)
        working = df.copy()

        working = self._drop_empty_rows(working, audit)
        working = self._drop_english_artifacts(working, audit)
        working, identity_df = self._separate_identity_rows(working, audit)
        working = self._apply_length_ratio_guardrail(working, audit)

        logger.info(
            "[%s] Phase 3 complete: %d -> %d rows (%d dropped, %d identity-separated)",
            dialect_name,
            len(df),
            len(working),
            audit.total_dropped(),
            len(identity_df),
        )
        return working.reset_index(drop=True), audit, identity_df.reset_index(drop=True)

    
    def _drop_empty_rows(self, df: pd.DataFrame, audit: AuditLog) -> pd.DataFrame:
        schema = self.config.schema
        cols = [schema.standard_col, schema.english_col, schema.dialect_text_col]

        def _is_blank(v) -> bool:
            return pd.isna(v) or (isinstance(v, str) and v.strip() == "")

        # DataFrame.applymap was removed in pandas 2.1+ in favor of .map;
        # try the modern API first and fall back for older pandas.
        try:
            blank_mask = df[cols].map(_is_blank).any(axis=1)
        except AttributeError:  # pragma: no cover - older pandas
            blank_mask = df[cols].applymap(_is_blank).any(axis=1)
        dropped_ids = df.loc[blank_mask, "row_id"].tolist()
        audit.log_dropped("empty_or_null_rows", dropped_ids)
        return df.loc[~blank_mask].copy()

    def _drop_english_artifacts(self, df: pd.DataFrame, audit: AuditLog) -> pd.DataFrame:
        schema = self.config.schema
        artifacts = self.config.artifacts
        english = df[schema.english_col].astype(str)

        marker_pattern = "|".join(
            _re_escape_upper(m) for m in artifacts.literal_markers
        )
        has_marker = english.str.upper().str.contains(marker_pattern, regex=True, na=False)
        has_bengali = english.str.contains(artifacts.bengali_script_regex, regex=True, na=False)
        bad_mask = has_marker | has_bengali

        dropped_ids = df.loc[bad_mask, "row_id"].tolist()
        audit.log_dropped("english_mt_artifact_or_untranslated", dropped_ids)
        return df.loc[~bad_mask].copy()

    def _separate_identity_rows(self, df: pd.DataFrame, audit: AuditLog) -> Tuple[pd.DataFrame, pd.DataFrame]:
        schema = self.config.schema
        identity_mask = (
            df[schema.standard_col].str.strip() == df[schema.dialect_text_col].str.strip()
        )
        identity_ids = df.loc[identity_mask, "row_id"].tolist()
        audit.log_flagged("identity_standard_eq_dialect", identity_ids)
        identity_df = df.loc[identity_mask].copy()
        remaining = df.loc[~identity_mask].copy()
        return remaining, identity_df

    def _apply_length_ratio_guardrail(self, df: pd.DataFrame, audit: AuditLog) -> pd.DataFrame:
        schema = self.config.schema
        th = self.config.thresholds

        std = df[schema.standard_col].astype(str)
        dia = df[schema.dialect_text_col].astype(str)

        std_word_len = std.str.split().map(len).replace(0, np.nan)
        dia_word_len = dia.str.split().map(len)
        word_ratio = dia_word_len / std_word_len

        std_char_len = std.str.len().replace(0, np.nan)
        dia_char_len = dia.str.len()
        char_ratio = dia_char_len / std_char_len

        df = df.assign(_word_ratio=word_ratio, _char_ratio=char_ratio)

        in_range = (
            (df["_word_ratio"] >= th.min_length_ratio)
            & (df["_word_ratio"] <= th.max_length_ratio)
            & (df["_char_ratio"] >= th.min_length_ratio)
            & (df["_char_ratio"] <= th.max_length_ratio)
        )
        # Rows with a NaN ratio (i.e. zero-length standard_bangla, which
        # should already have been dropped as "empty") are treated as
        # out-of-range defensively.
        in_range = in_range.fillna(False)

        dropped_ids = df.loc[~in_range, "row_id"].tolist()
        audit.log_dropped("length_ratio_guardrail", dropped_ids)
        return df.loc[in_range].drop(columns=["_word_ratio", "_char_ratio"])


def _re_escape_upper(literal: str) -> str:
    import re

    return re.escape(literal.upper())

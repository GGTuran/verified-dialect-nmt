"""
config.py
Centralized configuration for the Kothon Bengali Dialect MT dataset pipeline.

Uses `dataclasses` (rather than pydantic) for configuration/metadata validation,
per the architecture spec's allowance of "pydantic (v2) OR Python @dataclass".
This keeps the library dependency-light while still giving typed, validated,
immutable configuration objects with sensible defaults and `__post_init__`
validation.

All paths, hyper-parameters and thresholds used across the pipeline live here
so that every stage (ingestion, normalization, filtering, deduplication,
splitting, leakage audit, indexing, visualization) reads from a single
source of truth.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ConfigValidationError(ValueError):
    """Raised when a configuration object fails validation."""


# Paths
@dataclass(frozen=True)
class PathsConfig:
    """Filesystem layout used by the pipeline.

    Attributes
    ----------
    project_root:
        Root directory of the pipeline project.
    data_dir:
        Directory containing the raw ``Chittagonian.xlsx`` / ``Sylheti.xlsx``
        source files.
    chittagonian_xlsx:
        Path to the raw Chittagonian Excel workbook.
    sylheti_xlsx:
        Path to the raw Sylheti Excel workbook.
    outputs_dir:
        Directory where cleaned splits, indices and audit logs are written.
    figures_dir:
        Directory where publication-quality figures (PNG, 300 DPI) are written.
    """

    current_dir = Path(__file__).resolve().parent

    project_root: Path = current_dir.parent
    # D:\Practise\LRNLP\data\Chittagonian.xlsx
    # data/Sylheti.xlsx
    data_dir: Path = project_root / "data"
    chittagonian_xlsx: Path = data_dir / "Chittagonian.xlsx"
    sylheti_xlsx: Path = data_dir / "Sylheti.xlsx"
    outputs_dir: Path = project_root / "outputs"
    figures_dir: Path = project_root / "figures"

    def __post_init__(self) -> None:
        # Normalize everything to Path objects (handles str inputs too).
        object.__setattr__(self, "project_root", Path(self.project_root))
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        object.__setattr__(self, "chittagonian_xlsx", Path(self.chittagonian_xlsx))
        object.__setattr__(self, "sylheti_xlsx", Path(self.sylheti_xlsx))
        object.__setattr__(self, "outputs_dir", Path(self.outputs_dir))
        object.__setattr__(self, "figures_dir", Path(self.figures_dir))

    def ensure_dirs(self) -> None:
        """Create all output-side directories (idempotent)."""
        for d in (self.outputs_dir, self.figures_dir):
            d.mkdir(parents=True, exist_ok=True)
        for sub in ("splits", "index", "audit"):
            (self.outputs_dir / sub).mkdir(parents=True, exist_ok=True)


# Column schema
@dataclass(frozen=True)
class SchemaConfig:
    """Canonical, standardized column schema used after Phase 1 ingestion."""

    standard_col: str = "standard_bangla"
    english_col: str = "translated_english"
    dialect_text_col: str = "dialect_text"
    dialect_name_col: str = "dialect_name"

    # Expected *raw* column names per source workbook case-insensitive match is attempted
    raw_standard_col: str = "Standard Bangla"
    raw_english_col: str = "Translated English"
    raw_dialect_cols: Dict[str, str] = field(
        default_factory=lambda: {
            "chittagonian": "Chittagonian Dialect",
            "sylheti": "Sylheti Dialect",
        }
    )

    @property
    def canonical_columns(self) -> List[str]:
        return [
            self.standard_col,
            self.english_col,
            self.dialect_text_col,
            self.dialect_name_col,
        ]


# Thresholds / hyper-parameters
@dataclass(frozen=True)
class ThresholdConfig:
    """Numeric thresholds and hyper-parameters governing filtering,
    deduplication, splitting and leakage auditing."""

    # length ratio guardrail: len(dialect) / len(standard)
    min_length_ratio: float = 0.33
    max_length_ratio: float = 3.0

    # near-duplicate detection
    near_dup_similarity_threshold: float = 0.92  # rapidfuzz-style [0,1] ratio
    minhash_num_perm: int = 128
    minhash_shingle_size: int = 4  # character shingle size
    lsh_threshold: float = 0.85

    #  split ratios (must sum to 1.0)
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42

    # leakage audit
    leakage_jaccard_threshold: float = 0.70
    char_ngram_n: int = 4
    word_ngram_n: int = 2

    #  BM25
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    def __post_init__(self) -> None:
        if not (0.0 < self.min_length_ratio < self.max_length_ratio):
            raise ConfigValidationError(
                "min_length_ratio must be > 0 and < max_length_ratio "
                f"(got min={self.min_length_ratio}, max={self.max_length_ratio})"
            )
        if not (0.0 <= self.near_dup_similarity_threshold <= 1.0):
            raise ConfigValidationError(
                "near_dup_similarity_threshold must be in [0, 1] "
                f"(got {self.near_dup_similarity_threshold})"
            )
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ConfigValidationError(
                f"train/val/test ratios must sum to 1.0 (got {total})"
            )
        if not (0.0 <= self.leakage_jaccard_threshold <= 1.0):
            raise ConfigValidationError(
                "leakage_jaccard_threshold must be in [0, 1] "
                f"(got {self.leakage_jaccard_threshold})"
            )



# MT-artifact patterns (English column cleaning)
@dataclass(frozen=True)
class ArtifactPatternsConfig:
    """Regex-ish literal substrings / patterns flagging machine-translation
    API artifacts leaking into the English column."""

    literal_markers: Tuple[str, ...] = (
        "MYMEMORY WARNING",
        "QUOTA EXCEEDED",
        "HTTP 429",
        "INVALID SOURCE LANGUAGE",
        "PLEASE SELECT TWO DISTINCT LANGUAGES",
        "AMBIGUOUS INPUT",
        "TRANSLATION UNAVAILABLE",
        "[ERROR]",
    )
    # Any Bengali script codepoint inside the English column is suspicious.
    bengali_script_regex: str = r"[\u0980-\u09FF]"



# Top-level pipeline configuration
@dataclass(frozen=True)
class PipelineConfig:
    """Aggregate configuration object passed through the whole pipeline."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    schema: SchemaConfig = field(default_factory=SchemaConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    artifacts: ArtifactPatternsConfig = field(default_factory=ArtifactPatternsConfig)
    log_level: str = "INFO"

    def summary(self) -> str:
        lines = ["PipelineConfig summary:"]
        for f in dataclasses.fields(self):
            lines.append(f"  {f.name}: {getattr(self, f.name)!r}")
        return "\n".join(lines)


DEFAULT_CONFIG = PipelineConfig()

"""
main.py
=======
CLI orchestrator for the Kothon Bengali Dialect MT dataset pipeline.

Pulls together every stage defined across the other modules:

    Phase 1  ingestion & structural validation      -> preprocessor.DatasetIngestor
    Phase 2  Bengali text normalization              -> normalizer.BengaliTextNormalizer
    Phase 3  parallel quality & anomaly filtering     -> preprocessor.QualityFilter
    Phase 4  dedup + source-grouped splitting         -> deduplication.*
    Phase 5  leakage audit + BM25 index + export      -> leakage_and_index.*
    Phase 6  publication-quality visualization suite  -> visualizer.VisualizationSuite

Usage
-----
    python main.py --chittagonian-xlsx data/Chittagonian.xlsx \\
                    --sylheti-xlsx data/Sylheti.xlsx \\
                    --output-dir outputs --figures-dir figures

    python main.py --help
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config import (
    ArtifactPatternsConfig,
    PathsConfig,
    PipelineConfig,
    SchemaConfig,
    ThresholdConfig,
)
from normalizer import BengaliTextNormalizer
from preprocessor import AuditLog, DatasetIngestor, QualityFilter, apply_normalization
from deduplication import ExactDeduplicator, GroupSplitter, NearDuplicateClusterer
from leakage_and_index import ArtifactExporter, BM25Indexer, LeakageAuditor, word_tokenize
from visualizer import VisualizationSuite

# --------------------------------------------------------------------------- #
# Logging setup ("rich console logs" with a graceful plain-logging fallback,
# since `rich` may not be installed in every deployment environment).
# --------------------------------------------------------------------------- #
try:
    from rich.logging import RichHandler  # type: ignore
    from rich.console import Console  # type: ignore

    _HAS_RICH = True
except ImportError:  # pragma: no cover
    _HAS_RICH = False


def _configure_logging(level: str) -> logging.Logger:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level.upper())

    if _HAS_RICH:
        handler = RichHandler(rich_tracebacks=True, markup=False, show_path=False)
        fmt = "%(message)s"
    else:
        handler = logging.StreamHandler(sys.stdout)
        fmt = "%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s"
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    root.addHandler(handler)
    return logging.getLogger("kothon.main")


logger = logging.getLogger("kothon.main")


class PipelineError(Exception):
    """Raised when the end-to-end orchestration fails unrecoverably."""


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class DialectStageResult:
    dialect_key: str
    n_raw: int
    n_after_quality_filter: int
    n_identity_flagged: int
    n_after_exact_dedup: int
    n_after_near_dedup: int
    audit: AuditLog


@dataclass
class PipelineResult:
    stage_results: Dict[str, DialectStageResult] = field(default_factory=dict)
    splits: Dict[str, pd.DataFrame] = field(default_factory=dict)  # combined, both dialects
    leakage_report: object = None
    figures: List[Path] = field(default_factory=list)
    elapsed_seconds: float = 0.0


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
class KothonPipeline:
    """End-to-end pipeline orchestrator. Instantiate with a
    :class:`PipelineConfig` and call :meth:`run`.
    """

    #: Hard-coded validator agreement results from the Kothon paper
    #: (Data in Brief, Table 6) -- used to render Figure 6. These are not
    #: derived from the working data (the raw per-sentence validator
    #: annotations are not part of the released dataset), so they are
    #: supplied as a static reference constant here.
    # VALIDATION_AGREEMENT = {
    #     "Chittagonian": {
    #         "Completely Agree": 66.66,
    #         "Spelling Error": 2.67,
    #         "Word Disagree": 30.00,
    #         "Completely Disagree": 0.67,
    #     },
    #     "Sylheti": {
    #         "Completely Agree": 72.00,
    #         "Spelling Error": 3.67,
    #         "Word Disagree": 22.67,
    #         "Completely Disagree": 1.66,
    #     },
    # }

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.config.paths.ensure_dirs()
        self.normalizer = BengaliTextNormalizer()
        self.ingestor = DatasetIngestor(config)
        self.quality_filter = QualityFilter(config)
        self.exact_dedup = ExactDeduplicator(config)
        self.near_dup_clusterer = NearDuplicateClusterer(config)
        self.group_splitter = GroupSplitter(config)
        self.leakage_auditor = LeakageAuditor(config)
        self.exporter = ArtifactExporter(config)
        self.viz = VisualizationSuite(config)

    # ------------------------------------------------------------------ #
    def run(self) -> PipelineResult:
        t0 = time.time()
        result = PipelineResult()

        dialect_frames: Dict[str, pd.DataFrame] = {}
        for dialect_key, path in (
            ("chittagonian", self.config.paths.chittagonian_xlsx),
            ("sylheti", self.config.paths.sylheti_xlsx),
        ):
            logger.info("=" * 70)
            logger.info("Processing dialect: %s (%s)", dialect_key, path)
            clean_df, stage_result = self._process_single_dialect(dialect_key, path)
            dialect_frames[dialect_key] = clean_df
            result.stage_results[dialect_key] = stage_result

        # saving the preprocessed dataset
        preprocessed_dir = Path("preprocessed_data")
        preprocessed_dir.mkdir(parents=True, exist_ok=True)

        for dialect_key, df in dialect_frames.items():
            out_path = preprocessed_dir / f"{dialect_key}_preprocessed.csv"
            df.to_csv(out_path, index=False)
            logger.info("[%s] Saved preprocessed dataset to %s", dialect_key, out_path)

        logger.info("=" * 70)
        logger.info("Combining dialects for group-safe splitting (Phase 4.3)")
        combined = pd.concat(dialect_frames.values(), ignore_index=True)
        combined["row_id"] = combined["dialect_name"] + "_" + combined.index.astype(str)
        splits = self.group_splitter.split(combined)
        result.splits = splits

        for dialect_key in dialect_frames:
            per_dialect_splits = {
                name: df.loc[df["dialect_name"] == dialect_key].reset_index(drop=True)
                for name, df in splits.items()
            }
            self.exporter.export_splits(per_dialect_splits, dialect_key)

            indexer = BM25Indexer(self.config)
            if len(per_dialect_splits["train"]):
                indexer.build(per_dialect_splits["train"])
                self.exporter.export_bm25_index(indexer, dialect_key)
            else:
                logger.warning("[%s] empty train split; skipping BM25 index build", dialect_key)

        logger.info("=" * 70)
        logger.info("Running Phase 5 leakage audit on combined train/test")
        result.leakage_report = self.leakage_auditor.audit(
            splits["train"], splits["test"], strict=False
        )

        logger.info("=" * 70)
        logger.info("Generating publication-quality figures (Phase 6)")
        result.figures = self._generate_figures(dialect_frames, splits, result)

        self._write_run_report(result)

        result.elapsed_seconds = time.time() - t0
        logger.info("Pipeline completed in %.2f seconds", result.elapsed_seconds)
        return result

    # ------------------------------------------------------------------ #
    def _process_single_dialect(self, dialect_key: str, path: Path) -> "tuple[pd.DataFrame, DialectStageResult]":
        raw_df = self.ingestor.load(path, dialect_key)
        n_raw = len(raw_df)

        normalized_df = apply_normalization(raw_df, self.normalizer)

        filtered_df, audit, identity_df = self.quality_filter.filter(normalized_df, dialect_key)
        self._write_audit(audit, dialect_key)

        deduped_df, dropped_exact_ids = self.exact_dedup.deduplicate(filtered_df)
        audit.log_dropped("exact_duplicate", dropped_exact_ids)

        # Phase 4.2: near-duplicate identification/removal, clustering on the
        # full (standard_bangla, dialect_text) pair so we only collapse rows
        # whose *entire sentence pair* is near-identical (as opposed to
        # Phase 4.3's split-safety grouping, which clusters on
        # standard_bangla alone).
        schema = self.config.schema
        pair_key = deduped_df[schema.standard_col] + " \u2016 " + deduped_df[schema.dialect_text_col]
        keyed_df = deduped_df.assign(_pair_key=pair_key)
        near_deduped_df, dropped_near_ids = self.near_dup_clusterer.drop_near_duplicates(
            keyed_df.rename(columns={"row_id": "row_id"}), "_pair_key"
        )
        near_deduped_df = near_deduped_df.drop(columns=["_pair_key"])
        audit.log_dropped("near_duplicate", dropped_near_ids)

        stage_result = DialectStageResult(
            dialect_key=dialect_key,
            n_raw=n_raw,
            n_after_quality_filter=len(filtered_df),
            n_identity_flagged=len(identity_df),
            n_after_exact_dedup=len(deduped_df),
            n_after_near_dedup=len(near_deduped_df),
            audit=audit,
        )
        logger.info(
            "[%s] raw=%d -> quality-filtered=%d (identity-flagged=%d) -> exact-deduped=%d -> near-deduped=%d",
            dialect_key, n_raw, len(filtered_df), len(identity_df), len(deduped_df), len(near_deduped_df),
        )
        return near_deduped_df, stage_result

    def _write_audit(self, audit: AuditLog, dialect_key: str) -> None:
        audit_dir = self.config.paths.outputs_dir / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit.to_jsonl(audit_dir / f"{dialect_key}_audit.jsonl")
        summary_path = audit_dir / f"{dialect_key}_audit_summary.csv"
        audit.summary_table().to_csv(summary_path, index=False)
        logger.info("[%s] audit written to %s", dialect_key, summary_path)

    # ------------------------------------------------------------------ #
    def _generate_figures(
        self,
        dialect_frames: Dict[str, pd.DataFrame],
        splits: Dict[str, pd.DataFrame],
        result: PipelineResult,
    ) -> List[Path]:
        schema = self.config.schema
        figures: List[Path] = []

        # -- Figure 1: sentence length distribution -------------------- #
        combined_std = pd.concat([df[schema.standard_col] for df in dialect_frames.values()])
        word_counts_by_group = {
            "Standard Bangla": combined_std.str.split().map(len).tolist(),
            "Chittagonian": dialect_frames["chittagonian"][schema.dialect_text_col].str.split().map(len).tolist(),
            "Sylheti": dialect_frames["sylheti"][schema.dialect_text_col].str.split().map(len).tolist(),
        }
        figures.append(self.viz.plot_sentence_length_distribution(word_counts_by_group))

        # -- Figure 2: length-ratio scatter (one per dialect) ----------- #
        for dialect_key, label in (("chittagonian", "Chittagonian"), ("sylheti", "Sylheti")):
            df = dialect_frames[dialect_key]
            figures.append(
                self.viz.plot_length_ratio_scatter(
                    df[schema.standard_col].str.split().map(len).tolist(),
                    df[schema.dialect_text_col].str.split().map(len).tolist(),
                    dialect_label=label,
                    filename=f"fig2_length_ratio_scatter_{dialect_key}.png",
                )
            )

        # -- Figure 3: Zipfian vocabulary distribution ------------------ #
        tokens_by_group = {
            "Standard Bangla": [tok for text in combined_std for tok in word_tokenize(text)],
            "Chittagonian": [
                tok for text in dialect_frames["chittagonian"][schema.dialect_text_col] for tok in word_tokenize(text)
            ],
            "Sylheti": [
                tok for text in dialect_frames["sylheti"][schema.dialect_text_col] for tok in word_tokenize(text)
            ],
        }
        figures.append(self.viz.plot_zipfian_distribution(tokens_by_group))

        # -- Figure 4: cross-dialect lexical overlap --------------------- #
        vocab_by_group = {g: set(toks) for g, toks in tokens_by_group.items()}
        figures.append(self.viz.plot_lexical_overlap(vocab_by_group))

        # -- Figure 5: anomaly audit -------------------------------------- #
        figures.append(self.viz.plot_anomaly_audit(self.normalizer.audit_report()))

        # -- Figure 6: annotator agreement -------------------------------- #
        # figures.append(self.viz.plot_annotator_agreement(self.VALIDATION_AGREEMENT))

        # -- Figure 7: cross-split similarity ------------------------------ #
        report = result.leakage_report
        figures.append(
            self.viz.plot_cross_split_similarity(
                report.char_ngram_jaccard, report.word_ngram_jaccard
            )
        )

        return figures

    def _write_run_report(self, result: PipelineResult) -> None:
        report_path = self.config.paths.outputs_dir / "audit" / "pipeline_run_report.json"
        payload = {
            "config": {
                "thresholds": self.config.thresholds.__dict__,
                "schema": {
                    k: v for k, v in self.config.schema.__dict__.items() if not isinstance(v, dict)
                },
            },
            "stages": {
                key: {
                    "n_raw": sr.n_raw,
                    "n_after_quality_filter": sr.n_after_quality_filter,
                    "n_identity_flagged": sr.n_identity_flagged,
                    "n_after_exact_dedup": sr.n_after_exact_dedup,
                    "n_after_near_dedup": sr.n_after_near_dedup,
                    "total_dropped": sr.audit.total_dropped(),
                }
                for key, sr in result.stage_results.items()
            },
            "splits": {name: len(df) for name, df in result.splits.items()},
            "leakage": {
                "exact_match_leakage_count": result.leakage_report.exact_match_leakage_count,
                "exact_match_leakage_pct": result.leakage_report.exact_match_leakage_pct,
                "flagged_test_rows": len(result.leakage_report.flagged_test_ids),
            }
            if result.leakage_report is not None
            else None,
            "anomaly_counts": self.normalizer.audit_report(),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info("Run report written to %s", report_path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kothon-pipeline",
        description="End-to-end preprocessing, validation, deduplication, "
        "splitting, indexing and visualization for the Kothon Bengali "
        "Dialect MT dataset.",
    )
    parser.add_argument("--chittagonian-xlsx", type=Path, default=Path("data/Chittagonian.xlsx"))
    parser.add_argument("--sylheti-xlsx", type=Path, default=Path("data/Sylheti.xlsx"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument(
        "--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    parser.add_argument(
        "--near-dup-threshold", type=float, default=0.92,
        help="Levenshtein-ratio confirmation threshold for near-duplicate clustering (default: 0.92)",
    )
    parser.add_argument(
        "--leakage-threshold", type=float, default=0.70,
        help="N-gram Jaccard threshold above which a test row is flagged as leaking (default: 0.70)",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def build_config_from_args(args: argparse.Namespace) -> PipelineConfig:
    paths = PathsConfig(
        chittagonian_xlsx=args.chittagonian_xlsx,
        sylheti_xlsx=args.sylheti_xlsx,
        outputs_dir=args.output_dir,
        figures_dir=args.figures_dir,
    )
    thresholds = ThresholdConfig(
        near_dup_similarity_threshold=args.near_dup_threshold,
        leakage_jaccard_threshold=args.leakage_threshold,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
    )
    return PipelineConfig(paths=paths, schema=SchemaConfig(), thresholds=thresholds,
                           artifacts=ArtifactPatternsConfig(), log_level=args.log_level)


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)

    try:
        config = build_config_from_args(args)
    except Exception as exc:
        logger.error("Invalid configuration: %s", exc)
        return 2

    logger.info("Starting Kothon pipeline")
    logger.info("%s", config.summary())

    try:
        pipeline = KothonPipeline(config)
        result = pipeline.run()
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        return 1

    logger.info("=" * 70)
    logger.info("SUMMARY")
    for key, sr in result.stage_results.items():
        logger.info(
            "  %-14s raw=%-6d filtered=%-6d exact_deduped=%-6d near_deduped=%-6d",
            key, sr.n_raw, sr.n_after_quality_filter, sr.n_after_exact_dedup, sr.n_after_near_dedup,
        )
    for name, df in result.splits.items():
        logger.info("  split=%-6s rows=%d", name, len(df))
    logger.info("  figures written: %d", len(result.figures))
    logger.info("  total elapsed: %.2fs", result.elapsed_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

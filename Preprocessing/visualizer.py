"""
visualizer.py
=============
Publication-quality (300 DPI) visual analytics suite for the Kothon
dataset pipeline. Produces the 7 figures described in the pipeline spec:

1. Sentence length distribution (KDE + histogram) across dialects.
2. Source-vs-target length ratio scatter/hexbin with marginal KDEs.
3. Zipfian vocabulary rank-vs-frequency (log-log).
4. Cross-dialect lexical overlap (3-way Venn / fallback).
5. Character & Unicode anomaly audit (horizontal bar chart).
6. Annotator agreement/disagreement distribution (stacked bar).
7. Train/val/test cross-split similarity (box plot + CDF).

Every function is standalone (takes plain data structures, not pipeline
objects) so the suite can be reused/tested independently of the rest of
the pipeline, and every function returns the ``Path`` it wrote to.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe backend for CLI/pipeline use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle

from config import PipelineConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

try:
    from matplotlib_venn import venn3  # type: ignore

    _HAS_VENN = True
except ImportError:  # pragma: no cover
    _HAS_VENN = False

_DPI = 300
sns.set_theme(style="whitegrid", context="paper")
_PALETTE = {
    "Standard Bangla": "#2C3E50",
    "Chittagonian": "#E67E22",
    "Sylheti": "#2980B9",
}


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)
    return path


class VisualizationSuite:
    """Namespace/organizer for the 7 Kothon publication figures. Holds only
    a :class:`config.PipelineConfig` (for the output directory + thresholds
    used in figure 2's boundary lines); all data is passed per-call.
    """

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self.config.paths.figures_dir.mkdir(parents=True, exist_ok=True)

    
    # Figure 1: Sentence length distribution
    def plot_sentence_length_distribution(
        self,
        word_counts_by_group: Mapping[str, Sequence[int]],
        filename: str = "fig1_sentence_length_distribution.png",
    ) -> Path:
        """``word_counts_by_group``: e.g. {"Standard Bangla": [...], "Chittagonian": [...], "Sylheti": [...]}"""
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax_hist, ax_kde = axes

        for group, counts in word_counts_by_group.items():
            color = _PALETTE.get(group)
            counts = np.asarray(counts)
            if counts.size == 0:
                continue
            ax_hist.hist(counts, bins=30, alpha=0.45, label=group, color=color, density=True)
            sns.kdeplot(counts, ax=ax_kde, label=group, color=color, fill=True, alpha=0.2, linewidth=2)

        ax_hist.set_title("Sentence Word-Count Histogram")
        ax_hist.set_xlabel("Sentence length (words)")
        ax_hist.set_ylabel("Density")
        ax_hist.legend()

        ax_kde.set_title("Sentence Word-Count KDE")
        ax_kde.set_xlabel("Sentence length (words)")
        ax_kde.set_ylabel("Density")
        ax_kde.legend()

        fig.suptitle("Sentence Length Distribution: Chittagonian, Sylheti & Standard Bangla", fontsize=13)
        fig.tight_layout()
        return _save(fig, self.config.paths.figures_dir / filename)

   
    # Figure 2: Source vs. target length ratio scatter
    def plot_length_ratio_scatter(
        self,
        standard_word_counts: Sequence[int],
        dialect_word_counts: Sequence[int],
        dialect_label: str = "Dialect",
        filename: str = "fig2_length_ratio_scatter.png",
    ) -> Path:
        th = self.config.thresholds
        x = np.asarray(standard_word_counts, dtype=np.float64)
        y = np.asarray(dialect_word_counts, dtype=np.float64)
        mask = (x > 0) & (y > 0)
        x, y = x[mask], y[mask]

        fig = plt.figure(figsize=(8, 8))
        gs = GridSpec(4, 4, hspace=0.05, wspace=0.05)
        ax_main = fig.add_subplot(gs[1:4, 0:3])
        ax_top = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

        use_hexbin = len(x) > 1500
        if use_hexbin:
            hb = ax_main.hexbin(x, y, gridsize=40, cmap="viridis", mincnt=1, bins="log")
            fig.colorbar(hb, ax=ax_right, orientation="vertical", fraction=0.15, pad=0.3, label="log10(count)")
        else:
            ax_main.scatter(x, y, s=10, alpha=0.4, color=_PALETTE.get(dialect_label, "#2980B9"))

        max_val = float(max(x.max() if x.size else 1, y.max() if y.size else 1)) * 1.05
        line_x = np.linspace(0, max_val, 200)
        ax_main.plot(line_x, line_x, "k--", linewidth=1.2, label="y = x")
        ax_main.plot(line_x, th.min_length_ratio * line_x, color="red", linestyle=":", linewidth=1.2,
                     label=f"y = {th.min_length_ratio}x")
        ax_main.plot(line_x, th.max_length_ratio * line_x, color="red", linestyle=":", linewidth=1.2,
                     label=f"y = {th.max_length_ratio}x")
        ax_main.set_xlim(0, max_val)
        ax_main.set_ylim(0, max_val)
        ax_main.set_xlabel("Standard Bangla word count")
        ax_main.set_ylabel(f"{dialect_label} word count")
        ax_main.legend(loc="upper left", fontsize=8)

        if x.size:
            sns.kdeplot(x=x, ax=ax_top, color=_PALETTE.get("Standard Bangla", "#2C3E50"), fill=True)
        if y.size:
            sns.kdeplot(y=y, ax=ax_right, color=_PALETTE.get(dialect_label, "#2980B9"), fill=True)
        ax_top.axis("off")
        ax_right.axis("off")

        fig.suptitle(f"Standard Bangla vs. {dialect_label} Length Ratio", fontsize=13, y=0.98)
        return _save(fig, self.config.paths.figures_dir / filename)

    
    # Figure 3: Zipfian vocabulary rank vs frequency
    def plot_zipfian_distribution(
        self,
        tokens_by_group: Mapping[str, Sequence[str]],
        filename: str = "fig3_zipfian_distribution.png",
        top_k: int = 5000,
    ) -> Path:
        fig, ax = plt.subplots(figsize=(8, 6))
        for group, tokens in tokens_by_group.items():
            counts = Counter(tokens)
            freqs = np.array(sorted(counts.values(), reverse=True))[:top_k]
            if freqs.size == 0:
                continue
            ranks = np.arange(1, freqs.size + 1)
            ax.plot(ranks, freqs, label=group, color=_PALETTE.get(group), linewidth=1.6)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Token rank (log scale)")
        ax.set_ylabel("Token frequency (log scale)")
        ax.set_title("Zipfian Vocabulary Distribution: Rank vs. Frequency")
        ax.legend()
        fig.tight_layout()
        return _save(fig, self.config.paths.figures_dir / filename)

    # Figure 4: Cross-dialect lexical overlap
    def plot_lexical_overlap(
        self,
        vocab_by_group: Mapping[str, Iterable[str]],
        filename: str = "fig4_lexical_overlap.png",
    ) -> Path:
        groups = list(vocab_by_group.keys())
        if len(groups) != 3:
            raise ValueError(f"plot_lexical_overlap expects exactly 3 groups, got {len(groups)}: {groups}")
        sets = [set(vocab_by_group[g]) for g in groups]

        fig, ax = plt.subplots(figsize=(7, 7))
        if _HAS_VENN:
            venn3(sets, set_labels=groups, ax=ax)
        else:
            self._draw_manual_venn3(ax, sets, groups)
        ax.set_title("Cross-Dialect Lexical Overlap")
        fig.tight_layout()
        return _save(fig, self.config.paths.figures_dir / filename)

    def _draw_manual_venn3(self, ax: plt.Axes, sets: List[set], labels: List[str]) -> None:
        """Fallback 3-circle Venn (not area-proportional, but numerically
        annotated) used when ``matplotlib_venn`` is not installed."""
        A, B, C = sets
        centers = [(-0.6, 0.35), (0.6, 0.35), (0.0, -0.5)]
        colors = ["#E67E22", "#2980B9", "#27AE60"]
        radius = 1.1
        for (cx, cy), color, label in zip(centers, colors, labels):
            circle = Circle((cx, cy), radius, alpha=0.30, color=color, label=label)
            ax.add_patch(circle)

        only_a = len(A - B - C)
        only_b = len(B - A - C)
        only_c = len(C - A - B)
        ab = len(A & B - C)
        ac = len(A & C - B)
        bc = len(B & C - A)
        abc = len(A & B & C)

        annotations = {
            (-1.0, 0.55): only_a,
            (1.0, 0.55): only_b,
            (0.0, -1.05): only_c,
            (0.0, 0.55): ab,
            (-0.5, -0.35): ac,
            (0.5, -0.35): bc,
            (0.0, 0.0): abc,
        }
        for (x, y), val in annotations.items():
            ax.text(x, y, str(val), ha="center", va="center", fontsize=11, fontweight="bold")

        # Set-name labels are placed well clear of the circles (outside
        # their bounding box) so they never collide with the count
        # annotations inside the diagram: A/B labels sit above the top two
        # circles, C's label sits below the bottom circle.
        label_positions = [
            (centers[0][0] - 0.3, centers[0][1] + radius + 0.2),  # A (top-left)
            (centers[1][0] + 0.3, centers[1][1] + radius + 0.2),  # B (top-right)
            (centers[2][0], centers[2][1] - radius - 0.2),  # C (bottom)
        ]
        v_aligns = ["bottom", "bottom", "top"]
        for (lx, ly), label, va in zip(label_positions, labels, v_aligns):
            ax.text(lx, ly, label, ha="center", va=va, fontsize=11, fontweight="bold")

        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.3, 2.1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.text(
            0, -2.2,
            "(matplotlib_venn not installed: fallback schematic, not area-proportional)",
            ha="center", fontsize=7, style="italic", color="gray",
        )

   
    # Figure 5: Character & Unicode anomaly audit
    def plot_anomaly_audit(
        self,
        anomaly_counts: Mapping[str, int],
        filename: str = "fig5_anomaly_audit.png",
    ) -> Path:
        items = sorted(anomaly_counts.items(), key=lambda kv: kv[1])
        labels = [self._prettify_key(k) for k, _ in items]
        values = [v for _, v in items]

        fig, ax = plt.subplots(figsize=(9, max(3.5, 0.5 * len(items))))
        bars = ax.barh(labels, values, color=sns.color_palette("crest", len(items)))
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + max(values, default=1) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:,}", va="center", fontsize=9)
        ax.set_xlabel("Anomalies cleaned (count)")
        ax.set_title("Character & Unicode Anomaly Audit")
        fig.tight_layout()
        return _save(fig, self.config.paths.figures_dir / filename)

    @staticmethod
    def _prettify_key(key: str) -> str:
        return key.replace("_", " ").title()

   
    # Figure 6: Annotator agreement / disagreement distribution
    def plot_annotator_agreement(
        self,
        agreement_by_dialect: Mapping[str, Mapping[str, float]],
        filename: str = "fig6_annotator_agreement.png",
    ) -> Path:
        """``agreement_by_dialect``, e.g.::

            {
                "Chittagonian": {"Completely Agree": 66.66, "Spelling Error": 2.67,
                                  "Word Disagree": 30.00, "Completely Disagree": 0.67},
                "Sylheti": {"Completely Agree": 72.00, "Spelling Error": 3.67,
                            "Word Disagree": 22.67, "Completely Disagree": 1.66},
            }
        """
        categories = ["Completely Agree", "Spelling Error", "Word Disagree", "Completely Disagree"]
        colors = ["#27AE60", "#F1C40F", "#E67E22", "#C0392B"]
        dialects = list(agreement_by_dialect.keys())

        fig, ax = plt.subplots(figsize=(8, 5))
        bottoms = np.zeros(len(dialects))
        for cat, color in zip(categories, colors):
            vals = np.array([agreement_by_dialect[d].get(cat, 0.0) for d in dialects])
            bars = ax.bar(dialects, vals, bottom=bottoms, label=cat, color=color)
            for bar, v, b in zip(bars, vals, bottoms):
                if v >= 2:
                    ax.text(bar.get_x() + bar.get_width() / 2, b + v / 2, f"{v:.2f}%",
                            ha="center", va="center", fontsize=8, color="white" if cat != "Spelling Error" else "black")
            bottoms += vals

        ax.set_ylabel("Percentage of validated samples")
        ax.set_title("Annotator Agreement / Disagreement Distribution")
        ax.set_ylim(0, 100)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        return _save(fig, self.config.paths.figures_dir / filename)

    
    # Figure 7: Train/val/test cross-split similarity
    def plot_cross_split_similarity(
        self,
        char_ngram_jaccard: Sequence[float],
        word_ngram_jaccard: Sequence[float],
        leakage_threshold: Optional[float] = None,
        filename: str = "fig7_cross_split_similarity.png",
    ) -> Path:
        th = leakage_threshold if leakage_threshold is not None else self.config.thresholds.leakage_jaccard_threshold
        char_sims = np.asarray(char_ngram_jaccard, dtype=np.float64)
        word_sims = np.asarray(word_ngram_jaccard, dtype=np.float64)

        fig, (ax_box, ax_cdf) = plt.subplots(1, 2, figsize=(12, 5))

        data = pd.DataFrame(
            {
                "similarity": np.concatenate([char_sims, word_sims]),
                "metric": ["Char 4-gram Jaccard"] * len(char_sims) + ["Word 2-gram Jaccard"] * len(word_sims),
            }
        )
        sns.boxplot(
            data=data, x="metric", y="similarity", hue="metric", ax=ax_box,
            palette=["#2980B9", "#E67E22"], legend=False,
        )
        ax_box.axhline(th, color="red", linestyle="--", linewidth=1.2, label=f"Leakage threshold ({th:.2f})")
        ax_box.set_title("Max Test\u2192Train Similarity by Metric")
        ax_box.set_xlabel("")
        ax_box.set_ylabel("Jaccard similarity")
        ax_box.legend()

        for sims, label, color in (
            (char_sims, "Char 4-gram", "#2980B9"),
            (word_sims, "Word 2-gram", "#E67E22"),
        ):
            if sims.size == 0:
                continue
            sorted_vals = np.sort(sims)
            cdf = np.arange(1, sorted_vals.size + 1) / sorted_vals.size
            ax_cdf.plot(sorted_vals, cdf, label=label, color=color, linewidth=1.8)
        ax_cdf.axvline(th, color="red", linestyle="--", linewidth=1.2, label=f"Threshold ({th:.2f})")
        ax_cdf.set_xlabel("Max test\u2192train Jaccard similarity")
        ax_cdf.set_ylabel("Cumulative proportion of test set")
        ax_cdf.set_title("CDF of Cross-Split Similarity")
        ax_cdf.legend()

        fig.suptitle("Train / Validation / Test Leakage Diagnostic", fontsize=13)
        fig.tight_layout()
        return _save(fig, self.config.paths.figures_dir / filename)
